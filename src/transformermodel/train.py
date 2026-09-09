import time

PROCESS_START = time.perf_counter()

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np

from .data import KnowledgeData
from .model import LanguageModel


def evaluate(model, x, y, batch_size=256):
    correct, losses = 0, 0.0
    for start in range(0, len(y), batch_size):
        batch_x, batch_y = mx.array(x[start:start + batch_size]), mx.array(y[start:start + batch_size])
        logits = model.answer(batch_x)
        loss = nn.losses.cross_entropy(logits, batch_y, reduction="sum")
        right = mx.sum(mx.argmax(logits, axis=-1) == batch_y)
        mx.eval(loss, right)
        correct += int(right)
        losses += float(loss)
    return {"accuracy": correct / len(y), "loss": losses / len(y), "count": len(y)}


def source_digests():
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "src").rglob("*.py")) + list((root / "tests").rglob("*.py"))
    paths += list((root / "experiments").rglob("*.py"))
    paths += [root / "pyproject.toml", root / "uv.lock"]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file()}


def command_output(args):
    return subprocess.run(args, text=True, capture_output=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--block", choices=["dense", "rational"], default="dense")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--subjects", type=int, default=256)
    parser.add_argument("--relations", type=int, default=32)
    parser.add_argument("--objects", type=int, default=256)
    parser.add_argument("--kind", choices=["random", "structured"], default="random")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--data-seed", type=int, default=101)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--shuffle-labels", action="store_true")
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--no-positions", action="store_true")
    args = parser.parse_args()
    from .safe_run import require_guard
    require_guard()
    mx.set_memory_limit(1536*2**20);mx.set_cache_limit(256*2**20)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    data = KnowledgeData(args.subjects, args.relations, args.objects, args.data_seed, args.kind)
    train_x, train_y, train_ids = data.examples("train")
    val_x, val_y, val_ids = data.examples("validation")
    rng = np.random.default_rng(args.seed)
    if args.shuffle_labels:
        permutation = rng.permutation(args.objects)
        object_tokens = np.array([data.vocab[f"o{i}"] for i in range(args.objects)])
        replacement = dict(zip(object_tokens.tolist(), object_tokens[permutation].tolist()))
        train_y = np.array([replacement[int(y)] for y in train_y], dtype=np.int32)
    mx.random.seed(args.seed)
    model_args = dict(vocab_size=len(data.vocab), width=args.width, layers=args.layers,
                      heads=4, block=args.block, rank=args.rank, depth=args.depth, context=32,
                      positions=not args.no_positions)
    model = LanguageModel(**model_args)
    optimizer = optim.Adam(learning_rate=args.lr)
    optimizer.init(model.trainable_parameters())
    mx.eval(model.parameters(), optimizer.state)
    arrays = tree_flatten(model.parameters())
    manifest = {"configuration": vars(args), "model": model_args, "vocab": data.vocab,
                "data_sha256": data.digest, "source_sha256": source_digests(),
                "parameters": sum(a.size for _, a in arrays), "parameter_bytes": sum(a.nbytes for _, a in arrays),
                "python": platform.python_version(), "macos": platform.mac_ver(),
                "mlx": __import__("importlib.metadata", fromlist=["version"]).version("mlx"),
                "device": mx.device_info(), "precision_environment": os.environ.get("MLX_ENABLE_TF32", "default"),
                "command": [sys.executable, "-m", "transformermodel.train", *sys.argv[1:]],
                "power": command_output(["pmset", "-g", "batt"]),
                "swap_before": command_output(["sysctl", "vm.swapusage"]),
                "hypothesis": "rational factorization improves knowledge per byte or time to 90 percent recall",
                "threshold": 0.9, "phase": "exploratory", "all_training_local": True}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def loss_fn(m, x, y):
        return nn.losses.cross_entropy(m.answer(x), y, reduction="mean")

    loss_grad = nn.value_and_grad(model, loss_fn)

    def step(x, y):
        loss, gradients = loss_grad(model, x, y)
        gradients, norm = optim.clip_grad_norm(gradients, 1.0)
        optimizer.update(model, gradients)
        return loss, norm

    if args.compile:
        state = [model.state, optimizer.state, mx.random.state]
        step = mx.compile(step, inputs=state, outputs=state)

    tx, ty = mx.array(train_x), mx.array(train_y)
    mx.eval(tx, ty)
    start = PROCESS_START
    first_reached = None
    train_time = 0.0
    samples = 0
    status = "budget"
    first_half = np.where(train_ids[:, 0] < args.subjects // 2)[0]
    second_half = np.where(train_ids[:, 0] >= args.subjects // 2)[0]
    log = (output / "metrics.jsonl").open("w")
    print(json.dumps({"event": "start", "parameters": manifest["parameters"], "bytes": manifest["parameter_bytes"], "output": str(output)}), flush=True)
    for iteration in range(args.steps + 1):
        elapsed = time.perf_counter() - start
        if iteration % args.eval_every == 0 or elapsed >= args.seconds or iteration == args.steps:
            metrics = evaluate(model, val_x, val_y)
            if args.sequential:
                earlier = val_ids[:, 0] < args.subjects // 2
                metrics["earlier"] = evaluate(model, val_x[earlier], val_y[earlier])["accuracy"]
                metrics["later"] = evaluate(model, val_x[~earlier], val_y[~earlier])["accuracy"]
            elapsed = time.perf_counter() - start
            record = {"step": iteration, "elapsed": elapsed, "train_seconds": train_time, "samples": samples,
                      "validation": metrics, "peak_mlx_bytes": mx.get_peak_memory(),
                      "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            log.write(json.dumps(record) + "\n"); log.flush()
            print(json.dumps(record), flush=True)
            if metrics["accuracy"] >= 0.9 and first_reached is None:
                model.save_weights(str(output / "threshold.safetensors"))
                first_reached = time.perf_counter() - start
            if not np.isfinite(metrics["loss"]):
                status = "nonfinite"
                break
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 16 * 2**30:
                status = "memory_cap"
                break
            if elapsed >= args.seconds or iteration == args.steps:
                break
            if metrics["accuracy"] >= 0.99 and not args.sequential:
                status = "quality"
                break
        if args.sequential:
            pool = first_half if elapsed < args.seconds / 2 else second_half
            ids = rng.choice(pool, args.batch, replace=True)
            if elapsed >= args.seconds / 2:
                ids[:args.batch // 4] = rng.choice(first_half, args.batch // 4, replace=True)
        else:
            ids = rng.integers(0, len(train_y), size=args.batch)
        t0 = time.perf_counter()
        loss, norm = step(tx[mx.array(ids)], ty[mx.array(ids)])
        mx.eval(model.parameters(), optimizer.state, loss, norm)
        train_time += time.perf_counter() - t0
        samples += args.batch
        if not np.isfinite(float(loss)) or not np.isfinite(float(norm)):
            status = "nonfinite"
            break
    model.save_weights(str(output / "model.safetensors"))
    probe = mx.array(val_x[:16])
    expected = np.array(model.answer(probe))
    restored = LanguageModel(**model_args)
    restored.load_weights(str(output / "model.safetensors"))
    reload_equal = bool(np.array_equal(expected, np.array(restored.answer(probe))))
    summary = {"status": status, "elapsed": time.perf_counter() - start,
               "time_to_90": first_reached, "steps": iteration,
               "validation": evaluate(model, val_x, val_y),
               "train": evaluate(model, train_x[::4], train_y[::4]),
               "parameters": manifest["parameters"], "parameter_bytes": manifest["parameter_bytes"],
               "checkpoint_bytes": (output / "model.safetensors").stat().st_size,
               "reload_exact": reload_equal, "peak_mlx_bytes": mx.get_peak_memory(),
               "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "swap_after": command_output(["sysctl", "vm.swapusage"])}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    log.close()
    print(json.dumps({"event": "complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
