# Interlace technical supplement

This supplement documents the reference implementation, experiment controls
and reproduction procedures for Interlace-30M. The paper contains the
architecture, measured results and system card.

## Host and resource controls

Apple M5 Pro, 24 GiB unified memory, macOS 27.0, MLX 0.32.2. Project Python uses
NumPy 2.5.2 and tokenizers 0.23.1. Runs record actual device information, power
source, thermal warnings, source/data digests, model and optimizer dtypes,
peak MLX allocation and sampled physical process footprint. The external poller
requires 2 GiB maximum process footprint, 20 GiB free disk reserve, 40% system-memory
headroom and stops on 128 MiB new system swap. It is not a hard OS allocation cap.
No unrelated applications are stopped. Hardware/software and host contention
limit generalization of all timing results.

## Architecture comparison

`evidence/misul-dense-protocol.json` fixes the full dense control before its
training and evaluation. `experiments/train_misul_dense.py` is a frozen copy of
the original joint trainer with imports, constructor, loop count and provenance
adapted for the new model. The original production trainer remains unchanged.
`src/transformermodel/misul_dense.py` supplies full causal-row attention and
SwiGLU blocks; shared task adapters enable direct text/stream/Flow comparisons.
The BF16 dense model has 29,921,280 parameters, 0.358% more than Interlace's 29,814,640.
Both train on the same seed 1001, tokenizer, batch order, task mixture, NorMuon
recipe, schedule and 16,384-update budget. The completed BF16 Interlace checkpoint
is reused, with all original interrupted training time included. The dense
constructor initializes and replaces the inherited temporary hybrid blocks;
only the final dense blocks are retained, trained, counted and saved. That
initialization work remains included in its measured elapsed time.

This design compares the entire backbone under a common recipe. It does not
isolate rational gating, memory, selection or depth sharing, and does not prove
superiority to independently tuned model families. Seed 1001 is one observed
pair. The original official test was already open when the dense protocol was
registered; no dense architecture or hyperparameter is selected from its test
results. Raw evaluation retains each failure and the exact test-program digest.

## Reproduce

All device workloads must use the guard, for example:

```sh
PYTHONPATH=src .venv/bin/python -m transformermodel.safe_run \
  --report evidence/dense-reproduction-guard.json --seconds 1100 -- \
  .venv/bin/python experiments/train_misul_dense.py \
  --output artifacts/dense-reproduction --precision bf16 --memory-backend scan \
  --width 640 --hidden 1504 --layers 6 --heads 8 --context 128 --batch 4 \
  --program-batch 16 --steps 16384 --seconds 1000 --seed 1001 --eval-every 512
```

Use a new output directory for a new run; add `--resume` only for that same
run and unchanged recipe/source. The registered driver shows sequential bounded
segments. Training data are held separately and are not in the inference ZIP.
The dense evaluation wrapper uses the unchanged final evaluator with the dense
constructor. `experiments/benchmark_misul_architecture.py` evaluates frozen BF16
weights, with completed-device synchronization, separately recorded first-use
cost and nine subsequent repetitions per process. Two processes per arm run in
counterbalanced order, giving eighteen observations per workload. Greedy generation is uncached in both
arms. Timing includes generated-token host synchronization; it does not establish
cached serving throughput. Flow timing uses batch 8, length 6 and 1/4/8 passes.

## Precision and numerical history

The released model stores MXFP8 E4M3 values and E8M0 group scales. Working
parameters and optimizer arrays are BF16; the recurrent state and loss use FP16.
Some native operations accumulate internally in FP32. There is no claim of
strict eight-bit arithmetic. The trainable path and packed reload match on
text, streams and Flow probes. Model state, padding, scaling factors, embeddings,
norms and task/role/loop tables all count toward stored learned information.

The BF16 control originally overflowed before update 15,896. A preserved-state
recovery extended the permissible loss-scale floor, but its completed resumption
from 15,872 never exercised the fractional-scale branch. The numerical amendment,
failed qualification and direct-repeat witnesses remain in the evidence.
Small embedding optimizer differences prevent a full-state bit-exact replay
claim. Other inference tests and the saved finite-state checks passed.

The original memory fusion is retained in the measured Interlace checkpoint's
implementation. Its short pilot measures an implementation change separately
from the architecture comparison. Supporting experiments in multi-token
prediction and Fixed-Point Forcing remain separate from the release recipe.
Their original failures and adoption thresholds remain recorded.

## Naming and license

The architecture is Interlace, developed by Deyan Todorov at Misul. Frozen
training sources keep their original `misul_*` filenames so their recorded
checksums continue to identify the evaluated code. The `Interlace` public
Python alias points directly to that implementation. The distribution contains
the repository's Apache-2.0 license and a NOTICE crediting Deyan Todorov;
CITATION.cff supplies a suggested research citation. External datasets and
installed dependencies retain their own terms and are not relicensed here.

## Preparing the separate training corpus

The bundle includes the pinned download/preparation scripts and original
content manifests. To prepare a fresh corpus in an empty `data/` directory:

```sh
.venv/bin/python -m pip install -r requirements-research.txt
mkdir -p data
.venv/bin/python experiments/download_wikitext.py
PYTHONPATH=src .venv/bin/python -m transformermodel.safe_run \
  --report prepare-corpus-guard.json --seconds 180 -- \
  .venv/bin/python experiments/prepare_wikitext.py
mkdir -p data/misul-v2
cp model/tokenizer.json data/misul-v2/tokenizer.json
.venv/bin/python experiments/prepare_misul_test.py
```

Use the bundled tokenizer for the exact evaluated token IDs. The preparation
script's smaller legacy tokenizer is not used by the Interlace trainer.
`prepare_misul_data.py` separately documents how the 4,096-token tokenizer was
originally fitted. Compare prepared document contents to
`evidence/language-data-manifest.json` by split, title and content SHA-256;
absolute paths and preparation times naturally differ across machines.
The downloaded corpus retains its source license. It is not included in the
Apache-licensed model archive. Dense/BF16 control training states also remain
separate; the archive includes their exact weight hashes and full evaluations.

The included `tests/conftest.py` disables TF32 before MLX import for the strict
historical FP32 numerical oracles. This preserves their original qualification
conditions and tolerance bounds. The model implementations use their explicitly
specified BF16/FP16 or FP8 paths. Omitting that test configuration caused five
legacy tests to fail in an initial extraction; restoring it passes the same
unchanged tests. The original failed extraction evidence is retained locally.

## Capability and latency scope

The checkpoint generates corpus-style continuations; text remains repetitive
and fragmentary. General instruction following and factual question answering
have not been evaluated successfully. The structured task is prefix addition
modulo ten. Both backbones use the same stream interface, so their accuracy
difference measures the combined backbone, not the isolated effect of streams.

Flow is experimental. The released FP8 checkpoint reaches 1.95% exact
held-out sequences after eight passes and no exact longer-program sequences.
The BF16 architecture comparison records 0.00% for Interlace and 0.39% for
the dense control on the eight-pass held-out task. A stable candidate can be
incorrect. Full predictions are retained in the evaluation records.

Dense inference is faster in the measured workloads. Median 128-token forward
time is 6.14 ms for Interlace and 2.11 ms for dense; uncached 32-token generation
is 0.11 s versus 0.04 s, and eight-pass Flow is 30.60 ms versus 10.63 ms.
The next validation steps are multiple seeds, component and depth ablations,
broader corpora and longer contexts.


## Interpreting the backbone comparison

The parameter-matched dense control measures two complete backbones under a
common training recipe. Interlace applies six stored blocks twice; the dense
control applies six stored blocks once. The comparison therefore holds stored
parameters approximately equal, while reporting actual execution cost rather
than assuming equal computation. A dense shared-depth control would separately
test how much of the observed difference is associated with depth reuse.

The synthetic length-generalization result measures performance on a task
aligned with recurrent state updates. It supports that measured capability;
attributing the gain to a particular mechanism requires the additional controls.

## Flow interface and pass-count measurements

The implementation in `src/transformermodel/misul.py` constructs three roles:
input clues, candidate plus thought-role embedding, and the same candidate plus
output-role embedding. It runs the shared backbone and predicts through the
output role. This is one candidate-refinement process using three roles; it
does not maintain independently evolving Flow states for each stream.

The stored held-out evaluation contains 256 programs. Exact accuracy counts
whole correct sequences; state accuracy counts correct individual positions.

| Passes | FP8 exact sequences | FP8 state accuracy | BF16 exact sequences |
| --- | ---: | ---: | ---: |
| 1 | 1 / 256 | 30.69% | 0 / 256 |
| 2 | 4 / 256 | 45.92% | 0 / 256 |
| 4 | 4 / 256 | 45.20% | 0 / 256 |
| 8 | 5 / 256 | 43.93% | 0 / 256 |

Source: `evidence/misul-final-fp8.json` and `evidence/misul-final-bf16.json`.
The FP8 measurements show a small exact-solution increase and non-monotonic
state accuracy. Both checkpoints record zero exact longer-program solutions.
The existing results do not identify capacity, training allocation or refinement
objective as the cause. Greater width, additional shared-depth iterations and
more refinement passes change different aspects of the computation.

The separate Fixed-Point Forcing screen in
`evidence/misul-fixedpoint-comparison.json` did not meet its adoption criteria.
That result constrains the tested recipe, rather than every possible way to
train on recursively generated states.

## Next controlled experiment

The next proposed arm is a dense shared-depth backbone: retain the dense
control's six full-attention/SwiGLU blocks and task interfaces, but apply that
stack twice with loop identities. Compare it with both the single-pass dense
control and Interlace. Record exact parameter counts, twelve versus six block
evaluations, full validation loss, time to the fixed monitor target, stream
accuracy and synchronized inference latency.

Before training, freeze the source, fresh seed policy, optimizer recipe, data
order, resource limits and stop rule. Keep the existing 30M data and update
budget for the first diagnostic, then confirm any selected explanation across
fresh paired seeds. This is a proposed experiment; no looped-dense result is
reported here. Capacity and pass-count studies for Flow follow as separate
variables, with per-pass state accuracy and exact solutions both retained.
