import unittest
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
from transformermodel.misul_mtp import MisulMTP, lookahead_loss


class MisulMTPTests(unittest.TestCase):
    def test_primary_path_causality_alignment_and_export(self):
        import tempfile
        mx.random.seed(1051)
        model = MisulMTP(vocab_size=64, width=32, hidden=64, layers=2, heads=2,
                         context=8, loops=2, precision='mxfp8', memory_backend='metal')
        tokens = mx.array([[1, 4, 7, 12, 20, 25]], dtype=mx.int32)
        targets = mx.array([[4, 7, 12, 20, 25, 31]], dtype=mx.int32)
        first, second = model.lookahead_logits(tokens)
        self.assertTrue(mx.array_equal(first, model(tokens)).item())
        future_changed = mx.array([[1, 4, 7, 33, 45, 61]], dtype=mx.int32)
        changed_first, changed_second = model.lookahead_logits(future_changed)
        self.assertTrue(mx.array_equal(first[:, :3], changed_first[:, :3]).item())
        self.assertTrue(mx.array_equal(second[:, :3], changed_second[:, :3]).item())
        # Independent host log-softmax verifies the auxiliary target offset.
        def ce(logits, labels):
            values = np.array(logits.tolist(), dtype=np.float64)
            values -= values.max(axis=-1, keepdims=True)
            logp = values - np.log(np.exp(values).sum(axis=-1, keepdims=True))
            return -np.take_along_axis(logp, np.array(labels)[..., None], axis=-1).mean()
        expected = ce(first, targets) + .25 * ce(second, np.array(targets)[:, 1:])
        self.assertLess(abs(float(lookahead_loss(model, tokens, targets)) - expected), .01)
        with tempfile.TemporaryDirectory() as folder:
            model.export(folder + '/release')
            reloaded = MisulMTP.load(folder + '/release')
            self.assertTrue(mx.array_equal(reloaded(tokens), first).item())
            self.assertTrue(mx.array_equal(reloaded.lookahead_logits(tokens)[1], second).item())

    def test_shared_backbone_receives_both_gradients(self):
        mx.random.seed(1052)
        model = MisulMTP(vocab_size=64, width=32, hidden=64, layers=2, heads=2,
                         context=8, loops=2, precision='bf16', memory_backend='scan')
        tokens = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)
        targets = mx.array([[2, 3, 4, 5, 6, 7]], dtype=mx.int32)
        def loss(which):
            def compute(m):
                heads = m.lookahead_logits(tokens)
                return nn.losses.cross_entropy(heads[which].astype(mx.float16),
                                                targets if which == 0 else targets[:, 1:],
                                                reduction='mean') * 128
            return compute
        _, first = nn.value_and_grad(model, loss(0))(model)
        _, second = nn.value_and_grad(model, loss(1))(model)
        _, total = nn.value_and_grad(model, lambda m: lookahead_loss(m, tokens, targets) * 128)(model)
        first, second, total = [dict(tree_flatten(g)) for g in (first, second, total)]
        error = norm = 0.
        for name in total:
            expected = np.array(first[name].tolist(), dtype=np.float64) + .25 * np.array(second[name].tolist(), dtype=np.float64)
            got = np.array(total[name].tolist(), dtype=np.float64)
            self.assertTrue(np.isfinite(got).all())
            error += float(np.square(got - expected).sum())
            norm += float(np.square(expected).sum())
        self.assertLess(np.sqrt(error / norm), .02)
        self.assertGreater(float(mx.sum(mx.abs(second['blocks.0.ffn.up.weight']))), 0.)
        self.assertGreater(float(mx.sum(mx.abs(total['future_projection.weight']))), 0.)


if __name__ == '__main__':
    unittest.main()
