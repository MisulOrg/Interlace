import unittest
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from transformermodel.misul_muon import Muon16
from transformermodel.misul_update import CheckedUpdate


class ScalarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = mx.ones((4,), dtype=mx.bfloat16)


class CheckedUpdateTests(unittest.TestCase):
    def test_gradient_overflow_retries_without_mutating_state(self):
        model = ScalarModel()
        optimizer = Muon16(model.parameters())
        loss = lambda m: mx.sum(m.weight.astype(mx.float16) * 1024 - 1024)
        update = CheckedUpdate(model, optimizer, loss)
        observed = []

        def on_overflow(event):
            self.assertEqual(int(optimizer.state['step']), 0)
            self.assertTrue(bool(mx.all(model.weight == 1)))
            self.assertTrue(all(bool(mx.all(v == 0)) for _, v in tree_flatten(optimizer.state)))
            observed.append(event)

        value, norm = update(learning_rate_scale=mx.array(1., dtype=mx.bfloat16),
                             on_overflow=on_overflow)
        self.assertEqual([e['loss_scale'] for e in observed], [128, 64])
        self.assertEqual(update.loss_scale, 32)
        self.assertEqual(float(value), 0.)
        self.assertEqual(float(norm), 2048.)
        self.assertEqual(int(optimizer.state['step']), 1)
        control = ScalarModel()
        control_optimizer = Muon16(control.parameters())
        CheckedUpdate(control, control_optimizer, loss, loss_scale=1)(
            learning_rate_scale=mx.array(1., dtype=mx.bfloat16))
        for (_, got), (_, expected) in zip(tree_flatten(model.parameters()), tree_flatten(control.parameters())):
            self.assertTrue(bool(mx.array_equal(got, expected)))
        self.assertNotIn(mx.float32, [v.dtype for _, v in tree_flatten(optimizer.state)])

    def test_unrecoverable_gradient_does_not_advance_optimizer(self):
        model = ScalarModel()
        optimizer = Muon16(model.parameters())
        update = CheckedUpdate(model, optimizer,
                               lambda m: mx.sum(m.weight.astype(mx.float16) / 0), loss_scale=1)
        with self.assertRaises(FloatingPointError):
            update(learning_rate_scale=mx.array(1., dtype=mx.bfloat16))
        self.assertEqual(int(optimizer.state['step']), 0)
        self.assertTrue(bool(mx.all(model.weight == 1)))


if __name__ == '__main__':
    unittest.main()
