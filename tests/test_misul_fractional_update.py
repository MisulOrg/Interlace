import unittest
import mlx.core as mx
from mlx.utils import tree_flatten
from transformermodel.misul_fractional_update import FractionalCheckedUpdate
from transformermodel.misul_muon import Muon16
from transformermodel.safe_run import require_guard
from tests.test_misul_update import ScalarModel


class FractionalUpdateTests(unittest.TestCase):
    def setUp(self):
        require_guard()

    def test_fractional_retry_matches_direct_finite_scale(self):
        model = ScalarModel()
        optimizer = Muon16(model.parameters())
        loss = lambda m: 2 * mx.sum((m.weight.astype(mx.float16) - 1) * 40000)
        update = FractionalCheckedUpdate(model, optimizer, loss, 1)
        events = []
        def observe(event):
            self.assertEqual(int(optimizer.state['step']), 0)
            self.assertTrue(bool(mx.all(model.weight == 1)))
            self.assertTrue(all(bool(mx.all(v == 0)) for _, v in tree_flatten(optimizer.state)))
            events.append(event)
        value, norm = update(learning_rate_scale=mx.array(1., dtype=mx.bfloat16), on_overflow=observe)
        self.assertEqual([event['loss_scale'] for event in events], [1])
        self.assertEqual(update.loss_scale, .5)
        self.assertEqual(float(value), 0.)
        self.assertTrue(bool(mx.isfinite(norm)))
        control = ScalarModel()
        control_optimizer = Muon16(control.parameters())
        FractionalCheckedUpdate(control, control_optimizer, loss, .5)(
            learning_rate_scale=mx.array(1., dtype=mx.bfloat16))
        actual = tree_flatten([model.parameters(), optimizer.state])
        expected = tree_flatten([control.parameters(), control_optimizer.state])
        self.assertEqual([k for k, _ in actual], [k for k, _ in expected])
        self.assertTrue(all(bool(mx.array_equal(a, b)) for (_, a), (_, b) in zip(actual, expected)))
        self.assertNotIn(mx.float32, [v.dtype for _, v in actual])

    def test_singularity_stops_at_floor_without_state_mutation(self):
        model = ScalarModel()
        optimizer = Muon16(model.parameters())
        update = FractionalCheckedUpdate(model, optimizer,
            lambda m: mx.sum(m.weight.astype(mx.float16) / 0), 1)
        with self.assertRaises(FloatingPointError):
            update(learning_rate_scale=mx.array(1., dtype=mx.bfloat16))
        self.assertEqual(update.loss_scale, 1 / 128)
        self.assertEqual(int(optimizer.state['step']), 0)
        self.assertTrue(bool(mx.all(model.weight == 1)))


if __name__ == '__main__':
    unittest.main()
