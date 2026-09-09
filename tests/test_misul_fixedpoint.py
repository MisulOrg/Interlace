import unittest
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformermodel.misul_fixedpoint import trajectory_feedback, loss_with_feedback
from transformermodel.safe_run import require_guard


class ToyFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = mx.array(.25, dtype=mx.bfloat16)
        self._observe = None

    def flow(self, clues, state, time, carry):
        if self._observe is not None:
            self._observe(state)
        return self.gain * state + .125 * carry + time[:, None, None]


class FixedPointTests(unittest.TestCase):
    def setUp(self):
        require_guard()
        self.model = ToyFlow()
        self.clues = mx.zeros((2, 3), dtype=mx.int32)
        self.targets = mx.array([[0, 2, 4], [1, 3, 5]], dtype=mx.int32)
        self.noise = mx.array(np.random.default_rng(1067).normal(size=(2, 3, 10)),
                              dtype=mx.bfloat16)
        self.time = mx.array([.25, .75], dtype=mx.bfloat16)
        self.start_fraction = mx.array([.25, .5], dtype=mx.bfloat16)

    def test_trajectory_matches_independent_oracle_and_is_detached(self):
        carry, history = trajectory_feedback(self.model, self.clues, self.targets,
            self.time, self.noise, self.start_fraction, depth=4, trace=True)
        time = np.array(self.time.tolist(), dtype=np.float64)
        start = time * np.array(self.start_fraction.tolist(), dtype=np.float64)
        delta = (time - start) / 4
        endpoint = np.eye(10)[np.array(self.targets)]
        state = (1 - start[:, None, None]) * np.array(self.noise.tolist()) + start[:, None, None] * endpoint
        feedback = np.zeros_like(state)
        for index in range(4):
            clock = start + index * delta
            logits = .25 * state + .125 * feedback + clock[:, None, None]
            exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
            state += (delta / (1 - clock))[:, None, None] * (exp / exp.sum(axis=-1, keepdims=True) - state)
            feedback = logits
        got = np.array(carry.tolist(), dtype=np.float64)
        self.assertLess(np.linalg.norm(got - feedback) / np.linalg.norm(feedback), .02)
        gradient = nn.value_and_grad(self.model, lambda m: trajectory_feedback(m,
            self.clues, self.targets, self.time, self.noise, self.start_fraction).sum())(self.model)[1]
        self.assertEqual(float(gradient['gain']), 0.)
        self.assertEqual(len(history), 4)

    def test_supervision_keeps_canonical_input_and_endpoint_is_finite(self):
        seen = []
        self.model._observe = seen.append
        carry, history = trajectory_feedback(self.model, self.clues, self.targets,
            self.time, self.noise, self.start_fraction, trace=True)
        mask = mx.ones(self.targets.shape, dtype=mx.bool_)
        loss = loss_with_feedback(self.model, self.clues, self.targets, mask,
                                  self.time, self.noise, carry)
        endpoint = mx.eye(10, dtype=mx.bfloat16)[self.targets]
        canonical = (1 - self.time[:, None, None]) * self.noise + self.time[:, None, None] * endpoint
        self.assertTrue(bool(mx.array_equal(seen[-1], canonical)))
        self.assertFalse(bool(mx.array_equal(seen[-1], history[-1][0])))
        self.assertTrue(bool(mx.isfinite(loss)))
        edge_time = mx.array([0., 1.], dtype=mx.bfloat16)
        edge_fraction = mx.ones((2,), dtype=mx.bfloat16)
        edge, trace = trajectory_feedback(self.model, self.clues, self.targets,
            edge_time, self.noise, edge_fraction, trace=True)
        self.assertTrue(bool(mx.all(mx.isfinite(edge))))
        self.assertTrue(bool(mx.array_equal(trace[0][0], trace[-1][0])))


if __name__ == '__main__':
    unittest.main()
