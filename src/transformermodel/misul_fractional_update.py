"""Explicit numerical amendment: allow checked loss scales down to 1/128.

The original calculate/apply kernels are inherited unchanged. This extends only
the recovery branch after scale 1 fails; it changes no accepted update above 1.
"""
import math
import mlx.core as mx
from .misul_update import CheckedUpdate


class FractionalCheckedUpdate(CheckedUpdate):
    def __init__(self, model, optimizer, loss_function, loss_scale=128):
        if loss_scale not in tuple(2. ** power for power in range(-7, 8)):
            raise ValueError('loss scale must be a power of two from 1/128 to 128')
        super().__init__(model, optimizer, loss_function, max(1, loss_scale))
        self.loss_scale = loss_scale

    def __call__(self, *inputs, learning_rate_scale, on_overflow=None):
        while True:
            value, gradients, norm = self.calculate(
                *inputs, mx.array(self.loss_scale, dtype=mx.float16))
            mx.eval(value, gradients, norm)
            if math.isfinite(float(value)) and math.isfinite(float(norm)):
                self.apply(gradients, learning_rate_scale)
                mx.eval(self.model.parameters(), self.optimizer.state)
                return value, norm
            event = dict(loss_scale=self.loss_scale, loss=float(value),
                         gradient_norm=float(norm), retry=self.retries + 1)
            if on_overflow is not None:
                on_overflow(event)
            if self.loss_scale == 1 / 128:
                raise FloatingPointError('nonfinite loss or gradient at loss scale 1/128; no update applied')
            self.loss_scale /= 2
            self.retries += 1
