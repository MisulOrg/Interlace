"""Check scaled gradients before a 16-bit Misul optimizer update.

An overflow retries the same batch at half the loss scale. Parameters and
optimizer state remain untouched until loss and gradient norm are finite.
The accepted scale only decreases and is recorded by the training driver.
"""
import math
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from .misul_muon import clip16


class CheckedUpdate:
    def __init__(self, model, optimizer, loss_function, loss_scale=128):
        if loss_scale not in (1, 2, 4, 8, 16, 32, 64, 128):
            raise ValueError('loss scale must be a power of two from 1 through 128')
        self.loss_scale = loss_scale
        self.retries = 0
        self.model = model
        self.optimizer = optimizer
        value_gradient = nn.value_and_grad(
            model, lambda m, *args: loss_function(m, *args[:-1]) * args[-1])

        def calculate(*args):
            value, gradients = value_gradient(model, *args)
            scale = args[-1]
            gradients = tree_map(lambda g: g / scale.astype(g.dtype), gradients)
            gradients, norm = clip16(gradients)
            return value / scale, gradients, norm

        self.calculate = mx.compile(calculate, inputs=[model.state], outputs=[model.state])

        def apply(gradients, learning_rate_scale):
            optimizer.update(model, gradients, learning_rate_scale)

        state = [model.state, optimizer.state]
        self.apply = mx.compile(apply, inputs=state, outputs=state)

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
            if self.loss_scale == 1:
                raise FloatingPointError('nonfinite loss or gradient at loss scale 1; no update applied')
            self.loss_scale //= 2
            self.retries += 1
