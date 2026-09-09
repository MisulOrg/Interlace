import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def continuant_numpy(coefficients):
    a = np.asarray(coefficients, dtype=np.float64)
    p = np.ones(a.shape[1:], dtype=np.float64)
    q = np.zeros_like(p)
    for coefficient in a[::-1]:
        p, q = coefficient * p + q, p
    with np.errstate(divide="ignore", invalid="ignore"):
        return p / q


def reciprocal(x, epsilon=0.5):
    return x / (mx.square(x) + epsilon * epsilon)


class TensorLinear(nn.Module):
    def __init__(self, width, rank=4, out_width=None):
        super().__init__()
        side = math.isqrt(width)
        if side * side != width:
            raise ValueError("width must be a perfect square")
        out_width=width if out_width is None else out_width
        out_side=math.isqrt(out_width)
        if out_side*out_side!=out_width:raise ValueError('output width must be a perfect square')
        self.side = side
        self.out_side=out_side
        self.left = mx.random.normal((rank, out_side, side)) / math.sqrt(side * rank)
        self.right = mx.random.normal((rank, side, out_side)) / math.sqrt(side)
        self.bias = mx.zeros((out_width,))

    def __call__(self, x):
        side, rank, out_side = self.side, self.left.shape[0], self.out_side
        matrix = x.reshape(-1, side, side)
        count = matrix.shape[0]
        # Combine examples and matrix columns into large GEMMs. The broadcast
        # oracle below instead dispatches many small matrix multiplications.
        columns = matrix.transpose(1, 0, 2).reshape(side, count * side)
        intermediate = self.left.reshape(rank * out_side, side) @ columns
        intermediate = intermediate.reshape(rank, out_side, count, side)
        intermediate = intermediate.transpose(0, 2, 1, 3).reshape(rank, count * out_side, side)
        result = mx.matmul(intermediate, self.right).reshape(rank, count, out_side, out_side).sum(0)
        return result.reshape(*x.shape[:-1],out_side*out_side) + self.bias

    def broadcast_reference(self, x):
        matrix = x.reshape(*x.shape[:-1], self.side, self.side)
        intermediate = mx.matmul(self.left, mx.expand_dims(matrix, -3))
        return mx.sum(mx.matmul(intermediate, self.right), axis=-3).reshape(*x.shape[:-1],self.out_side*self.out_side) + self.bias


class RationalTensor(nn.Module):
    def __init__(self, width, rank=4, depth=3, epsilon=0.5):
        super().__init__()
        self.coefficients = [TensorLinear(width, rank) for _ in range(depth)]
        self.epsilon = epsilon
        self.scale = mx.ones((width,)) * 0.1

    def __call__(self, x):
        value = self.coefficients[-1](x)
        for layer in self.coefficients[-2::-1]:
            value = layer(x) + reciprocal(value, self.epsilon)
        return self.scale * value
