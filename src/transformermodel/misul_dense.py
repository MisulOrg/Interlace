"""Parameter-matched dense Transformer control with the same task interfaces."""
import mlx.core as mx
import mlx.nn as nn
from .misul import MisulModel, FP8Linear, FP8Norm


class SwiGLU(nn.Module):
    def __init__(self, width, hidden, precision):
        super().__init__()
        self.up = FP8Linear(width, 2 * hidden, precision)
        self.down = FP8Linear(hidden, width, precision)

    def __call__(self, x):
        a, b = mx.split(self.up(x), 2, axis=-1)
        return self.down(nn.silu(a) * b)


class DenseAttention(nn.Module):
    def __init__(self, width, heads, precision):
        super().__init__()
        self.heads = heads
        self.qkv = FP8Linear(width, 3 * width, precision)
        self.output = FP8Linear(width, width, precision)

    def __call__(self, x, causal=True):
        batch, time, roles, width = x.shape
        q, k, v = mx.split(self.qkv(x).reshape(batch, time * roles, 3,
                           self.heads, width // self.heads).transpose(2, 0, 3, 1, 4), 3, axis=0)
        row = mx.arange(time * roles) // roles
        mask = row[:, None] >= row[None, :] if causal else None
        out = mx.fast.scaled_dot_product_attention(q[0], k[0], v[0],
                      scale=(width // self.heads) ** -.5, mask=mask)
        return self.output(out.transpose(0, 2, 1, 3).reshape(batch, time, roles, width))


class DenseBlock(nn.Module):
    def __init__(self, width, hidden, heads, precision):
        super().__init__()
        self.norm1 = FP8Norm(width, precision)
        self.norm2 = FP8Norm(width, precision)
        self.mixer = DenseAttention(width, heads, precision)
        self.ffn = SwiGLU(width, hidden, precision)

    def __call__(self, x, causal=True):
        x = x + self.mixer(self.norm1(x), causal)
        return x + self.ffn(self.norm2(x))


class DenseModel(MisulModel):
    def __init__(self, **config):
        if config.get('loops', 1) != 1:
            raise ValueError('dense control has one traversal of unique blocks')
        config['loops'] = 1
        super().__init__(**config)
        self.blocks = [DenseBlock(self.config['width'], self.config['hidden'],
                         self.config['heads'], self.precision)
                       for _ in range(self.config['layers'])]
