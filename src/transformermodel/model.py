import mlx.core as mx
import mlx.nn as nn
import math

from .rational import RationalTensor,TensorLinear


class DenseFFN(nn.Module):
    def __init__(self, width, expansion=4):
        super().__init__()
        self.up = nn.Linear(width, width * expansion)
        self.down = nn.Linear(width * expansion, width)

    def __call__(self, x):
        return self.down(nn.gelu(self.up(x)))


class TensorFFN(nn.Module):
    def __init__(self,width,rank=4):
        super().__init__()
        self.up=TensorLinear(width,rank,out_width=4*width)
        self.down=TensorLinear(4*width,rank,out_width=width)
        # Set the expected weight contribution to variance 1/3 for independent
        # unit-variance inputs. Biases start at zero; both pairs train directly.
        self.up.left=self.up.left/math.sqrt(3.)
        self.down.left=self.down.left/math.sqrt(3.)

    def __call__(self,x):return self.down(nn.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, width, heads, kind, rank, depth):
        super().__init__()
        self.norm1 = nn.RMSNorm(width)
        self.norm2 = nn.RMSNorm(width)
        self.attention = nn.MultiHeadAttention(width, heads)
        self.ffn = DenseFFN(width) if kind == "dense" else TensorFFN(width,rank) if kind=='tensor' else RationalTensor(width, rank, depth)

    def __call__(self, x):
        h = self.norm1(x)
        x = x + self.attention(h, h, h, mask="causal")
        return x + self.ffn(self.norm2(x))


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, width=256, layers=2, heads=4, block="dense", rank=4, depth=3, context=256, positions=True, loops=1, tied_embeddings=False, position_qat16=False):
        super().__init__()
        if block not in ("dense", "rational", "tensor"):
            raise ValueError(block)
        self.embedding = nn.Embedding(vocab_size, width)
        self.position = mx.random.normal((context, width)) * 0.01
        self.positions = positions
        self.position_qat16 = position_qat16
        self.layers = [Block(width, heads, block, rank, depth) for _ in range(layers)]
        if loops<1:raise ValueError('loops must be positive')
        self.loops=loops
        if loops>1:self.loop_embedding=mx.random.normal((loops,width))*.01
        self.tied_embeddings=tied_embeddings
        self.norm = nn.RMSNorm(width)
        if not tied_embeddings:self.output = nn.Linear(width, vocab_size, bias=False)

    def position_values(self):
        if not self.position_qat16:return self.position
        rounded=self.position.astype(mx.float16).astype(mx.float32)
        return mx.stop_gradient(rounded)+(self.position-mx.stop_gradient(self.position))

    def features(self, tokens, embedding_weight=None):
        h = self.embedding(tokens) if embedding_weight is None else embedding_weight[tokens]
        if self.positions:
            h = h + self.position_values()[:tokens.shape[1]]
        for iteration in range(self.loops):
            if self.loops>1:h=h+self.loop_embedding[iteration]
            for layer in self.layers:
                h = layer(h)
        return self.norm(h)

    def project(self,h,embedding_weight=None):
        if not self.tied_embeddings:return self.output(h)
        weight=self.embedding.weight if embedding_weight is None else embedding_weight
        return h@weight.T

    def __call__(self, tokens):
        table=self.embedding.weight if self.tied_embeddings else None
        return self.project(self.features(tokens,table),table)

    def answer(self, tokens):
        table=self.embedding.weight if self.tied_embeddings else None
        return self.project(self.features(tokens,table)[:, -1],table)
