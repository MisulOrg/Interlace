"""Two-token lookahead experiment with one shared Misul backbone.

This is a local auxiliary-objective variant motivated by Gloeckle et al.,
arXiv:2404.19737, not a reconstruction of Magic's undisclosed training recipe.
The next-token path is unchanged. A residual linear head predicts token t+2
from the same causal representation used for t+1. Its weights and cost count.
"""
import mlx.core as mx
import mlx.nn as nn
from .misul import MisulModel, FP8Linear, fp8_matmul


class MisulMTP(MisulModel):
    def __init__(self, lookahead=2, **configuration):
        if lookahead != 2:
            raise ValueError('this bounded experiment has two prediction heads')
        super().__init__(**configuration)
        self.future_projection = FP8Linear(self.config['width'], self.config['width'],
                                           self.precision, zero=True)
        self.config['lookahead'] = lookahead

    def lookahead_logits(self, tokens):
        if tokens.ndim != 2 or not 2 <= tokens.shape[1] <= self.config['context']:
            raise ValueError('lookahead requires a causal text window of at least two tokens')
        if tokens.size > 1024:
            raise ValueError('lookahead activation allocation bound exceeded')
        embedding = self.effective(self.embedding)
        role = self.effective(self.roles)[2]
        mode = self.effective(self.mode_embedding)[0]
        hidden = self.backbone((embedding[tokens] + role + mode)[:, :, None])[:, :, 0]
        future = hidden[:, :-1] + self.future_projection(hidden[:, :-1])
        def project(value):
            return fp8_matmul(value, self.embedding) if self.precision == 'mxfp8' else value @ self.embedding.T
        return project(hidden), project(future)


def lookahead_loss(model, inputs, targets, auxiliary_weight=.25):
    if inputs.shape != targets.shape or inputs.ndim != 2:
        raise ValueError('inputs and next-token targets must be aligned text windows')
    if auxiliary_weight == 0:
        return nn.losses.cross_entropy(model(inputs).astype(mx.float16), targets, reduction='mean')
    if not 0 < auxiliary_weight <= 1:
        raise ValueError('auxiliary weight must be in [0,1]')
    first, second = model.lookahead_logits(inputs)
    primary = nn.losses.cross_entropy(first.astype(mx.float16), targets, reduction='mean')
    # At position t, targets[t] is x[t+1], so targets[t+1] is x[t+2].
    auxiliary = nn.losses.cross_entropy(second.astype(mx.float16), targets[:, 1:], reduction='mean')
    return primary + auxiliary_weight * auxiliary
