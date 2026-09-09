"""Bounded Fixed-Point Forcing prefill, adapted from FRM v3 (2606.29150).

Training-only prefill starts from a reference/noise interpolation, follows the
model's own trajectory, and detaches its feedback. The supervised input remains
the original canonical interpolation. This module does not change inference.
"""
import mlx.core as mx
import mlx.nn as nn


def trajectory_feedback(model, clues, targets, time, noise, start_fraction,
                        depth=4, trace=False):
    if not 1 <= depth <= 16 or start_fraction.shape != time.shape:
        raise ValueError('invalid bounded feedback trajectory')
    endpoint = mx.eye(10, dtype=mx.bfloat16)[targets]
    start = time * start_fraction
    span = time - start
    state = (1 - start[:, None, None]) * noise + start[:, None, None] * endpoint
    carry = mx.zeros_like(state)
    history = []
    for index in range(depth):
        clock = start + span * (index / depth)
        logits = model.flow(clues, state, clock, carry)
        posterior = mx.softmax(logits.astype(mx.float16), axis=-1).astype(mx.bfloat16)
        # dt/(1-clock), rearranged to avoid cancellation when clocks round to 1.
        # A zero-span interval at time 1 leaves the state unchanged.
        denominator = depth * (1 - time) + (depth - index) * span
        fraction = span / mx.where(denominator == 0, mx.ones_like(denominator), denominator)
        state = mx.stop_gradient(state + fraction[:, None, None] * (posterior - state))
        carry = mx.stop_gradient(logits)
        if trace:
            history.append((state, clock, carry))
    mx.eval(carry, state)
    return (carry, history) if trace else carry


def loss_with_feedback(model, clues, targets, mask, time, noise, feedback):
    endpoint = mx.eye(10, dtype=mx.bfloat16)[targets]
    t = time[:, None, None]
    canonical = (1 - t) * noise + t * endpoint
    logits = model.flow(clues, canonical, time, mx.stop_gradient(feedback))
    losses = nn.losses.cross_entropy(logits.astype(mx.float16), targets, reduction='none')
    return mx.sum(losses * mask) / mx.sum(mask)
