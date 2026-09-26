"""MC-Dropout inference: eval mode everywhere except ``nn.Dropout`` modules, T stochastic passes."""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn


@contextmanager
def mc_dropout_mode(model: nn.Module):
    """Temporarily set ``model.eval()`` and re-enable only dropout layers; restores the previous mode."""
    was_training = model.training
    model.eval()
    drops = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    for m in drops:
        m.train()
    try:
        yield model
    finally:
        model.train(was_training)


@torch.no_grad()
def mc_predict(
    model: nn.Module, x: torch.Tensor, pad_mask: torch.Tensor, T: int = 50, generator_seed: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample mean and std of sigmoid probabilities over T dropout samples.

    Implemented by repeating the batch T times in one forward (fast on GPU) in chunks that keep memory
    bounded. ``generator_seed`` makes the dropout masks reproducible.
    """
    if generator_seed is not None:
        torch.manual_seed(generator_seed)
    b = x.shape[0]
    chunk = max(1, 8192 // max(1, b))  # passes per forward
    probs = []
    with mc_dropout_mode(model):
        done = 0
        while done < T:
            k = min(chunk, T - done)
            xr = x.repeat(k, 1, 1)
            mr = pad_mask.repeat(k, 1)
            probs.append(torch.sigmoid(model(xr, mr)).view(k, b))
            done += k
    pr = torch.cat(probs, 0)  # [T, B]
    return pr.mean(0), pr.std(0, unbiased=False)
