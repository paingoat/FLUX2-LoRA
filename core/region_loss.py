"""Spatial flow-MSE weights: keep background supervision, emphasize edits."""
import math


def region_weights(mask, strength: float):
    """Accept Bx1xHxW masks at latent resolution; broadcast over channels."""
    if not math.isfinite(strength) or strength < 0:
        raise ValueError('region loss strength must be finite and nonnegative')
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError('expected Bx1xHxW supervision mask')
    if not mask.isfinite().all() or (mask < 0).any() or (mask > 1).any():
        raise ValueError('supervision mask must be finite and in [0, 1]')
    return 1.0 + strength * mask.detach().float()
