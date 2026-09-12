"""Residual-stream interventions (Sec 2.3, App C).

* ``ablate``  : h' = h - d d^T h            (projection ablation, every token, during generation)
* ``steer``   : h' = h + alpha * d           (addition / steering with a unit direction)
* ``subtract``: h' = h - alpha * v           (Syed-style adapter-base subtraction with a *raw* direction)

All three are forward hooks on a decoder block's output and are applied at a single layer
unless you register several. Use the context managers so hooks are always removed.
"""
import contextlib

import numpy as np
import torch

from .models import get_layer


def _as_tensor(direction, like):
    d = torch.as_tensor(np.asarray(direction, dtype=np.float32), device=like.device, dtype=like.dtype)
    return d


def make_ablation_hook(unit_direction):
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        d = _as_tensor(unit_direction, h)
        d = d / (d.norm() + 1e-12)
        h2 = h - (h @ d).unsqueeze(-1) * d
        return (h2,) + output[1:] if isinstance(output, tuple) else h2
    return hook


def make_steering_hook(unit_direction, alpha):
    def hook(module, inputs, output):
        if alpha == 0:
            return output
        h = output[0] if isinstance(output, tuple) else output
        h2 = h + float(alpha) * _as_tensor(unit_direction, h)
        return (h2,) + output[1:] if isinstance(output, tuple) else h2
    return hook


def make_subtraction_hook(raw_direction, alpha=1.0):
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h2 = h - float(alpha) * _as_tensor(raw_direction, h).view(1, 1, -1)
        return (h2,) + output[1:] if isinstance(output, tuple) else h2
    return hook


@contextlib.contextmanager
def hooked(model, layer_to_hook):
    """``layer_to_hook``: dict {layer_idx: hook_fn}. Registers on entry, removes on exit."""
    handles = [get_layer(model, l).register_forward_hook(fn) for l, fn in layer_to_hook.items()]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def ablate(model, layer, unit_direction):
    return hooked(model, {layer: make_ablation_hook(unit_direction)})


def steer(model, layer, unit_direction, alpha):
    return hooked(model, {layer: make_steering_hook(unit_direction, alpha)})


def subtract(model, layer, raw_direction, alpha=1.0):
    return hooked(model, {layer: make_subtraction_hook(raw_direction, alpha)})


# ---- controls (random / orthogonal same-norm directions) ---------------------------
def random_same_norm(direction, seed=0, orthogonal_to=None):
    rng = np.random.default_rng(seed)
    d = np.asarray(direction, dtype=np.float64)
    r = rng.normal(size=d.shape)
    if orthogonal_to is not None:
        u = np.asarray(orthogonal_to, dtype=np.float64)
        u = u / max(np.linalg.norm(u), 1e-12)
        r = r - np.dot(r, u) * u
    r = r / max(np.linalg.norm(r), 1e-12) * np.linalg.norm(d)
    return r.astype(np.float32)
