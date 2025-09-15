# autoregressive/__init__.py

"""
Autoregressive (PixelCNN+Transformer) package.

What this package provides
--------------------------
- :func:`build_ar_model` (in :mod:`autoregressive.models`):
    Builds a conditional autoregressive image model that predicts pixels in [0, 1]
    given an input one-hot label vector. The model compiles with BCE and is ready
    for training or weight loading.

- :class:`AutoregressivePipeline` (in :mod:`autoregressive.pipeline`):
    High-level orchestration for training, checkpointing, and synthesizing
    per-class samples to disk. Synthesis writes files under
    ``ARTIFACTS/synthetic`` using the standard contract:

      * ``gen_class_{k}.npy``  – images for class k in [0, 1], shape (N, H, W, 1)
      * ``labels_class_{k}.npy`` – integer labels (k) shape (N,)

    This contract is consumed by the common evaluator so results are comparable
    with GAN / VAE runs.

- :func:`save_grid_from_model` (optional, in :mod:`autoregressive.sample`):
    Convenience helper to render a quick PNG grid (one sample per class) for logs.

Typical usage
-------------
>>> from autoregressive import build_ar_model, AutoregressivePipeline
>>> model = build_ar_model(img_shape=(40, 40, 1), num_classes=9)
>>> pipe = AutoregressivePipeline(cfg)
>>> model = pipe.train(x_train, y_train, x_val, y_val)
>>> x_s, y_s = pipe.synthesize(model)  # also saves per-class .npy files

The package is intentionally lightweight; heavy lifting lives in the submodules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Public API re-exports
from .models import build_ar_model
from .pipeline import AutoregressivePipeline

# Optional helper (keep import soft to avoid hard dependency)
try:
    from .sample import save_grid_from_model  # type: ignore
except Exception:  # pragma: no cover - helper is optional
    save_grid_from_model = None  # type: ignore

__all__ = [
    "build_ar_model",
    "AutoregressivePipeline",
    "save_grid_from_model",
]

# Simple package version (override in your build/CI if desired)
__version__ = "0.1.0"
