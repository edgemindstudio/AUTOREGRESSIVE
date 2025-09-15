# autoregressive/sample.py

"""
Small helpers to preview / export samples from a trained conditional
autoregressive model (PixelCNN-style).

The main entry point is:

    save_grid_from_ar(model, num_classes, img_shape,
                      n_per_class=1, path="artifacts/autoregressive/summaries/preview.png")

This will:
- Autoregressively sample `n_per_class` images per class label
- Assemble them into a tidy (num_classes × n_per_class) grid
- Save the result to the given `path` and return the Path

Notes
-----
- Images are generated in [0, 1] (Bernoulli outputs), shaped (H, W, C).
- Model signature is expected to be `model([images, onehot_labels]) -> probs`,
  where `images` is the partially filled batch (B, H, W, C) and `probs` are
  pixel probabilities with the same shape.
- Generation is vectorized over the batch but loops over pixels (raster scan).
- Intended for *previews* or small batches; large-scale synthesis should use
  `ARAutoregressivePipeline.synthesize()` for per-class dumps and caching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# ---------------------------------------------------------------------
# Core sampling (raster scan, vectorized across the batch)
# ---------------------------------------------------------------------
def _sample_autoregressive(
    model: tf.keras.Model,
    batch_size: int,
    num_classes: int,
    img_shape: tuple[int, int, int],
    class_ids: np.ndarray,
) -> np.ndarray:
    """
    Sample a batch of images conditioned on provided class ids.

    Args
    ----
    model       : trained AR model with signature probs = model([imgs, onehot])
    batch_size  : number of images to sample (= len(class_ids))
    num_classes : number of classes (for one-hot expansion)
    img_shape   : (H, W, C)
    class_ids   : (B,) int array with class indices in [0, num_classes)

    Returns
    -------
    imgs : (B, H, W, C) float32 in [0, 1]
    """
    H, W, C = img_shape
    assert batch_size == len(class_ids), "batch_size must equal len(class_ids)"

    imgs = np.zeros((batch_size, H, W, C), dtype=np.float32)
    y_onehot = tf.keras.utils.to_categorical(class_ids.astype(int), num_classes=num_classes).astype("float32")

    # Raster scan over pixels; for each position sample Bernoulli from predicted p
    for i in range(H):
        # One forward pass per row can be used; simplest (and still readable)
        probs_row = model.predict([imgs, y_onehot], verbose=0)  # (B, H, W, C)
        for j in range(W):
            pij = probs_row[:, i, j, :]  # (B, C)
            u = np.random.rand(batch_size, C).astype(np.float32)
            imgs[:, i, j, :] = (u < pij).astype(np.float32)

    return imgs


# ---------------------------------------------------------------------
# Public helper: save a tidy preview grid
# ---------------------------------------------------------------------
def save_grid_from_ar(
    model: tf.keras.Model,
    num_classes: int,
    img_shape: tuple[int, int, int],
    n_per_class: int = 1,
    path: str | Path = "artifacts/autoregressive/summaries/preview.png",
    seed: Optional[int] = 42,
) -> Path:
    """
    Generate a small, class-conditioned preview grid and save to disk.

    Layout
    ------
    Rows correspond to class IDs (0..num_classes-1).
    Columns are independent samples per class (n_per_class).

    Args
    ----
    model        : trained AR model
    num_classes  : number of classes
    img_shape    : (H, W, C)
    n_per_class  : number of samples per class (columns)
    path         : output PNG path
    seed         : optional RNG seed for reproducibility

    Returns
    -------
    out_path : Path to the saved image
    """
    if seed is not None:
        np.random.seed(seed)

    H, W, C = img_shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare conditioning labels: [0,0,...,1,1,..., ...] each repeated n_per_class times
    class_ids = np.repeat(np.arange(num_classes, dtype=np.int32), n_per_class)
    batch_size = len(class_ids)

    # Sample
    imgs = _sample_autoregressive(model, batch_size, num_classes, img_shape, class_ids)
    imgs = np.clip(imgs, 0.0, 1.0).astype("float32")

    # Assemble grid: rows = classes, cols = n_per_class
    fig_h = max(2, num_classes * 1.2)
    fig_w = max(2, n_per_class * 1.2)
    fig, axes = plt.subplots(num_classes, n_per_class, figsize=(fig_w, fig_h))
    # Normalize axes to 2D indexable array even for n_per_class == 1 or num_classes == 1
    if num_classes == 1 and n_per_class == 1:
        axes = np.array([[axes]])
    elif num_classes == 1:
        axes = axes.reshape(1, -1)
    elif n_per_class == 1:
        axes = axes.reshape(-1, 1)

    idx = 0
    for r in range(num_classes):
        for c in range(n_per_class):
            ax = axes[r, c]
            img = imgs[idx]
            idx += 1
            if C == 1:
                ax.imshow(img.squeeze(axis=-1), cmap="gray", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(img, vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"C{r}", rotation=0, labelpad=10, fontsize=9, va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


__all__ = ["save_grid_from_ar"]
