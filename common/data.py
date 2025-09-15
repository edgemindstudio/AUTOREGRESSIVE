# common/data.py

"""
Shared data utilities for all model families (GAN, VAE, Autoregressive).

This module standardizes how we:
- Load numpy datasets from disk (train/test splits).
- Normalize images to [0, 1] (training code may later map to [-1, 1] if needed).
- One-hot encode labels.
- Split the provided test set into (val, test).
- Build tf.data pipelines for efficient training/evaluation.
- Load synthetic per-class dumps produced by pipeline.synthesize(...) across models.

Conventions
-----------
- On disk we expect four files inside `data_dir`:
    train_data.npy, train_labels.npy, test_data.npy, test_labels.npy
- Image arrays are returned in HWC layout (N, H, W, C) and **[0, 1]** range.
  (Models that need [-1, 1] should call `to_minus1_1` themselves.)
- Labels are returned as one-hot vectors of length `num_classes`.
- Synthetic data loaders support two layouts:
    1) Aggregated per-class .npy files:
        gen_class_<k>.npy     (float images in [0, 1], shape (Nk, H, W, C))
        labels_class_<k>.npy  (int labels length Nk, all = k)
    2) Folder-of-files fallback (class_<k>/*.npy), each file is one sample.

This single module is imported by each project’s `app/main.py` or `pipeline.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, Iterable

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------
def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Ensure labels are one-hot (N, C). Accepts int labels (N,) or already one-hot (N, C)."""
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == num_classes:
        return y.astype(np.float32, copy=False)
    if y.ndim != 1:
        y = y.reshape(-1)
    return tf.keras.utils.to_categorical(y.astype(int), num_classes=num_classes).astype(np.float32)


def to_minus1_1(x01: np.ndarray) -> np.ndarray:
    """Map images from [0, 1] to [-1, 1]."""
    return (x01.astype(np.float32) - 0.5) * 2.0


def _to_01_hwc(x: np.ndarray, img_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Convert arbitrary incoming array to float32 (N, H, W, C) in [0, 1].
    Accepts (N, H, W) or (N, H, W, C) with values in {0..255} or [0,1].
    """
    H, W, C = img_shape
    x = np.asarray(x)
    x = x.astype(np.float32, copy=False)

    # Scale if appears to be 0..255
    if x.max() > 1.5:
        x = x / 255.0

    # Reshape to HWC if needed
    if x.ndim == 3:  # (N, H, W) -> (N, H, W, 1) if C==1
        if C != 1 or x.shape[1:] != (H, W):
            x = x.reshape((-1, H, W))  # trust caller's shape spec
        x = x[..., None]
    elif x.ndim == 4:
        # Try to coerce to requested shape (N, H, W, C)
        if x.shape[1:] != (H, W, C):
            x = x.reshape((-1, H, W, C))
    else:
        raise ValueError(f"Expected image array with 3 or 4 dims; got shape {x.shape}")

    # Clip numeric noise
    return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)


# ---------------------------------------------------------------------
# Loading the base dataset
# ---------------------------------------------------------------------
def load_dataset_npy(
    data_dir: str | Path,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    val_fraction: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset from `data_dir` expecting:
        - train_data.npy, train_labels.npy
        - test_data.npy,  test_labels.npy   (will be split into val/test)

    Returns
    -------
    (x_train01, y_train_1h, x_val01, y_val_1h, x_test01, y_test_1h)
    where all images are float32 in [0, 1], HWC and labels one-hot.
    """
    data_dir = Path(data_dir)
    H, W, C = img_shape

    x_train = np.load(data_dir / "train_data.npy")
    y_train = np.load(data_dir / "train_labels.npy")
    x_test  = np.load(data_dir / "test_data.npy")
    y_test  = np.load(data_dir / "test_labels.npy")

    x_train01 = _to_01_hwc(x_train, (H, W, C))
    x_test01  = _to_01_hwc(x_test,  (H, W, C))

    y_train_1h = one_hot(y_train, num_classes)
    y_test_1h  = one_hot(y_test,  num_classes)

    # Split provided test -> (val, test)
    n_val = int(len(x_test01) * float(val_fraction))
    x_val01, y_val_1h = x_test01[:n_val], y_test_1h[:n_val]
    x_test01, y_test_1h = x_test01[n_val:], y_test_1h[n_val:]

    return x_train01, y_train_1h, x_val01, y_val_1h, x_test01, y_test_1h


# ---------------------------------------------------------------------
# tf.data pipelines
# ---------------------------------------------------------------------
def as_tf_dataset(
    x: np.ndarray,
    y_1h: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
    shuffle_buffer: int = 10240,
) -> tf.data.Dataset:
    """
    Build a batched, prefetched tf.data pipeline from numpy arrays.
    """
    ds = tf.data.Dataset.from_tensor_slices((x, y_1h))
    if shuffle:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def make_tf_datasets(
    x_train01: np.ndarray,
    y_train_1h: np.ndarray,
    x_val01: np.ndarray,
    y_val_1h: np.ndarray,
    batch_size: int,
    shuffle_buffer: int = 10240,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Convenience wrapper to create (train_ds, val_ds).
    """
    train_ds = as_tf_dataset(x_train01, y_train_1h, batch_size, shuffle=True,  shuffle_buffer=shuffle_buffer)
    val_ds   = as_tf_dataset(x_val01,   y_val_1h,   batch_size, shuffle=False)
    return train_ds, val_ds


# ---------------------------------------------------------------------
# Synthetic data loaders (works for GAN/VAE/AR pipelines)
# ---------------------------------------------------------------------
def load_synth_balanced(
    synth_dir: str | Path,
    num_classes: int,
    img_shape: Optional[Tuple[int, int, int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load synthetic per-class dumps saved by our pipelines.

    Preferred format (fast path):
        synth_dir/gen_class_<k>.npy     -> images for class k in [0, 1]
        synth_dir/labels_class_<k>.npy  -> int labels (all = k)

    Fallback format (slower):
        synth_dir/class_<k>/*.npy       -> each file is one (H, W, C) sample

    Returns
    -------
    (x_synth01, y_synth_1h)
    """
    synth_dir = Path(synth_dir)
    xs, ys = [], []

    # Fast path: aggregated .npy per class
    fast_found = False
    for k in range(num_classes):
        x_path = synth_dir / f"gen_class_{k}.npy"
        y_path = synth_dir / f"labels_class_{k}.npy"
        if x_path.exists() and y_path.exists():
            xk = np.load(x_path)
            yk = np.load(y_path).reshape(-1)
            xs.append(xk)
            ys.append(yk)
            fast_found = True

    if fast_found:
        x = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        return x.astype(np.float32, copy=False), one_hot(y, num_classes)

    # Fallback: folder-of-files per class
    if img_shape is None:
        raise FileNotFoundError(
            f"No aggregated per-class files found under {synth_dir}. "
            f"For folder-of-files fallback, provide img_shape."
        )

    H, W, C = img_shape
    for k in range(num_classes):
        class_dir = synth_dir / f"class_{k}"
        if not class_dir.exists():
            continue
        files = sorted(class_dir.glob("*.npy"))
        for f in files:
            sample = np.load(f)
            sample = sample.astype(np.float32)
            # Coerce to HWC
            if sample.ndim == 2:
                sample = sample.reshape(H, W, 1)
            elif sample.ndim == 3 and sample.shape != (H, W, C):
                sample = sample.reshape(H, W, C)
            xs.append(sample)
            ys.append(k)

    if not xs:
        raise FileNotFoundError(f"No synthetic samples found under {synth_dir}")

    x = np.stack(xs, axis=0)
    y = np.array(ys, dtype=int)
    # Ensure [0, 1]
    if x.max() > 1.5:
        x = x / 255.0
    x = np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)
    return x, one_hot(y, num_classes)


__all__ = [
    "one_hot",
    "to_minus1_1",
    "load_dataset_npy",
    "as_tf_dataset",
    "make_tf_datasets",
    "load_synth_balanced",
]
