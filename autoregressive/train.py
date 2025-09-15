# autoregressive/train.py

"""
Training utilities for the conditional autoregressive model (PixelCNN-style).

This module keeps the training loop small, explicit, and framework-idiomatic:
- tf.data pipelines for (image, one-hot label) pairs
- @tf.function-compiled train/val steps
- Early stopping on validation loss
- TensorBoard logging (optional)
- Robust checkpointing using Keras 3-friendly names (*.weights.h5)

Typical usage (inside your pipeline):

    from pathlib import Path
    import tensorflow as tf
    from autoregressive.models import build_conditional_pixelcnn
    from autoregressive.train import (
        make_datasets, fit_autoregressive, make_writer
    )

    model = build_conditional_pixelcnn(img_shape, num_classes)
    opt   = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

    ds_train, ds_val = make_datasets(x_train01, y_train_1h, x_val01, y_val_1h, batch_size=256)
    writer = make_writer("artifacts/tensorboard")   # optional

    fit_autoregressive(
        model=model,
        optimizer=opt,
        train_ds=ds_train,
        val_ds=ds_val,
        epochs=200,
        checkpoint_dir=Path("artifacts/autoregressive/checkpoints"),
        writer=writer,
        patience=10,
        save_every=25,
    )

Notes
-----
- The model is expected to be conditioned on labels: `model([x, y_onehot]) -> probs`,
  where probs ∈ [0, 1] has the same shape as x.
- Inputs `x_*` should be in [0, 1]. We use pixelwise BinaryCrossentropy.
- Labels `y_*` must be one-hot encoded (shape (N, num_classes)).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------
# Public configuration dataclass (optional syntactic sugar)
# ---------------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 256
    patience: int = 10
    save_every: int = 25
    # Loss: BCE over pixels (with probabilities, not logits)
    from_logits: bool = False


# ---------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------
def make_datasets(
    x_train01: np.ndarray,
    y_train_1h: np.ndarray,
    x_val01: np.ndarray,
    y_val_1h: np.ndarray,
    batch_size: int,
    shuffle_buffer: int = 10240,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Build shuffling, batched tf.data pipelines for training and validation.

    Args
    ----
    x_train01, x_val01 : float32 arrays in [0, 1], shape (N, H, W, C)
    y_train_1h, y_val_1h : one-hot labels, shape (N, num_classes)
    batch_size : batch size
    shuffle_buffer : buffer for shuffling the train dataset

    Returns
    -------
    (train_ds, val_ds)
    """
    def _ds(x, y, training=False):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if training:
            ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = _ds(x_train01, y_train_1h, training=True)
    val_ds   = _ds(x_val01,   y_val_1h,   training=False)
    return train_ds, val_ds


def make_writer(tensorboard_dir: str | Path | None) -> Optional[tf.summary.SummaryWriter]:
    """Create a TensorBoard writer if a directory is provided."""
    if not tensorboard_dir:
        return None
    tb_path = Path(tensorboard_dir)
    tb_path.mkdir(parents=True, exist_ok=True)
    return tf.summary.create_file_writer(str(tb_path))


# ---------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------
def fit_autoregressive(
    *,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int,
    checkpoint_dir: Path,
    writer: Optional[tf.summary.SummaryWriter] = None,
    patience: int = 10,
    save_every: int = 25,
    from_logits: bool = False,
) -> None:
    """
    Train an autoregressive conditional model with early stopping & checkpoints.

    Args
    ----
    model : expects `model([x, y_onehot], training=...) -> probs`
    optimizer : tf.keras optimizer
    train_ds, val_ds : tf.data datasets of (x, y_onehot)
    epochs : total epochs to run
    checkpoint_dir : where to save *.weights.h5 checkpoints
    writer : optional TensorBoard summary writer
    patience : early-stopping patience on validation loss
    save_every : save periodic epoch snapshots as AR_epoch_XXXX.weights.h5
    from_logits : if True, uses BCE(from_logits=True). Default False (sigmoid probs)

    Saves
    -----
    - AR_best.weights.h5  (lowest validation loss)
    - AR_last.weights.h5  (last epoch finished)
    - AR_epoch_XXXX.weights.h5 (periodic snapshots)
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)
    train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
    val_loss_metric   = tf.keras.metrics.Mean(name="val_loss")

    @tf.function
    def train_step(x, y1h):
        with tf.GradientTape() as tape:
            probs = model([x, y1h], training=True)
            loss  = bce(x, probs)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss_metric.update_state(loss)

    @tf.function
    def val_step(x, y1h):
        probs = model([x, y1h], training=False)
        loss  = bce(x, probs)
        val_loss_metric.update_state(loss)

    best_val = np.inf
    no_improve = 0

    # Try to resume if AR_last exists (optional / harmless)
    last_ckpt = checkpoint_dir / "AR_last.weights.h5"
    if last_ckpt.exists():
        try:
            model.load_weights(str(last_ckpt))
            print(f"[resume] Loaded {last_ckpt.name}")
        except Exception:
            print("[resume] Found AR_last but failed to load; training from scratch.")

    for epoch in range(1, epochs + 1):
        # --- Train
        train_loss_metric.reset_state()
        for xb, yb in train_ds:
            train_step(xb, yb)
        train_loss = float(train_loss_metric.result())

        # --- Validate
        val_loss_metric.reset_state()
        for xb, yb in val_ds:
            val_step(xb, yb)
        val_loss = float(val_loss_metric.result())

        # --- Log
        print(f"[epoch {epoch:05d}] train={train_loss:.4f} | val={val_loss:.4f}")
        if writer:
            with writer.as_default():
                tf.summary.scalar("loss/train_total", train_loss, step=epoch)
                tf.summary.scalar("loss/val_total", val_loss, step=epoch)
                writer.flush()

        # --- Periodic snapshot
        if (epoch == 1) or (epoch % save_every == 0):
            snap = checkpoint_dir / f"AR_epoch_{epoch:04d}.weights.h5"
            model.save_weights(str(snap))

        # --- Best checkpoint & early stopping
        if val_loss < best_val - 1e-6:  # small margin to reduce ties
            best_val  = val_loss
            no_improve = 0
            model.save_weights(str(checkpoint_dir / "AR_best.weights.h5"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[early-stop] No val improvement for {patience} epochs.")
                break

    # Always save a final "last" checkpoint
    model.save_weights(str(checkpoint_dir / "AR_last.weights.h5"))


__all__ = ["TrainConfig", "make_datasets", "make_writer", "fit_autoregressive"]
