# autoregressive/train.py

"""
Training utilities for the conditional autoregressive model (PixelCNN-style).

This module keeps the training loop small, explicit, and framework-idiomatic:
- tf.data pipelines for (image, one-hot label) pairs
- @tf.function-compiled train/val steps
- Early stopping on validation loss
- TensorBoard logging (optional)
- Robust checkpointing using Keras 3-friendly names (*.weights.h5)

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
from typing import Optional, Tuple, Dict

import numpy as np
import tensorflow as tf

# Model builder (assumed available in your repo)
from autoregressive.models import build_conditional_pixelcnn

# Optional shared loader used across modules
try:
    from common.data import load_dataset_npy  # type: ignore
except Exception:
    load_dataset_npy = None  # we'll fall back to raw .npy files


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
# Small helpers
# ---------------------------------------------------------------------
def _cfg_get(cfg: Dict, dotted: str, default=None):
    cur = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _to_float01(x: np.ndarray) -> np.ndarray:
    """Ensure float32 in [0,1] (handles 0..255)."""
    x = x.astype("float32", copy=False)
    if x.max() > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def _to_onehot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Accept (N,) ints or (N,K) one-hot; return (N,K) one-hot float32."""
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == num_classes:
        y1h = y
    else:
        y = y.astype(int).reshape(-1)
        y1h = np.eye(num_classes, dtype="float32")[y]
    return y1h.astype("float32", copy=False)


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

    @tf.function(reduce_retracing=True)
    def train_step(x, y1h):
        with tf.GradientTape() as tape:
            probs = model([x, y1h], training=True)
            loss  = bce(x, probs)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss_metric.update_state(loss)

    @tf.function(reduce_retracing=True)
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
        if val_loss < best_val - 1e-6:
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


# ---------------------------------------------------------------------
# Unified-CLI adapter
# ---------------------------------------------------------------------
def train(cfg: Dict) -> None:
    """
    Adapter for the unified `gencs` CLI.

    Expects keys (with reasonable fallbacks):
      IMG_SHAPE (H,W,C), NUM_CLASSES, DATA_DIR
      EPOCHS, BATCH_SIZE, PATIENCE, SAVE_EVERY, LR, FROM_LOGITS
      paths.artifacts or ARTIFACTS.autoregressive_checkpoints/summaries
    """
    # --- Shapes & classes
    H, W, C = tuple(_cfg_get(cfg, "IMG_SHAPE", _cfg_get(cfg, "img.shape", (40, 40, 1))))
    K = int(_cfg_get(cfg, "NUM_CLASSES", _cfg_get(cfg, "num_classes", 9)))
    img_shape = (H, W, C)

    # --- Hyperparameters
    epochs     = int(_cfg_get(cfg, "EPOCHS", 200))
    batch_size = int(_cfg_get(cfg, "BATCH_SIZE", 256))
    patience   = int(_cfg_get(cfg, "PATIENCE", 10))
    save_every = int(_cfg_get(cfg, "SAVE_EVERY", 25))
    lr         = float(_cfg_get(cfg, "LR", 2e-4))
    from_logits = bool(_cfg_get(cfg, "FROM_LOGITS", False))
    seed       = int(_cfg_get(cfg, "SEED", 42))
    tf.keras.utils.set_random_seed(seed)

    # --- Artifact paths
    artifacts_root = Path(_cfg_get(cfg, "paths.artifacts", "artifacts"))
    model_root     = artifacts_root / "autoregressive"
    ckpt_dir = Path(_cfg_get(cfg, "ARTIFACTS.autoregressive_checkpoints", model_root / "checkpoints"))
    sums_dir = Path(_cfg_get(cfg, "ARTIFACTS.autoregressive_summaries",   model_root / "summaries"))
    tb_dir   = Path(_cfg_get(cfg, "ARTIFACTS.autoregressive_tensorboard", model_root / "tensorboard"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sums_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data
    data_dir = Path(_cfg_get(cfg, "DATA_DIR", _cfg_get(cfg, "data.root", "data")))
    if load_dataset_npy is not None:
        x_tr, y_tr, x_va, y_va, x_te, y_te = load_dataset_npy(
            data_dir, img_shape, K, val_fraction=_cfg_get(cfg, "VAL_FRACTION", 0.5)
        )
    else:
        x_tr = np.load(data_dir / "train_data.npy")
        y_tr = np.load(data_dir / "train_labels.npy")
        x_te = np.load(data_dir / "test_data.npy")
        y_te = np.load(data_dir / "test_labels.npy")
        # Split test -> (val, test)
        n_val = int(len(x_te) * float(_cfg_get(cfg, "VAL_FRACTION", 0.5)))
        x_va, y_va = x_te[:n_val], y_te[:n_val]
        x_te, y_te = x_te[n_val:], y_te[n_val:]

    # Normalize and ensure shapes
    x_tr = _to_float01(x_tr).reshape((-1, H, W, C))
    x_va = _to_float01(x_va).reshape((-1, H, W, C))
    y_tr_1h = _to_onehot(y_tr, K)
    y_va_1h = _to_onehot(y_va, K)

    # --- Build datasets
    train_ds, val_ds = make_datasets(x_tr, y_tr_1h, x_va, y_va_1h, batch_size=batch_size)

    # --- Build model & optimizer
    model = build_conditional_pixelcnn(img_shape, K)
    opt   = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)

    # --- TensorBoard writer (optional)
    writer = make_writer(tb_dir)

    # --- Train
    print(f"[config] img_shape={img_shape} | classes={K} | epochs={epochs} | bs={batch_size} | lr={lr}")
    print(f"[paths]  ckpts={ckpt_dir.resolve()} | summaries={sums_dir.resolve()} | tb={tb_dir.resolve()}")

    fit_autoregressive(
        model=model,
        optimizer=opt,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=epochs,
        checkpoint_dir=ckpt_dir,
        writer=writer,
        patience=patience,
        save_every=save_every,
        from_logits=from_logits,
    )

    # Simple marker file so orchestrators know training completed
    (sums_dir / "train_done.txt").write_text("ok", encoding="utf-8")


__all__ = ["TrainConfig", "make_datasets", "make_writer", "fit_autoregressive", "train"]
