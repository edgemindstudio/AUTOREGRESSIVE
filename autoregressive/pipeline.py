# autoregressive/pipeline.py

"""
Training + synthesis pipeline for the Conditional Autoregressive model
(PixelCNN-style masked convs + Transformer attention).

This module mirrors the structure used in the GAN and VAE projects so the
downstream evaluation & aggregation tooling works unchanged.

What you get
------------
- ARAutoregressivePipeline(cfg):
    * .train(x_train, y_train, x_val, y_val) -> tf.keras.Model
    * .synthesize(model: Optional[tf.keras.Model]) -> (x_synth, y_synth)

Conventions
-----------
- Images are (H, W, C) with values in [0, 1].
- Labels are one-hot, shape (N, num_classes).
- Checkpoints use Keras 3-friendly filenames: *.weights.h5
- Synthetic dumps:
    artifacts/autoregressive/synthetic/
        gen_class_<k>.npy         (float32 in [0,1], shape: [N_k, H, W, C])
        labels_class_<k>.npy      (int32 class IDs, shape: [N_k])
        x_synth.npy, y_synth.npy  (concatenated convenience arrays)

Notes
-----
- The autoregressive sampling loops over pixels in raster order. It’s vectorized
  over the batch dimension for reasonable speed; still, expect generation to be
  slower than feed-forward models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import tensorflow as tf

from autoregressive.models import build_ar_model


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------
def _as_float(x) -> float:
    """Convert scalars / 0-D tensors / arrays to float."""
    try:
        return float(x)
    except Exception:
        arr = np.asarray(x)
        return float(arr.reshape(-1)[0])


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _one_hot_from_ids(ids: np.ndarray, num_classes: int) -> np.ndarray:
    return tf.keras.utils.to_categorical(ids.astype(int), num_classes=num_classes).astype("float32")


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------
class ARAutoregressivePipeline:
    """Orchestrates train + synth for the conditional autoregressive model."""

    DEFAULTS = {
        "IMG_SHAPE": (40, 40, 1),
        "NUM_CLASSES": 9,
        "EPOCHS": 200,                # AR models can converge in fewer epochs; tune on HPC
        "BATCH_SIZE": 128,
        "LR": 2e-4,
        "BETA_1": 0.5,
        "NUM_FILTERS": 64,
        "NUM_LAYERS": 4,
        "NUM_HEADS": 4,
        "FF_MULT": 2,
        "LOG_EVERY": 25,              # save periodic epoch checkpoints
        "PATIENCE": 10,               # early stopping
        "SAMPLES_PER_CLASS": 1000,
        "ARTIFACTS": {
            "checkpoints": "artifacts/autoregressive/checkpoints",
            "synthetic": "artifacts/autoregressive/synthetic",
        },
    }

    def __init__(self, cfg: Dict):
        self.cfg = cfg or {}
        d = self.DEFAULTS

        # Hyperparameters / shapes
        self.img_shape: Tuple[int, int, int] = tuple(self.cfg.get("IMG_SHAPE", d["IMG_SHAPE"]))
        self.num_classes: int = int(self.cfg.get("NUM_CLASSES", d["NUM_CLASSES"]))
        self.epochs: int = int(self.cfg.get("EPOCHS", d["EPOCHS"]))
        self.batch_size: int = int(self.cfg.get("BATCH_SIZE", d["BATCH_SIZE"]))
        self.lr: float = float(self.cfg.get("LR", d["LR"]))
        self.beta_1: float = float(self.cfg.get("BETA_1", d["BETA_1"]))
        self.num_filters: int = int(self.cfg.get("NUM_FILTERS", d["NUM_FILTERS"]))
        self.num_layers: int = int(self.cfg.get("NUM_LAYERS", d["NUM_LAYERS"]))
        self.num_heads: int = int(self.cfg.get("NUM_HEADS", d["NUM_HEADS"]))
        self.ff_mult: int = int(self.cfg.get("FF_MULT", d["FF_MULT"]))
        self.log_every: int = int(self.cfg.get("LOG_EVERY", d["LOG_EVERY"]))
        self.patience: int = int(self.cfg.get("PATIENCE", d["PATIENCE"]))
        self.samples_per_class: int = int(self.cfg.get("SAMPLES_PER_CLASS", d["SAMPLES_PER_CLASS"]))

        # Artifacts
        arts = self.cfg.get("ARTIFACTS", d["ARTIFACTS"])
        self.ckpt_dir = Path(arts.get("checkpoints", d["ARTIFACTS"]["checkpoints"]))
        self.synth_dir = Path(arts.get("synthetic", d["ARTIFACTS"]["synthetic"]))
        _ensure_dir(self.ckpt_dir)
        _ensure_dir(self.synth_dir)

        # Optional logging callback: cb(epoch:int, train_loss, val_loss)
        self.log_cb = self.cfg.get("LOG_CB", None)

        # Build a fresh model (compiled)
        self.model: tf.keras.Model = build_ar_model(
            img_shape=self.img_shape,
            num_classes=self.num_classes,
            num_filters=self.num_filters,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            ff_multiplier=self.ff_mult,
            learning_rate=self.lr,
            beta_1=self.beta_1,
        )

    # ----------------------- Training -----------------------
    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> tf.keras.Model:
        """
        Fit the AR model.

        Args
        ----
        x_train : float32 array, shape (N, H, W, C), values in [0, 1]
        y_train : float32 array, shape (N, num_classes), one-hot labels
        x_val, y_val : optional validation sets (same format)

        Returns
        -------
        The trained model (also saved to checkpoints periodically).
        """
        H, W, C = self.img_shape
        assert x_train.shape[1:] == (H, W, C), "x_train shape mismatch"
        assert y_train.shape[1] == self.num_classes, "y_train must be one-hot"

        # Keras Callbacks
        callbacks: List[tf.keras.callbacks.Callback] = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.patience, restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.ckpt_dir / "AR_best.weights.h5"),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
            ),
        ]

        # Periodic manual checkpoint via custom callback
        class _PeriodicSaver(tf.keras.callbacks.Callback):
            def __init__(self, outer, log_every: int):
                super().__init__()
                self.outer = outer
                self.log_every = max(1, int(log_every))

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                e = epoch + 1
                if self.outer.log_cb is not None:
                    tr = logs.get("loss")
                    vl = logs.get("val_loss")
                    self.outer.log_cb(e, _as_float(tr) if tr is not None else None, _as_float(vl) if vl is not None else None)
                if e % self.log_every == 0 or e == 1:
                    path = self.outer.ckpt_dir / f"AR_epoch_{e:04d}.weights.h5"
                    self.model.save_weights(str(path))

        callbacks.append(_PeriodicSaver(self, self.log_every))

        # Fit
        history = self.model.fit(
            x=[x_train, y_train],
            y=x_train,                           # predict pixels of the input image
            validation_data=([x_val, y_val], x_val) if x_val is not None and y_val is not None else None,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            callbacks=callbacks,
            verbose=0,                           # logs handled by LOG_CB
        )

        # Save final checkpoint
        self.model.save_weights(str(self.ckpt_dir / "AR_last.weights.h5"))
        return self.model

    # ----------------------- Synthesis -----------------------
    def synthesize(self, model: Optional[tf.keras.Model] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a class-balanced synthetic dataset using autoregressive sampling.

        If `model` is None, a fresh model is built and (if found) the latest
        checkpoint is loaded from `self.ckpt_dir`.

        Returns
        -------
        x_synth : float32, shape (N_total, H, W, C), values in [0, 1]
        y_synth : float32, shape (N_total, num_classes), one-hot labels
        """
        if model is None:
            model = build_ar_model(
                img_shape=self.img_shape,
                num_classes=self.num_classes,
                num_filters=self.num_filters,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                ff_multiplier=self.ff_mult,
                learning_rate=self.lr,
                beta_1=self.beta_1,
            )
            # Try to load a recent checkpoint (prefer best, then last, then latest epoch)
            ckpt = self._latest_checkpoint()
            if ckpt is not None:
                model.load_weights(str(ckpt))

        H, W, C = self.img_shape
        per_class = self.samples_per_class

        xs, ys = [], []

        self.synth_dir.mkdir(parents=True, exist_ok=True)

        # Generate per-class batches to keep memory modest
        for cls in range(self.num_classes):
            labels = np.full((per_class,), cls, dtype=np.int32)
            y_onehot = _one_hot_from_ids(labels, self.num_classes)

            gen = self._sample_autoregressive(model, batch_size=per_class, y_onehot=y_onehot)
            gen = np.clip(gen, 0.0, 1.0).astype("float32")  # [0, 1]

            xs.append(gen)
            ys.append(y_onehot)

            # Save per-class dumps (and integer labels for convenience)
            np.save(self.synth_dir / f"gen_class_{cls}.npy", gen)
            np.save(self.synth_dir / f"labels_class_{cls}.npy", labels)

        x_synth = np.concatenate(xs, axis=0)
        y_synth = np.concatenate(ys, axis=0)

        # Convenience combined dumps
        np.save(self.synth_dir / "x_synth.npy", x_synth)
        np.save(self.synth_dir / "y_synth.npy", y_synth)

        print(f"[synthesize] {x_synth.shape[0]} samples ({per_class} per class) -> {self.synth_dir}")
        return x_synth, y_synth

    # ----------------------- Internals -----------------------
    def _latest_checkpoint(self) -> Optional[Path]:
        """
        Choose a checkpoint to load for synthesis:
            prefer AR_best.weights.h5, then AR_last.weights.h5,
            else the newest AR_epoch_*.weights.h5 (or legacy *.h5).
        """
        order = [
            self.ckpt_dir / "AR_best.weights.h5",
            self.ckpt_dir / "AR_last.weights.h5",
        ]
        # newest epoch checkpoint if present
        epoch_ckpts = sorted(self.ckpt_dir.glob("AR_epoch_*.weights.h5"))
        if epoch_ckpts:
            order.append(max(epoch_ckpts, key=lambda p: p.stat().st_mtime))

        for p in order:
            if p.exists():
                return p

        legacy = sorted(self.ckpt_dir.glob("AR_epoch_*.h5"))
        return max(legacy, key=lambda p: p.stat().st_mtime) if legacy else None

    def _sample_autoregressive(
        self,
        model: tf.keras.Model,
        batch_size: int,
        y_onehot: np.ndarray,
    ) -> np.ndarray:
        """
        Pixel-wise raster scan sampling (vectorized over the batch).

        Args
        ----
        model     : trained AR model (predicts p(x_ij|context,y) in [0,1])
        batch_size: number of images to sample
        y_onehot  : (B, num_classes), one-hot labels for the batch

        Returns
        -------
        imgs : (B, H, W, C) float32 in [0, 1]
        """
        H, W, C = self.img_shape
        imgs = np.zeros((batch_size, H, W, C), dtype=np.float32)

        # For each pixel (and channel), query the model and sample a Bernoulli
        # with the predicted probability.
        for i in range(H):
            for j in range(W):
                # Single forward pass gets p for ALL pixels; we only use (i, j, :)
                # This is simpler to read and still vectorized across the batch.
                probs = model.predict([imgs, y_onehot], verbose=0)  # (B, H, W, C)
                # Sample channel-wise (C is usually 1 for grayscale)
                pij = probs[:, i, j, :]
                # bernoulli sampling: U < p
                u = np.random.rand(batch_size, C).astype(np.float32)
                imgs[:, i, j, :] = (u < pij).astype(np.float32)

        return imgs


AutoregressivePipeline = ARAutoregressivePipeline
__all__ = ["ARAutoregressivePipeline", "AutoregressivePipeline"]
