# autoregressive/sample.py
"""
Sampling helpers + unified-CLI synth entrypoint for conditional PixelCNN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

from .models import build_conditional_pixelcnn


# ----------------------------- small utils ----------------------------- #
def _cfg_get(cfg: dict, dotted: str, default=None):
    cur = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _to_uint8(img01: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img01 * 255.0), 0, 255).astype(np.uint8)


def _save_png(img01: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = img01
    if x.ndim == 3 and x.shape[-1] == 1:
        Image.fromarray(_to_uint8(x[..., 0]), mode="L").save(out_path)
    elif x.ndim == 3 and x.shape[-1] == 3:
        Image.fromarray(_to_uint8(x), mode="RGB").save(out_path)
    else:
        Image.fromarray(_to_uint8(x.squeeze()), mode="L").save(out_path)


# ----------------------------- checkpoints ----------------------------- #
def load_ar_from_checkpoints(
    ckpt_dir: Path | str,
    *,
    img_shape: Tuple[int, int, int],
    num_classes: int,
) -> tf.keras.Model:
    """Build model and try to load AR_best→AR_last. Fall back to random weights."""
    ckpt_dir = Path(ckpt_dir)
    H, W, C = img_shape
    model = build_conditional_pixelcnn(img_shape, num_classes)

    # Build variables (Keras 3)
    dummy_x = tf.zeros((1, H, W, C), dtype=tf.float32)
    dummy_y = tf.one_hot([0], depth=num_classes, dtype=tf.float32)
    _ = model([dummy_x, dummy_y], training=False)

    best = ckpt_dir / "AR_best.weights.h5"
    last = ckpt_dir / "AR_last.weights.h5"
    to_load = best if best.exists() else last
    if not to_load.exists():
        print(f"[ckpt][warn] no AR checkpoint in {ckpt_dir}; using random weights.")
        return model

    try:
        model.load_weights(str(to_load))
        print(f"[ckpt] loaded {to_load.name}")
    except Exception as e:
        print(f"[ckpt][warn] failed to load {to_load.name}: {e}\n→ continuing with random weights.")
    return model


# ----------------------------- sampling ----------------------------- #
def _sample_autoregressive(
    model: tf.keras.Model,
    *,
    class_ids: np.ndarray,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Raster-scan sampling; vectorized over batch, sequential over pixels."""
    if seed is not None:
        np.random.seed(int(seed))

    H, W, C = img_shape
    B = int(class_ids.shape[0])
    imgs = np.zeros((B, H, W, C), dtype=np.float32)
    onehot = tf.keras.utils.to_categorical(class_ids.astype(int), num_classes=num_classes).astype("float32")

    for i in range(H):
        for j in range(W):
            probs = model.predict([imgs, onehot], verbose=0)  # (B,H,W,C) in [0,1]
            pij = probs[:, i, j, :]                           # (B,C)
            u = np.random.rand(B, C).astype(np.float32)
            imgs[:, i, j, :] = (u < pij).astype(np.float32)

    return imgs


def save_grid_from_ar(
    model: tf.keras.Model,
    num_classes: int,
    img_shape: Tuple[int, int, int],
    n_per_class: int = 1,
    path: Path | str = "artifacts/autoregressive/summaries/preview.png",
    seed: Optional[int] = 42,
) -> Path:
    """Render a small grid for previews."""
    H, W, C = img_shape
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)

    class_ids = np.repeat(np.arange(num_classes, dtype=np.int32), n_per_class)
    imgs = _sample_autoregressive(model, class_ids=class_ids, img_shape=img_shape, num_classes=num_classes, seed=seed)
    imgs = np.clip(imgs, 0.0, 1.0).astype("float32")

    fig_h = max(2, num_classes * 1.2); fig_w = max(2, n_per_class * 1.2)
    fig, axes = plt.subplots(num_classes, n_per_class, figsize=(fig_w, fig_h))
    if num_classes == 1 and n_per_class == 1: axes = np.array([[axes]])
    elif num_classes == 1: axes = axes.reshape(1, -1)
    elif n_per_class == 1: axes = axes.reshape(-1, 1)

    idx = 0
    for r in range(num_classes):
        for c in range(n_per_class):
            ax = axes[r, c]; img = imgs[idx]; idx += 1
            if C == 1: ax.imshow(img[:, :, 0], cmap="gray", vmin=0.0, vmax=1.0)
            else:      ax.imshow(img, vmin=0.0, vmax=1.0)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0: ax.set_ylabel(f"C{r}", rotation=0, labelpad=10, fontsize=9, va="center")

    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    return path


# ----------------------------- unified CLI ----------------------------- #
def synth(cfg: dict, output_root: str, seed: int = 42) -> Dict:
    """
    Generate S PNGs per class to {output_root}/{class}/{seed}/... and return a manifest.
    """
    H, W, C = tuple(_cfg_get(cfg, "IMG_SHAPE", _cfg_get(cfg, "img.shape", (40, 40, 1))))
    K = int(_cfg_get(cfg, "NUM_CLASSES", _cfg_get(cfg, "num_classes", 9)))
    S = int(_cfg_get(cfg, "SAMPLES_PER_CLASS", _cfg_get(cfg, "samples_per_class", 25)))

    artifacts_root = Path(_cfg_get(cfg, "paths.artifacts", "artifacts"))
    ckpt_dir = Path(_cfg_get(cfg, "ARTIFACTS.autoregressive_checkpoints",
                             artifacts_root / "autoregressive" / "checkpoints"))

    model = load_ar_from_checkpoints(ckpt_dir, img_shape=(H, W, C), num_classes=K)

    np.random.seed(int(seed)); tf.keras.utils.set_random_seed(int(seed))

    out_root = Path(output_root)
    per_class_counts: Dict[str, int] = {str(k): 0 for k in range(K)}
    paths: List[Dict] = []

    for k in range(K):
        class_ids = np.full((S,), k, dtype=np.int32)
        imgs = _sample_autoregressive(model, class_ids=class_ids, img_shape=(H, W, C), num_classes=K, seed=seed)

        cls_dir = out_root / str(k) / str(seed); cls_dir.mkdir(parents=True, exist_ok=True)
        for j in range(S):
            p = cls_dir / f"ar_{j:05d}.png"
            _save_png(imgs[j], p)
            paths.append({"path": str(p), "label": int(k)})
        per_class_counts[str(k)] = int(S)

    return {
        "dataset": _cfg_get(cfg, "data.root", _cfg_get(cfg, "DATA_DIR", "data")),
        "seed": int(seed),
        "per_class_counts": per_class_counts,
        "paths": paths,
    }
