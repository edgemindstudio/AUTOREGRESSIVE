# autoregressive/models.py

"""
Model builders for the Conditional Autoregressive generator (PixelCNN + Transformer).

This module exposes a single public factory:
    - build_ar_model(...): returns a compiled tf.keras.Model that predicts
      pixels in [0, 1] (sigmoid output) conditioned on a one-hot class label.

Design
------
- Conditioning: the class one-hot vector is projected and reshaped to an
  (H, W, 1) "label map", then concatenated with the input image.
- Causal dependency: we use *masked* convolutions in the PixelCNN style:
  * Mask 'A' in the first layer (blocks the current pixel),
  * Mask 'B' afterwards (allows the current pixel, but not future pixels).
- Long-range structure: a lightweight Transformer block attends over the
  flattened H×W sequence with learnable 2D positional embeddings.
- Output: a 1×1 convolution with `sigmoid` to produce per-pixel Bernoulli
  parameters in [0, 1]. Use Binary Cross Entropy during training.

Notes
-----
- The returned model is **compiled** (Adam + BCE) for convenience.
- Images must be shaped (H, W, C) with values in [0, 1].
- Labels must be one-hot vectors of length `num_classes`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, constraints, initializers, optimizers


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class ARDefaults:
    num_filters: int = 64       # channels in masked conv stack
    num_layers: int = 4         # number of Mask 'B' layers
    num_heads: int = 4          # transformer heads
    ff_multiplier: int = 2      # width of FFN = ff_multiplier * channels
    learning_rate: float = 2e-4
    beta_1: float = 0.5


# ---------------------------------------------------------------------
# Masked convolution (PixelCNN)
# ---------------------------------------------------------------------
class _MaskedKernel(constraints.Constraint):
    """Kernel constraint that multiplies weights by a fixed binary mask."""
    def __init__(self, mask: tf.Tensor):
        self.mask = tf.convert_to_tensor(mask, dtype=tf.float32)

    def __call__(self, w):
        return w * self.mask

    def get_config(self):
        # Storing a huge mask in config isn't ideal, but allows serialization.
        return {"mask": self.mask.numpy().tolist()}


def _make_causal_mask(kh: int, kw: int, in_ch: int, out_ch: int, mask_type: str) -> np.ndarray:
    """
    Build a PixelCNN mask of shape (kh, kw, in_ch, out_ch).
    - 'A': blocks the current pixel (zeroes center and all "future" positions).
    - 'B': allows the current pixel (keeps center, zeroes future positions).
    """
    assert mask_type in ("A", "B")
    mask = np.ones((kh, kw, in_ch, out_ch), dtype=np.float32)
    ch, cw = kh // 2, kw // 2

    # Zero everything to the "right" of the center in the center row
    start = cw + (1 if mask_type == "B" else 0)
    mask[ch, start:, :, :] = 0.0
    # Zero all rows below the center
    mask[ch + 1 :, :, :, :] = 0.0
    return mask


class MaskedConv2D(layers.Layer):
    """
    Masked conv layer that internally uses Conv2D with a kernel constraint.

    Parameters
    ----------
    filters : int
        Number of output channels.
    kernel_size : Tuple[int, int]
        Spatial kernel size (kh, kw).
    mask_type : {'A', 'B'}
        PixelCNN mask variant.
    """
    def __init__(self, filters: int, kernel_size: Tuple[int, int], mask_type: str, **kwargs):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = tuple(kernel_size)
        self.mask_type = mask_type.upper()
        if self.mask_type not in ("A", "B"):
            raise ValueError("mask_type must be 'A' or 'B'")
        self._conv: layers.Conv2D | None = None

    def build(self, input_shape):
        kh, kw = self.kernel_size
        in_ch = int(input_shape[-1])
        mask = _make_causal_mask(kh, kw, in_ch, self.filters, self.mask_type)

        self._conv = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding="same",
            use_bias=True,
            kernel_initializer=initializers.GlorotUniform(),
            kernel_constraint=_MaskedKernel(mask),
            name=f"masked_conv2d_{self.mask_type}",
        )
        # Manually build with the known input shape so weights are created now.
        self._conv.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        # The kernel constraint enforces masking on every update.
        return self._conv(inputs)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "mask_type": self.mask_type,
        })
        return cfg


# ---------------------------------------------------------------------
# Positional embedding & Transformer block
# ---------------------------------------------------------------------
class PositionalEmbedding2D(layers.Layer):
    """Learnable 2D positional embeddings added channel-wise."""
    def __init__(self, height: int, width: int, dim: int):
        super().__init__()
        self.height, self.width, self.dim = int(height), int(width), int(dim)

    def build(self, input_shape):
        self.row_embed = self.add_weight(
            name="row_embed", shape=(self.height, self.dim),
            initializer="random_normal"
        )
        self.col_embed = self.add_weight(
            name="col_embed", shape=(self.width, self.dim),
            initializer="random_normal"
        )
        super().build(input_shape)

    def call(self, x):
        # (H, W, D) via broadcast of (H, D) + (W, D)
        pos = tf.expand_dims(self.row_embed, 1) + tf.expand_dims(self.col_embed, 0)
        pos = tf.expand_dims(pos, 0)  # (1, H, W, D)
        return x + pos

    def get_config(self):
        return {"height": self.height, "width": self.width, "dim": self.dim}


class TransformerBlock(layers.Layer):
    """MHA + FFN with pre-norm residuals."""
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

    def call(self, x):
        attn = self.mha(x, x)
        x = self.norm1(x + attn)
        ffn = self.ffn(x)
        return self.norm2(x + ffn)


# ---------------------------------------------------------------------
# Public factory: build_ar_model
# ---------------------------------------------------------------------
def build_ar_model(
    *,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    num_filters: int = ARDefaults.num_filters,
    num_layers: int = ARDefaults.num_layers,
    num_heads: int = ARDefaults.num_heads,
    ff_multiplier: int = ARDefaults.ff_multiplier,
    learning_rate: float = ARDefaults.learning_rate,
    beta_1: float = ARDefaults.beta_1,
) -> tf.keras.Model:
    """
    Build and compile a conditional autoregressive model (PixelCNN + Transformer).

    Parameters
    ----------
    img_shape : (H, W, C)
        Input image shape in channels-last format. Values expected in [0, 1].
    num_classes : int
        Number of discrete class labels for conditioning.
    num_filters : int
        Channel width of the masked conv stack.
    num_layers : int
        Number of subsequent 'Mask B' layers after the first 'Mask A'.
    num_heads : int
        Number of attention heads in the Transformer block.
    ff_multiplier : int
        Multiplier for FFN width inside the Transformer (FF = multiplier * channels).
    learning_rate : float
        Adam learning rate.
    beta_1 : float
        Adam beta_1 (use 0.5 for GAN-style stability, or 0.9 for standard).

    Returns
    -------
    tf.keras.Model
        A compiled model with inputs [image, one_hot_label] and `sigmoid` output.
    """
    H, W, C = img_shape
    if C <= 0:
        raise ValueError("img_shape must include a positive channel dimension (H, W, C).")

    # Inputs
    x_in = layers.Input(shape=img_shape, name="input_image")
    y_in = layers.Input(shape=(num_classes,), name="label_input")

    # Project one-hot label to an (H, W, 1) map and concatenate with image
    label_map = layers.Dense(H * W, activation="relu", name="label_proj")(y_in)
    label_map = layers.Reshape((H, W, 1), name="label_reshape")(label_map)
    x = layers.Concatenate(axis=-1, name="concat_img_label")([x_in, label_map])

    # Masked conv stack (causal)
    x = MaskedConv2D(filters=num_filters, kernel_size=(7, 7), mask_type="A")(x)
    x = layers.ReLU()(x)

    for i in range(num_layers):
        x = MaskedConv2D(filters=num_filters, kernel_size=(3, 3), mask_type="B", name=f"masked_B_{i}")(x)
        x = layers.ReLU()(x)

    # Add learnable 2D positional embeddings and a lightweight Transformer
    x = PositionalEmbedding2D(height=H, width=W, dim=num_filters)(x)
    x = layers.Reshape((H * W, num_filters))(x)
    x = TransformerBlock(embed_dim=num_filters, num_heads=num_heads, ff_dim=ff_multiplier * num_filters)(x)
    x = layers.Reshape((H, W, num_filters))(x)

    # Final 1×1 conv to per-pixel Bernoulli parameter
    out = layers.Conv2D(1, kernel_size=1, activation="sigmoid", name="pixel_logits")(x)

    model = models.Model(inputs=[x_in, y_in], outputs=out, name="Conditional_PixelCNN_Transformer")

    # Compile (BCE between input pixels and predicted probabilities)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1),
        loss="binary_crossentropy",
    )
    return model


__all__ = ["build_ar_model", "MaskedConv2D", "PositionalEmbedding2D", "TransformerBlock"]
