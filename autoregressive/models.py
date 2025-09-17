# autoregressive/models.py
"""
Conditional PixelCNN (masked conv) builder used by synth/train.

Exposes:
- build_conditional_pixelcnn(img_shape, num_classes, ...): tf.keras.Model
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, initializers


# ----------------------------- Masked Conv2D ----------------------------- #
def _make_causal_mask(kh: int, kw: int, in_ch: int, out_ch: int, mask_type: str) -> np.ndarray:
    """PixelCNN mask for kernel (kh, kw, in_ch, out_ch)."""
    assert mask_type in ("A", "B")
    m = np.ones((kh, kw, in_ch, out_ch), dtype=np.float32)
    ch, cw = kh // 2, kw // 2
    # zero everything strictly to the right of center in the center row
    m[ch, cw + (1 if mask_type == "B" else 0) :, :, :] = 0.0
    # zero all rows below the center
    m[ch + 1 :, :, :, :] = 0.0
    return m


class MaskedConv2D(layers.Layer):
    """
    PixelCNN-style masked conv implemented via explicit weights so that
    variable names are simply 'kernel' and 'bias' (robust for load/save).
    """
    def __init__(self, filters: int, kernel_size: int | Tuple[int, int], mask_type: str, **kwargs):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.mask_type = mask_type.upper()
        if self.mask_type not in ("A", "B"):
            raise ValueError("mask_type must be 'A' or 'B'")

    def build(self, input_shape):
        kh, kw = self.kernel_size
        in_ch = int(input_shape[-1])
        mask = _make_causal_mask(kh, kw, in_ch, self.filters, self.mask_type)

        self.kernel = self.add_weight(
            name="kernel",
            shape=(kh, kw, in_ch, self.filters),
            initializer=initializers.GlorotUniform(),
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.filters,),
            initializer="zeros",
            trainable=True,
        )
        # store mask as constant
        self._mask = tf.constant(mask, dtype=self.dtype if self.dtype else tf.float32)
        super().build(input_shape)

    def call(self, inputs):
        k = self.kernel * self._mask
        y = tf.nn.conv2d(inputs, k, strides=1, padding="SAME")
        y = tf.nn.bias_add(y, self.bias)
        return y


# ----------------------------- Model builder ----------------------------- #
def build_conditional_pixelcnn(
    img_shape: Tuple[int, int, int],
    num_classes: int,
    *,
    filters: int = 64,
    masked_layers: int = 6,
) -> tf.keras.Model:
    """
    Build a small conditional PixelCNN:
      - First masked 'A' conv (7x7)
      - Several masked 'B' convs (3x3)
      - 1x1 projection + sigmoid head
      - Conditioning via label map concatenation (Dense -> (H,W,1))
    """
    H, W, C = img_shape
    x_in = layers.Input(shape=(H, W, C), name="image")
    y_in = layers.Input(shape=(num_classes,), name="onehot")

    # Label -> (H,W,1) map
    lab = layers.Dense(H * W, activation="relu", name="label_proj")(y_in)
    lab = layers.Reshape((H, W, 1), name="label_map")(lab)

    x = layers.Concatenate(axis=-1, name="concat_img_label")([x_in, lab])

    # Masked conv stack
    h = MaskedConv2D(filters, kernel_size=7, mask_type="A", name="maskedA_7x7")(x)
    h = layers.ReLU()(h)
    for i in range(1, masked_layers + 1):
        h = MaskedConv2D(filters, kernel_size=3, mask_type="B", name=f"maskedB_{i}")(h)
        h = layers.ReLU()(h)

    h = layers.Conv2D(filters, kernel_size=1, padding="same", activation="relu", name="proj_1x1")(h)
    out = layers.Conv2D(1, kernel_size=1, padding="same", activation="sigmoid", name="probs")(h)

    return models.Model([x_in, y_in], out, name="ConditionalPixelCNN")
