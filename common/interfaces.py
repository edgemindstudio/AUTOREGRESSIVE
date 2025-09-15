# common/interfaces.py

"""
Model-agnostic interfaces and evaluation schema.

Use this file in *each* standalone model repo (GAN, VAE, Autoregressive, etc.)
so every project produces the same evaluation JSON. That’s what enables your
cross-repo comparative plots.

Contents
--------
- TypedDict schemas for the evaluation JSON output.
- Lightweight Protocols for train/synthesize/evaluate contracts (optional).
- Simple type aliases and a generic training log-callback type.

This module has no TensorFlow/Keras dependencies.
"""

from __future__ import annotations

from typing import Any, Protocol, TypedDict
import os
from pathlib import Path
import numpy as np

# ----------------------------
# Type aliases
# ----------------------------
PathLike = str | os.PathLike[str] | Path
NDArray = np.ndarray


# ----------------------------
# Config (minimal shape)
# ----------------------------
class ArtifactsConfig(TypedDict, total=False):
    checkpoints: str   # where model weights land
    synthetic: str     # where synthesized npy files land
    summaries: str     # where eval_summary_*.json lands
    tensorboard: str   # optional TB runs


class TrainingConfig(TypedDict, total=False):
    SEED: int
    DATA_DIR: str
    IMG_SHAPE: tuple[int, int, int]
    NUM_CLASSES: int
    LATENT_DIM: int
    EPOCHS: int
    BATCH_SIZE: int
    LR: float
    BETA_1: float
    BETA_KL: float
    VAL_FRACTION: float
    FID_CAP: int
    ARTIFACTS: ArtifactsConfig


# ----------------------------
# Evaluation JSON schema
# (this is what tools/aggregate_and_plot.py expects)
# ----------------------------
class EvalImages(TypedDict):
    train_real: int
    val_real: int
    test_real: int
    synthetic: int


class GenerativeMetrics(TypedDict, total=False):
    fid: float
    cfid_macro: float
    cfid_per_class: list[float]
    js: float
    kl: float
    diversity: float
    fid_domain: float | None


class PerClassMetrics(TypedDict):
    precision: list[float]
    recall: list[float]
    f1: list[float]
    support: list[int]


class UtilityBlock(TypedDict, total=False):
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    macro_auprc: float
    recall_at_1pct_fpr: float
    ece: float
    brier: float
    per_class: PerClassMetrics


class UtilityDeltas(TypedDict, total=False):
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    macro_auprc: float
    recall_at_1pct_fpr: float
    ece: float
    brier: float


class EvalSummary(TypedDict):
    model: str                 # e.g., "ConditionalAR" (or any model name)
    seed: int
    images: EvalImages
    generative: GenerativeMetrics
    utility_real_only: UtilityBlock
    utility_real_plus_synth: UtilityBlock
    deltas_RS_minus_R: UtilityDeltas


# ----------------------------
# Generic training callback
# ----------------------------
# Use this for simple “train/val loss” logging; adapt if you need more.
TrainLogCallback = callable[[int, float, float], None]  # epoch, train_loss, val_loss


# ----------------------------
# Optional Protocols
# ----------------------------
class TrainPipeline(Protocol):
    """Minimal interface for a training pipeline."""
    def train(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """
        Start training and return one or more trained components.
        Example returns:
          - (model,)
          - (encoder, decoder)
          - (generator, discriminator)
        """
        ...


class Synthesizer(Protocol):
    """Return (x_synth in [0,1], y_synth one-hot)."""
    def synthesize(self, *args: Any, **kwargs: Any) -> tuple[NDArray, NDArray]:
        ...


class EvaluateModelSuite(Protocol):
    """
    Contract for an evaluation function used by your `app/main.py`.
    Keyword-only args (names matter for compatibility with your evaluator):

      model_name: str
      img_shape: tuple[int, int, int]
      x_train_real, y_train_real,
      x_val_real,   y_val_real,
      x_test_real,  y_test_real: np.ndarray
      x_synth, y_synth: np.ndarray | None
      per_class_cap_for_fid: int
      seed: int
    """
    def __call__(
        self,
        *,
        model_name: str,
        img_shape: tuple[int, int, int],
        x_train_real: NDArray, y_train_real: NDArray,
        x_val_real: NDArray,   y_val_real: NDArray,
        x_test_real: NDArray,  y_test_real: NDArray,
        x_synth: NDArray | None,
        y_synth: NDArray | None,
        per_class_cap_for_fid: int,
        seed: int
    ) -> EvalSummary: ...
