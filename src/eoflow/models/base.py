"""
eoflow/models/base.py
──────────────────────
Common interface that every eoflow regression model must implement.

Models consume the feature matrix produced by
:func:`eoflow.features.extract_features_batch` — a DataFrame carrying both
identity columns (``notation``, ``site_name``, ``date``, ``result``, ``unit``)
and numeric predictors. :func:`split_features_target` is the shared helper
for turning that DataFrame into an ``(X, y)`` pair before calling `fit`.

Public API
──────────
  BaseModel
      Abstract base class. Subclasses implement `fit`, `predict`, `save`,
      and `load`; `score` and `is_fitted` are provided.

  split_features_target(df, ...) -> (X, y)
      Split a raw features DataFrame into predictors and target.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from eoflow.log_utils import get_logger

logger = get_logger(__name__)

#: Non-predictor columns emitted by extract_features / extract_features_batch
IDENTITY_COLUMNS: tuple[str, ...] = ("notation", "site_name", "date", "result", "unit")

#: Default regression target column name
TARGET_COLUMN = "result"


def split_features_target(
    df: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    drop_columns: Sequence[str] = IDENTITY_COLUMNS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a features DataFrame into predictors ``X`` and target ``y``.

    Args:
        df: A features DataFrame as returned by
            :func:`eoflow.features.extract_features_batch`.
        target_column: Column to use as the regression target.
        drop_columns: Identity/metadata columns to exclude from ``X``
            (defaults to :data:`IDENTITY_COLUMNS`).

    Returns:
        ``(X, y)`` — ``X`` excludes *drop_columns*; ``y`` is the target series.
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column {target_column!r} not found in DataFrame.")

    y = df[target_column]
    drop = [c for c in drop_columns if c in df.columns]
    X = df.drop(columns=drop)
    return X, y


class BaseModel(ABC):
    """Abstract interface for eoflow regression models.

    A model wraps an underlying estimator and is trained on a numeric
    feature matrix ``X`` (rows = samples, columns = named features) against
    a target ``y``. Implementations are responsible for their own column
    selection, missing-value handling, and persistence format.
    """

    #: Column order the model was trained on; set by `fit`, `None` until then.
    feature_names_: Optional[list[str]] = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Fit the model to *X*, *y* and return self.

        Implementations must set `feature_names_` to the training column
        order before returning.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the target for each row of *X*.

        Args:
            X: Feature matrix containing at least the columns in
                `feature_names_`, in any order.

        Returns:
            1-D array of predictions, one per row of *X*.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Persist the fitted model to *path*."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: Path | str) -> "BaseModel":
        """Load a previously `save`-d model from *path*."""
        raise NotImplementedError

    @property
    def is_fitted(self) -> bool:
        """Whether `fit` has been called successfully."""
        return self.feature_names_ is not None

    def score(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Evaluate predictions for *X* against ground truth *y*.

        Returns:
            Dict with keys ``r2``, ``rmse``, ``mae``.
        """
        if not self.is_fitted:
            raise RuntimeError(f"{type(self).__name__} must be fit before scoring.")

        y_pred = np.asarray(self.predict(X), dtype=float)
        y_true = np.asarray(y, dtype=float)

        resid = y_true - y_pred
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))

        return {
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "rmse": float(np.sqrt(np.mean(resid**2))),
            "mae": float(np.mean(np.abs(resid))),
        }

    def _align_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reindex *X* to the training column order, raising if any are missing."""
        if self.feature_names_ is None:
            raise RuntimeError(f"{type(self).__name__} must be fit before predicting.")

        missing = set(self.feature_names_) - set(X.columns)
        if missing:
            raise KeyError(f"Missing required feature columns: {sorted(missing)}")

        return X[self.feature_names_]
