"""
eoflow/models/LinearRegression.py
──────────────────────────────────
Thin wrapper around scikit-learn's ordinary least squares regressor,
conforming to :class:`eoflow.models.base.BaseModel`.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as _SKLinearRegression

from eoflow.log_utils import get_logger
from eoflow.models.base import BaseModel

logger = get_logger(__name__)


class LinearRegression(BaseModel):
    """Ordinary least squares regression over a numeric feature matrix.

    Non-numeric columns are dropped and rows containing NaN in either ``X``
    or ``y`` are excluded before fitting (both with a warning), since the
    underlying estimator cannot handle missing values.

    Args:
        fit_intercept: Whether to fit an intercept term.
        positive: Force all coefficients to be non-negative.
    """

    def __init__(self, *, fit_intercept: bool = True, positive: bool = False) -> None:
        self._estimator = _SKLinearRegression(fit_intercept=fit_intercept, positive=positive)
        self.feature_names_: Optional[list[str]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearRegression":
        X_num = X.select_dtypes(include="number")
        dropped_cols = set(X.columns) - set(X_num.columns)
        if dropped_cols:
            logger.warning("Dropping non-numeric feature columns: %s", sorted(dropped_cols))

        mask = X_num.notna().all(axis=1) & y.notna()
        n_dropped = int((~mask).sum())
        if n_dropped:
            logger.warning(
                "Dropping %d/%d rows containing NaN before fitting.", n_dropped, len(X_num)
            )

        X_clean = X_num.loc[mask]
        y_clean = y.loc[mask]
        if X_clean.empty:
            raise ValueError("No complete rows remain after dropping NaNs; cannot fit.")

        self._estimator.fit(X_clean.to_numpy(dtype=float), y_clean.to_numpy(dtype=float))
        self.feature_names_ = list(X_clean.columns)
        logger.info(
            "Fit LinearRegression on %d rows × %d features.", len(X_clean), len(self.feature_names_)
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_aligned = self._align_columns(X)
        return self._estimator.predict(X_aligned.to_numpy(dtype=float))

    @property
    def coefficients(self) -> pd.Series:
        """Fitted coefficients indexed by feature name."""
        if not self.is_fitted:
            raise RuntimeError("LinearRegression must be fit before accessing coefficients.")
        return pd.Series(self._estimator.coef_, index=self.feature_names_)

    @property
    def intercept(self) -> float:
        """Fitted intercept term."""
        if not self.is_fitted:
            raise RuntimeError("LinearRegression must be fit before accessing intercept.")
        return float(self._estimator.intercept_)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"estimator": self._estimator, "feature_names": self.feature_names_}, f)
        logger.info("Saved LinearRegression model to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> "LinearRegression":
        path = Path(path)
        with path.open("rb") as f:
            state = pickle.load(f)

        model = cls()
        model._estimator = state["estimator"]
        model.feature_names_ = state["feature_names"]
        return model
