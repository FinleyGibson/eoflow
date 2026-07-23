"""
eoflow/models/XGBoost.py
─────────────────────────
Thin wrapper around xgboost's XGBRegressor, conforming to
:class:`eoflow.models.base.BaseModel`.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor as _XGBRegressor

from eoflow.log_utils import get_logger
from eoflow.models.base import BaseModel

logger = get_logger(__name__)


class XGBoost(BaseModel):
    """Gradient-boosted tree regression over a numeric feature matrix.

    Boosted trees, fit sequentially to residuals. Like `RandomForest`, no
    linearity or feature-scale assumptions are required.

    Non-numeric columns are dropped and rows containing NaN in either ``X``
    or ``y`` are excluded before fitting (both with a warning) — xgboost can
    natively handle missing values, but rows are dropped anyway here so
    accuracy stays directly comparable against the other `eoflow.models`
    wrappers, which cannot.

    Args:
        n_estimators: Number of boosting rounds (trees).
        max_depth: Maximum tree depth.
        learning_rate: Shrinkage applied to each tree's contribution.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        random_state: int = 0,
    ) -> None:
        self._estimator = _XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            objective="reg:squarederror",
        )
        self.feature_names_: Optional[list[str]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoost":
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
            "Fit XGBoost on %d rows × %d features.", len(X_clean), len(self.feature_names_)
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_aligned = self._align_columns(X)
        return self._estimator.predict(X_aligned.to_numpy(dtype=float))

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"estimator": self._estimator, "feature_names": self.feature_names_}, f)
        logger.info("Saved XGBoost model to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> "XGBoost":
        path = Path(path)
        with path.open("rb") as f:
            state = pickle.load(f)

        model = cls()
        model._estimator = state["estimator"]
        model.feature_names_ = state["feature_names"]
        return model
