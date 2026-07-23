"""
eoflow/models/RandomForest.py
──────────────────────────────
Thin wrapper around scikit-learn's RandomForestRegressor, conforming to
:class:`eoflow.models.base.BaseModel`.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as _SKRandomForestRegressor

from eoflow.log_utils import get_logger
from eoflow.models.base import BaseModel

logger = get_logger(__name__)


class RandomForest(BaseModel):
    """Random forest regression over a numeric feature matrix.

    An ensemble of bagged decision trees. Unlike the linear models, it makes
    no linearity or feature-scale assumptions and captures non-linear
    interactions natively.

    Non-numeric columns are dropped and rows containing NaN in either ``X``
    or ``y`` are excluded before fitting (both with a warning), since
    scikit-learn's tree ensembles cannot handle missing values.

    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum tree depth (``None`` = expand until leaves are pure).
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        random_state: int = 0,
    ) -> None:
        self._estimator = _SKRandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        self.feature_names_: Optional[list[str]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForest":
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
            "Fit RandomForest on %d rows × %d features.", len(X_clean), len(self.feature_names_)
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_aligned = self._align_columns(X)
        return self._estimator.predict(X_aligned.to_numpy(dtype=float))

    @property
    def feature_importances(self) -> pd.Series:
        """Impurity-based feature importances indexed by feature name.

        Unlike linear-model coefficients, these are unsigned (>= 0) and sum
        to 1 — magnitude alone indicates importance, there's no direction to
        interpret.
        """
        if not self.is_fitted:
            raise RuntimeError("RandomForest must be fit before accessing feature_importances.")
        return pd.Series(self._estimator.feature_importances_, index=self.feature_names_)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"estimator": self._estimator, "feature_names": self.feature_names_}, f)
        logger.info("Saved RandomForest model to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> "RandomForest":
        path = Path(path)
        with path.open("rb") as f:
            state = pickle.load(f)

        model = cls()
        model._estimator = state["estimator"]
        model.feature_names_ = state["feature_names"]
        return model
