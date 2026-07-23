"""
eoflow/models/Lasso.py
───────────────────────
Thin wrapper around scikit-learn's Lasso (L1-regularized) regressor,
conforming to :class:`eoflow.models.base.BaseModel`.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso as _SKLasso
from sklearn.preprocessing import StandardScaler

from eoflow.log_utils import get_logger
from eoflow.models.base import BaseModel

logger = get_logger(__name__)


class Lasso(BaseModel):
    """L1-regularized (Lasso) regression over a numeric feature matrix.

    Features are standardized (zero mean, unit variance) internally before
    fitting. The L1 penalty is scale-dependent, so an unstandardized feature
    with a large numeric range (e.g. ``catchment_area_km2``, in the hundreds)
    would otherwise be penalized far less than one on a ``[0, 1]`` scale
    (e.g. a soil fraction), regardless of actual predictive value.
    Standardizing also makes the fitted `coefficients` directly comparable
    across features, unlike raw-scale OLS coefficients.

    Non-numeric columns are dropped and rows containing NaN in either ``X``
    or ``y`` are excluded before fitting (both with a warning), since the
    underlying estimator cannot handle missing values. The L1 penalty drives
    weak-feature coefficients to exactly zero, giving built-in feature
    selection — useful when the feature count rivals or exceeds the row
    count (see ``docs/PIPELINE.md``).

    Args:
        alpha: L1 regularization strength (higher = more coefficients pushed to zero).
        fit_intercept: Whether to fit an intercept term.
        positive: Force all coefficients to be non-negative.
        max_iter: Maximum number of coordinate-descent iterations.
    """

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        positive: bool = False,
        max_iter: int = 10_000,
    ) -> None:
        self._scaler = StandardScaler()
        self._estimator = _SKLasso(
            alpha=alpha, fit_intercept=fit_intercept, positive=positive, max_iter=max_iter
        )
        self.feature_names_: Optional[list[str]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Lasso":
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

        X_scaled = self._scaler.fit_transform(X_clean.to_numpy(dtype=float))
        self._estimator.fit(X_scaled, y_clean.to_numpy(dtype=float))
        self.feature_names_ = list(X_clean.columns)

        n_selected = int(np.count_nonzero(self._estimator.coef_))
        logger.info(
            "Fit Lasso on %d rows × %d features (%d/%d coefficients non-zero, alpha=%g).",
            len(X_clean), len(self.feature_names_), n_selected, len(self.feature_names_),
            self._estimator.alpha,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_aligned = self._align_columns(X)
        X_scaled = self._scaler.transform(X_aligned.to_numpy(dtype=float))
        return self._estimator.predict(X_scaled)

    @property
    def coefficients(self) -> pd.Series:
        """Fitted coefficients (standardized-feature scale) indexed by feature name."""
        if not self.is_fitted:
            raise RuntimeError("Lasso must be fit before accessing coefficients.")
        return pd.Series(self._estimator.coef_, index=self.feature_names_)

    @property
    def n_selected(self) -> int:
        """Number of features with a non-zero coefficient."""
        return int(np.count_nonzero(self.coefficients.to_numpy()))

    @property
    def intercept(self) -> float:
        """Fitted intercept term."""
        if not self.is_fitted:
            raise RuntimeError("Lasso must be fit before accessing intercept.")
        return float(self._estimator.intercept_)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "estimator": self._estimator,
                    "scaler": self._scaler,
                    "feature_names": self.feature_names_,
                },
                f,
            )
        logger.info("Saved Lasso model to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> "Lasso":
        path = Path(path)
        with path.open("rb") as f:
            state = pickle.load(f)

        model = cls()
        model._estimator = state["estimator"]
        model._scaler = state["scaler"]
        model.feature_names_ = state["feature_names"]
        return model
