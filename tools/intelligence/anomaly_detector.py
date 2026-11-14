"""Industrial ML Anomaly Detection Engine.

Implements Isolation Forest, Robust Statistical Z-Score / Mahalanobis Anomaly Detection,
and an Ensemble Anomaly Scorer tailored to industrial machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import pickle
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from sklearn.ensemble import IsolationForest

from .feature_extractor import FEATURE_NAMES


@dataclass
class AnomalyResult:
    is_anomaly: bool
    anomaly_score: float             # 0.0 (normal) to 1.0 (critical anomaly)
    model_name: str
    top_contributing_features: list[tuple[str, float]] = field(default_factory=list)
    confidence: float = 0.85
    message: str = ""


class RobustStatisticalDetector:
    """Detects multivariate anomalies using robust median and median absolute deviation (MAD)."""

    def __init__(self, threshold_z: float = 3.5) -> None:
        self.threshold_z = threshold_z
        self.medians: np.ndarray | None = None
        self.mads: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> RobustStatisticalDetector:
        self.medians = np.median(X, axis=0)
        deviations = np.abs(X - self.medians)
        self.mads = np.median(deviations, axis=0)
        # Prevent division by zero for invariant signals
        self.mads = np.where(self.mads < 1e-4, 1.0, self.mads)
        return self

    def score(self, x: np.ndarray) -> tuple[float, list[tuple[str, float]]]:
        if self.medians is None or self.mads is None:
            return 0.0, []
        
        # Modified z-score: 0.6745 * (x - median) / MAD
        z_scores = 0.6745 * np.abs(x - self.medians) / self.mads
        max_z = float(np.max(z_scores))
        
        # Sigmoid scaling to [0, 1]
        score = float(1.0 / (1.0 + np.exp(-0.8 * (max_z - self.threshold_z))))

        # Find top 3 features with highest z-scores
        top_indices = np.argsort(z_scores)[::-1][:3]
        top_features = [(FEATURE_NAMES[i], float(z_scores[i])) for i in top_indices if z_scores[i] > 1.5]

        return score, top_features


class IsolationForestDetector:
    """Unsupervised tree-based anomaly detector using Scikit-Learn IsolationForest."""

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.02,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> IsolationForestDetector:
        self.model.fit(X)
        self.is_fitted = True
        return self

    def score(self, x: np.ndarray) -> float:
        """Returns anomaly score in [0.0, 1.0], where > 0.6 indicates abnormal machine behavior."""
        if not self.is_fitted:
            return 0.0
        # Decision function: lower values mean more abnormal
        raw_score = self.model.decision_function(x.reshape(1, -1))[0]
        # Map raw decision function to [0, 1]
        score = float(1.0 / (1.0 + np.exp(8.0 * raw_score)))
        return score

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str | Path) -> IsolationForestDetector:
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        return self


class HybridIndustrialAnomalyDetector:
    """Ensemble combining Isolation Forest and Statistical MAD detectors with explainability."""

    def __init__(self, anomaly_threshold: float = 0.65) -> None:
        self.threshold = anomaly_threshold
        self.iforest = IsolationForestDetector()
        self.statistical = RobustStatisticalDetector()
        self.is_trained = False

    def train_on_baseline(self, X_normal: np.ndarray) -> None:
        """Trains both detectors on verified healthy operational telemetry."""
        self.iforest.fit(X_normal)
        self.statistical.fit(X_normal)
        self.is_trained = True

    def detect(self, feature_vector: np.ndarray) -> AnomalyResult:
        if not self.is_trained:
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                model_name="Untrained",
                message="Detector not initialized with baseline data.",
            )

        if_score = self.iforest.score(feature_vector)
        stat_score, top_features = self.statistical.score(feature_vector)

        # Weighted blend (60% statistical physical deviation + 40% non-linear tree density)
        combined_score = round(0.40 * if_score + 0.60 * stat_score, 4)
        is_anom = combined_score >= self.threshold

        message = (
            f"Abnormal machine behavior detected (Anomaly Score: {combined_score:.2f})."
            if is_anom
            else "Machine operation within normal baseline envelope."
        )

        return AnomalyResult(
            is_anomaly=is_anom,
            anomaly_score=combined_score,
            model_name="Ensemble(IsolationForest + RobustMAD)",
            top_contributing_features=top_features,
            confidence=round(min(0.99, max(0.50, combined_score)), 3),
            message=message,
        )

