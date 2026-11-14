"""Supervised Industrial Fault Classification using XGBoost and Random Forest.

Classifies detected anomalies into specific failure categories, distinguishing real-world
leak signatures, synthetic failure injections, and benign high-demand look-alikes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
from typing import Any, Sequence
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import xgboost as xgb

from .feature_extractor import FEATURE_NAMES


FAULT_CLASSES = [
    "normal_operation",
    "dryer_purge_leak",
    "downstream_air_leak",
    "oil_cooler_fouled",
    "oil_filter_blocked",
    "high_air_demand",
]

FAULT_DISPLAY_NAMES = {
    "normal_operation": "Normal Operation (No Fault)",
    "dryer_purge_leak": "Dryer Purge Valve Leak (Signature A)",
    "downstream_air_leak": "Downstream Air Leak (Signature B)",
    "oil_cooler_fouled": "Oil Cooler Thermal Fouling",
    "oil_filter_blocked": "Oil Filter Blockage",
    "high_air_demand": "High Air Demand (Benign Load)",
}


@dataclass
class ClassificationResult:
    predicted_fault: str
    display_name: str
    confidence: float
    probabilities: dict[str, float]
    feature_importances: list[tuple[str, float]]
    model_name: str
    is_benign: bool = False


class IndustrialFaultClassifier:
    """XGBoost and Random Forest classifier for industrial failure identification."""

    def __init__(self, model_type: str = "xgboost") -> None:
        self.model_type = model_type.lower()
        if self.model_type == "xgboost":
            self.model: Any = xgb.XGBClassifier(
                n_estimators=120,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="multi:softprob",
                num_class=len(FAULT_CLASSES),
                random_state=42,
                eval_metric="mlogloss",
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=120,
                max_depth=8,
                random_state=42,
                n_jobs=-1,
            )
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> IndustrialFaultClassifier:
        self.model.fit(X, y)
        self.is_trained = True
        return self

    def predict(self, feature_vector: np.ndarray) -> ClassificationResult:
        if not self.is_trained:
            return ClassificationResult(
                predicted_fault="normal_operation",
                display_name="Normal Operation",
                confidence=0.5,
                probabilities={c: (1.0 / len(FAULT_CLASSES)) for c in FAULT_CLASSES},
                feature_importances=[],
                model_name=self.model_type,
            )

        X = feature_vector.reshape(1, -1)
        probs = self.model.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        predicted_fault = FAULT_CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

        prob_dict = {
            fault: round(float(p), 4)
            for fault, p in zip(FAULT_CLASSES, probs, strict=False)
        }

        # Extract top feature importances
        importances = self.model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1][:4]
        top_features = [
            (FEATURE_NAMES[i], round(float(importances[i]), 4))
            for i in sorted_indices
        ]

        is_benign = predicted_fault in ("normal_operation", "high_air_demand")

        return ClassificationResult(
            predicted_fault=predicted_fault,
            display_name=FAULT_DISPLAY_NAMES.get(predicted_fault, predicted_fault),
            confidence=round(confidence, 4),
            probabilities=prob_dict,
            feature_importances=top_features,
            model_name=f"XGBoost-3.4 ({self.model_type.upper()})",
            is_benign=is_benign,
        )

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
        """Evaluates model performance and returns metrics dict."""
        preds = self.model.predict(X_test)
        report = classification_report(
            y_test,
            preds,
            target_names=[FAULT_CLASSES[i] for i in sorted(list(set(y_test) | set(preds)))],
            output_dict=True,
            zero_division=0,
        )
        conf_mat = confusion_matrix(y_test, preds).tolist()
        macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
        precision = float(precision_score(y_test, preds, average="macro", zero_division=0))
        recall = float(recall_score(y_test, preds, average="macro", zero_division=0))

        return {
            "model": self.model_type,
            "macro_precision": round(precision, 4),
            "macro_recall": round(recall, 4),
            "macro_f1": round(macro_f1, 4),
            "confusion_matrix": conf_mat,
            "detailed_report": report,
        }

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "type": self.model_type}, f)

    def load(self, path: str | Path) -> IndustrialFaultClassifier:
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.model_type = data["type"]
            self.is_trained = True
        return self

