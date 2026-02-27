from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import (
    DEFAULT_MODEL_DIR,
    DYNAMIC_MODEL_ORDER,
    ROOT_CAUSE_HINTS,
)
from .preprocess import clean_numeric_frame, prepare_dynamic_features


def _load_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    return joblib.load(path)


def load_models(model_dir: Path = DEFAULT_MODEL_DIR) -> dict[str, dict[str, Any]]:
    model_dir = Path(model_dir)
    models: dict[str, dict[str, Any]] = {}
    for name in DYNAMIC_MODEL_ORDER:
        models[name] = _load_artifact(model_dir / f"{name}.joblib")
    models["static_layer"] = _load_artifact(model_dir / "static_layer.joblib")
    return models


def _transform_dynamic(
    features: pd.DataFrame, artifact: dict[str, Any]
) -> np.ndarray:
    aligned = features[artifact["feature_columns"]]
    scaled = artifact["scaler"].transform(aligned)
    return scaled[:, artifact["selected_indices"]]


def _predict_binary_probability(
    transformed: np.ndarray, artifact: dict[str, Any]
) -> float:
    probs = artifact["model"].predict_proba(transformed)[:, 1]
    return float(np.mean(probs))


def predict_csv(csv_path: Path, models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    frame = pd.read_csv(csv_path)
    return predict_frame(frame, models, source_name=str(csv_path))


def predict_frame(
    frame: pd.DataFrame,
    models: dict[str, dict[str, Any]],
    *,
    source_name: str = "uploaded.csv",
) -> dict[str, Any]:
    features = prepare_dynamic_features(frame)

    result: dict[str, Any] = {
        "file": source_name,
        "classifiers": {},
    }

    for name in DYNAMIC_MODEL_ORDER:
        artifact = models[name]
        transformed = _transform_dynamic(features, artifact)
        probability = _predict_binary_probability(transformed, artifact)
        threshold = float(artifact["threshold"])
        is_positive = probability >= threshold
        result["classifiers"][name] = {
            "probability": probability,
            "threshold": threshold,
            "predicted_positive": bool(is_positive),
        }

    detection_positive = result["classifiers"]["detection"]["predicted_positive"]
    categories = []
    if detection_positive:
        for name in DYNAMIC_MODEL_ORDER[1:]:
            if result["classifiers"][name]["predicted_positive"]:
                categories.append(name)

    result["detected_fault"] = bool(detection_positive)
    result["predicted_categories"] = categories
    return result


def predict_path(path: Path, models: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    path = Path(path)
    if path.is_file():
        return [predict_csv(path, models)]
    if not path.is_dir():
        raise FileNotFoundError(f"Path does not exist: {path}")
    results = []
    for csv_file in sorted(path.glob("*.csv")):
        results.append(predict_csv(csv_file, models))
    return results


def explain_static_features(
    features: pd.DataFrame,
    models: dict[str, dict[str, Any]],
    *,
    top_n: int = 5,
) -> dict[str, Any]:
    artifact = models["static_layer"]
    expected_columns = artifact["feature_columns"]
    aligned = features.reindex(columns=expected_columns, fill_value=0.0)
    aligned = clean_numeric_frame(aligned)
    scaled = artifact["scaler"].transform(aligned)
    transformed = scaled[:, artifact["selected_indices"]]

    prob = float(artifact["model"].predict_proba(transformed)[:, 1][0])
    threshold = float(artifact["threshold"])
    predicted_buggy = prob >= threshold

    top_features: list[dict[str, Any]] = []
    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(artifact["model"])
        shap_values = explainer.shap_values(transformed)
        if isinstance(shap_values, list):
            sample_values = np.abs(np.array(shap_values[1][0]))
        else:
            sample_values = np.abs(np.array(shap_values[0]))
        ranked = np.argsort(sample_values)[::-1][:top_n]
        selected_names = artifact["selected_features"]
        for idx in ranked:
            name = selected_names[idx]
            top_features.append(
                {
                    "feature": name,
                    "score": float(sample_values[idx]),
                    "hint": ROOT_CAUSE_HINTS.get(name, "Inspect this feature."),
                }
            )
    except Exception:
        importances = artifact["model"].feature_importances_
        ranked = np.argsort(importances)[::-1][:top_n]
        selected_names = artifact["selected_features"]
        for idx in ranked:
            name = selected_names[idx]
            top_features.append(
                {
                    "feature": name,
                    "score": float(importances[idx]),
                    "hint": ROOT_CAUSE_HINTS.get(name, "Inspect this feature."),
                }
            )

    return {
        "probability_buggy": prob,
        "threshold": threshold,
        "predicted_buggy": bool(predicted_buggy),
        "top_features": top_features,
    }
