from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL_DIR,
    DYNAMIC_MODEL_ORDER,
    DYNAMIC_TRAINING_SPECS,
    STATIC_DATASET,
    STATIC_ID_COLUMN,
    STATIC_TARGET_COLUMN,
)
from .preprocess import clean_numeric_frame, prepare_dynamic_features


def _to_binary_labels(series: pd.Series, positive_label: Any) -> pd.Series:
    if is_numeric_dtype(series):
        return (series.astype(float) == float(positive_label)).astype(int)
    target = str(positive_label).strip().lower()
    return series.astype(str).str.strip().str.lower().eq(target).astype(int)


def _find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def _train_binary_rf(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 42,
    top_k: int = 20,
) -> dict[str, Any]:
    if y.nunique() < 2:
        raise ValueError("Binary training requires both classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    selector = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    selector.fit(X_train_scaled, y_train)
    importances = selector.feature_importances_
    selected_indices = np.argsort(importances)[::-1][: min(top_k, X.shape[1])]

    model = RandomForestClassifier(
        n_estimators=400,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train_scaled[:, selected_indices], y_train)

    y_prob = model.predict_proba(X_test_scaled[:, selected_indices])[:, 1]
    threshold = _find_best_threshold(y_test.to_numpy(), y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
    metrics["threshold"] = float(threshold)

    return {
        "model": model,
        "scaler": scaler,
        "feature_columns": X.columns.tolist(),
        "selected_indices": selected_indices.tolist(),
        "selected_features": X.columns[selected_indices].tolist(),
        "threshold": float(threshold),
        "metrics": metrics,
    }


def train_dynamic_model(
    dataset_path: Path,
    *,
    positive_label: Any,
    random_state: int = 42,
) -> dict[str, Any]:
    frame = pd.read_csv(dataset_path)
    if "label" not in frame.columns:
        raise ValueError(f"Expected 'label' column in {dataset_path}")
    X = prepare_dynamic_features(frame)
    y = _to_binary_labels(frame["label"], positive_label)
    artifact = _train_binary_rf(X, y, random_state=random_state)
    artifact["positive_label"] = positive_label
    artifact["kind"] = "dynamic"
    return artifact


def train_static_explainer(
    dataset_path: Path,
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    frame = pd.read_csv(dataset_path)
    if STATIC_TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Expected '{STATIC_TARGET_COLUMN}' in {dataset_path}")

    X = frame.drop(columns=[STATIC_TARGET_COLUMN, STATIC_ID_COLUMN], errors="ignore")
    X = clean_numeric_frame(X)
    y = _to_binary_labels(frame[STATIC_TARGET_COLUMN], 1)

    artifact = _train_binary_rf(X, y, random_state=random_state)
    artifact["positive_label"] = 1
    artifact["kind"] = "static"
    return artifact


def _save_artifact(artifact: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def train_all_models(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    model_dir: Path = DEFAULT_MODEL_DIR,
    random_state: int = 42,
) -> dict[str, Any]:
    data_root = Path(data_root)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"models": {}}

    for model_name in DYNAMIC_MODEL_ORDER:
        spec = DYNAMIC_TRAINING_SPECS[model_name]
        dataset_path = data_root / spec["relative_path"]
        artifact = train_dynamic_model(
            dataset_path,
            positive_label=spec["positive_label"],
            random_state=random_state,
        )
        output_path = model_dir / f"{model_name}.joblib"
        _save_artifact(artifact, output_path)
        manifest["models"][model_name] = {
            "dataset": str(dataset_path),
            "positive_label": spec["positive_label"],
            "threshold": artifact["threshold"],
            "metrics": artifact["metrics"],
        }

    static_dataset = data_root / STATIC_DATASET
    static_artifact = train_static_explainer(
        static_dataset, random_state=random_state
    )
    static_path = model_dir / "static_layer.joblib"
    _save_artifact(static_artifact, static_path)
    manifest["models"]["static_layer"] = {
        "dataset": str(static_dataset),
        "positive_label": 1,
        "threshold": static_artifact["threshold"],
        "metrics": static_artifact["metrics"],
    }

    manifest_path = model_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return manifest
