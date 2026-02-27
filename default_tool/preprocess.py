from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DYNAMIC_FEATURE_COLUMNS

_ALIASES = {
    "mean_grad": ("mean_grad", "mean_gradient"),
    "std_grad": ("std_grad", "gradient_std"),
    "gradient_std": ("gradient_std", "std_grad"),
}


def _to_bool(value: object) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value != 0)
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return 1
    return int("true" in lowered)


def clean_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned = cleaned.apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    for col in cleaned.columns:
        mean_value = cleaned[col].mean()
        fill_value = 0.0 if pd.isna(mean_value) else float(mean_value)
        cleaned[col] = cleaned[col].fillna(fill_value)
    for col in cleaned.columns:
        lower_q = cleaned[col].quantile(0.01)
        upper_q = cleaned[col].quantile(0.99)
        cleaned[col] = cleaned[col].clip(lower=lower_q, upper=upper_q)
    for col in cleaned.columns:
        mean_value = cleaned[col].mean()
        fill_value = 0.0 if pd.isna(mean_value) else float(mean_value)
        cleaned[col] = cleaned[col].fillna(fill_value)
    return cleaned


def prepare_dynamic_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    for col in ("dying_relu", "saturated_activation"):
        if col in df.columns:
            df[col] = df[col].apply(_to_bool)

    columns = {}
    for target in DYNAMIC_FEATURE_COLUMNS:
        if target in df.columns:
            columns[target] = df[target]
            continue
        found = False
        for alias in _ALIASES.get(target, ()):
            if alias in df.columns:
                columns[target] = df[alias]
                found = True
                break
        if not found:
            raise ValueError(f"Missing required feature column: {target}")

    selected = pd.DataFrame(columns)
    return clean_numeric_frame(selected)

