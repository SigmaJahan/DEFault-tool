"""
Inference utilities for the DEFault professional tool.

Provides three capabilities not in the original inference.py:
1. explain_static_signed() — SHAP with signed values for the waterfall chart
2. derive_dynamic_features() — derive dynamic features from Keras training history
3. get_fault_taxonomy() — the full DNN fault hierarchy from the paper (Fig. 3)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from default_tool.config import DYNAMIC_FEATURE_COLUMNS, ROOT_CAUSE_HINTS
from default_tool.preprocess import clean_numeric_frame
from webapp.schemas import (
    CategoryResult,
    FaultTaxonomyNode,
    FaultTaxonomyResponse,
    ShapFeature,
    StageOneResult,
    StaticAnalysisResult,
    StageTwoResult,
    TrainingSummary,
)

# Categories in the same order as DYNAMIC_MODEL_ORDER (excluding "detection")
_CATEGORY_NAMES = [
    "activation",
    "layer",
    "hyperparameter",
    "loss",
    "optimization",
    "regularizer",
    "weights",
]

# Dynamic features that CAN be derived from standard Keras history.history
_DERIVABLE_FEATURES = {
    "train_acc",
    "val_acc",
    "loss_oscillation",
    "acc_gap_too_big",
    "adjusted_lr",
    "gradient_vanish",    # zero-filled with note
    "gradient_explode",   # zero-filled with note
    "gradient_median",    # zero-filled
    "std_grad",           # zero-filled
    "gradient_min",       # zero-filled
    "mean_grad",          # zero-filled
    "large_weight_count", # zero-filled
    "mean_activation",    # zero-filled
    "std_activation",     # zero-filled
    "gradient_std",       # zero-filled
    "gradient_max",       # zero-filled
    "gpu_memory_utilization",  # zero-filled
    "cpu_utilization",         # zero-filled
    "memory_usage",            # zero-filled
    "dying_relu",              # zero-filled
}

_TRULY_DERIVABLE = {
    "train_acc",
    "val_acc",
    "loss_oscillation",
    "acc_gap_too_big",
}


# ── 1. Signed SHAP explanation ────────────────────────────────────────────────

def explain_static_signed(
    features_df: pd.DataFrame,
    models: dict[str, dict[str, Any]],
    top_n: int = 7,
) -> StaticAnalysisResult:
    """
    Run Stage 3 static analysis with signed SHAP values for the waterfall chart.

    The existing explain_static_features() discards sign with np.abs().
    This version preserves sign so the frontend can render:
      - Red bars (positive SHAP) = pushes toward 'buggy'
      - Green bars (negative SHAP) = pushes toward 'correct'
    """
    artifact = models["static_layer"]
    expected_columns = artifact["feature_columns"]
    aligned = features_df.reindex(columns=expected_columns, fill_value=0.0)
    aligned = clean_numeric_frame(aligned)
    scaled = artifact["scaler"].transform(aligned)
    transformed = scaled[:, artifact["selected_indices"]]

    prob = float(artifact["model"].predict_proba(transformed)[:, 1][0])
    threshold = float(artifact["threshold"])
    predicted_buggy = prob >= threshold
    selected_names: list[str] = artifact["selected_features"]

    # Build all_features dict (original unscaled values, aligned to expected cols)
    original_aligned = features_df.reindex(columns=expected_columns, fill_value=0.0)
    all_features: dict[str, float] = {
        col: float(original_aligned.iloc[0].get(col, 0.0))
        for col in expected_columns
    }

    shap_features: list[ShapFeature] = []
    base_value = 0.5  # fallback

    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(artifact["model"])
        shap_values = explainer.shap_values(transformed)

        # shap_values is a list [class_0_values, class_1_values] for binary RF
        # We want class 1 (buggy) with SIGN preserved
        if isinstance(shap_values, list) and len(shap_values) >= 2:
            signed_values = np.array(shap_values[1][0])
            # expected_value is also per-class
            ev = explainer.expected_value
            base_value = float(ev[1] if hasattr(ev, "__len__") else ev)
        else:
            signed_values = np.array(shap_values[0])
            ev = explainer.expected_value
            base_value = float(ev[0] if hasattr(ev, "__len__") else ev)

        # Sort by absolute value descending, take top_n
        abs_values = np.abs(signed_values)
        ranked_indices = np.argsort(abs_values)[::-1][:top_n]

        for idx in ranked_indices:
            name = selected_names[idx]
            sv = float(signed_values[idx])
            feature_val = float(transformed[0][idx])  # scaled value used by model
            shap_features.append(ShapFeature(
                feature=name,
                shap_value=sv,
                abs_shap=abs(sv),
                feature_value=feature_val,
                hint=ROOT_CAUSE_HINTS.get(name, "Inspect this architectural feature."),
            ))

    except Exception:
        # Fallback: use feature importances (unsigned) — still useful
        importances = artifact["model"].feature_importances_
        ranked_indices = np.argsort(importances)[::-1][:top_n]
        for idx in ranked_indices:
            name = selected_names[idx]
            imp = float(importances[idx])
            shap_features.append(ShapFeature(
                feature=name,
                shap_value=imp,    # positive only in fallback
                abs_shap=imp,
                feature_value=float(transformed[0][idx]),
                hint=ROOT_CAUSE_HINTS.get(name, "Inspect this architectural feature."),
            ))

    return StaticAnalysisResult(
        probability_buggy=prob,
        threshold=threshold,
        predicted_buggy=predicted_buggy,
        top_features=shap_features,
        all_features=all_features,
        base_value=base_value,
    )


# ── 2. Dynamic feature derivation from training history ───────────────────────

def derive_dynamic_features(
    loss: list[float],
    val_loss: list[float],
    train_acc: list[float],
    val_acc: list[float],
) -> tuple[pd.DataFrame, TrainingSummary]:
    """
    Derive dynamic features from standard Keras training history.
    Many features (gradients, GPU stats, dying_relu) cannot be derived
    and are zero-filled; the TrainingSummary reports what was available.

    Returns:
        (feature_dataframe, training_summary)
    """
    n = len(loss)
    if n == 0:
        raise ValueError("Training history must have at least 1 epoch.")

    # Compute derivable features
    # loss_oscillation: std of epoch-to-epoch changes
    loss_diffs = [loss[i] - loss[i - 1] for i in range(1, n)]
    loss_oscillation = float(np.std(loss_diffs)) if loss_diffs else 0.0

    final_train_acc = float(train_acc[-1])
    final_val_acc = float(val_acc[-1])
    acc_gap = final_val_acc - final_train_acc   # negative = overfitting
    acc_gap_too_big = 1.0 if acc_gap < -0.05 else 0.0

    decrease_acc_count = sum(
        1 for i in range(1, len(train_acc)) if train_acc[i] < train_acc[i - 1]
    )
    increase_loss_count = sum(
        1 for i in range(1, n) if loss[i] > loss[i - 1]
    )

    # Adjusted LR: if acc stagnates for > 3 epochs, signal potential LR issue
    stagnant_epochs = sum(
        1 for i in range(1, len(train_acc))
        if abs(train_acc[i] - train_acc[i - 1]) < 0.001
    )
    adjusted_lr_signal = float(stagnant_epochs) / max(n, 1)

    # Build the full 20-feature dict, zero-filling unavailable ones
    feature_dict: dict[str, float] = {col: 0.0 for col in DYNAMIC_FEATURE_COLUMNS}
    feature_dict.update({
        "train_acc": final_train_acc,
        "val_acc": final_val_acc,
        "loss_oscillation": loss_oscillation,
        "acc_gap_too_big": acc_gap_too_big,
        "adjusted_lr": adjusted_lr_signal,
    })

    # Average loss and val_loss as context
    feature_dict["memory_usage"] = 0.0  # unknown without runtime

    # Convert to single-row DataFrame that prepare_dynamic_features can consume
    df = pd.DataFrame([feature_dict])

    available = list(_TRULY_DERIVABLE)
    missing = [c for c in DYNAMIC_FEATURE_COLUMNS if c not in _TRULY_DERIVABLE]

    summary = TrainingSummary(
        epochs=n,
        final_train_acc=final_train_acc,
        final_val_acc=final_val_acc,
        final_loss=float(loss[-1]),
        final_val_loss=float(val_loss[-1]) if val_loss else None,
        loss_oscillation=loss_oscillation,
        acc_gap=acc_gap,
        decrease_acc_count=decrease_acc_count,
        increase_loss_count=increase_loss_count,
        available_dynamic_features=available,
        missing_dynamic_features=missing,
    )

    return df, summary


def build_stage1_result(classifiers: dict[str, Any]) -> StageOneResult:
    det = classifiers["detection"]
    return StageOneResult(
        probability=det["probability"],
        threshold=det["threshold"],
        predicted_positive=det["predicted_positive"],
    )


def build_stage2_result(classifiers: dict[str, Any], detected: bool) -> StageTwoResult:
    categories: list[CategoryResult] = []
    flagged: list[str] = []
    for name in _CATEGORY_NAMES:
        clf = classifiers[name]
        is_positive = bool(clf["predicted_positive"]) and detected
        if is_positive:
            flagged.append(name)
        categories.append(CategoryResult(
            name=name,
            probability=clf["probability"],
            threshold=clf["threshold"],
            predicted_positive=is_positive,
        ))
    # Sort by probability descending for UI
    categories.sort(key=lambda c: c.probability, reverse=True)
    return StageTwoResult(categories=categories, flagged=flagged)


# ── 3. Fault taxonomy tree (paper Fig. 3) ─────────────────────────────────────

def get_fault_taxonomy() -> FaultTaxonomyResponse:
    """
    Return the full DNN fault hierarchy from the ICSE 2025 paper (Fig. 3).
    fault_category fields map to Stage 2 classifier names for frontend highlighting.
    """
    root = FaultTaxonomyNode(
        id="dnn_faults",
        label="DNN Program Faults",
        description="All fault types in deep neural network programs",
        children=[
            FaultTaxonomyNode(
                id="training_faults",
                label="Training Faults",
                description="Faults that affect the training process",
                children=[
                    FaultTaxonomyNode(
                        id="hyperparameter",
                        label="Hyperparameter",
                        description="Incorrect hyperparameter configuration",
                        fault_category="hyperparameter",
                        children=[
                            FaultTaxonomyNode(id="learning_rate", label="Learning Rate",
                                description="Learning rate too high, too low, or wrong schedule", fault_category="hyperparameter"),
                            FaultTaxonomyNode(id="batch_size", label="Batch Size",
                                description="Batch size incompatible with data or memory", fault_category="hyperparameter"),
                            FaultTaxonomyNode(id="epochs", label="Epochs",
                                description="Insufficient or excessive training epochs", fault_category="hyperparameter"),
                            FaultTaxonomyNode(id="disable_batching", label="Disable Batching",
                                description="Batching disabled when it should be enabled", fault_category="hyperparameter"),
                        ],
                    ),
                    FaultTaxonomyNode(
                        id="regularization",
                        label="Regularization",
                        description="Missing, excessive, or misconfigured regularization",
                        fault_category="regularizer",
                        children=[
                            FaultTaxonomyNode(id="l1_reg", label="L1 Regularization",
                                description="L1 penalty coefficient incorrect", fault_category="regularizer"),
                            FaultTaxonomyNode(id="l2_reg", label="L2 Regularization",
                                description="L2 penalty coefficient incorrect", fault_category="regularizer"),
                            FaultTaxonomyNode(id="dropout", label="Dropout",
                                description="Dropout rate incorrect or placed incorrectly", fault_category="regularizer"),
                        ],
                    ),
                    FaultTaxonomyNode(
                        id="optimization",
                        label="Optimization",
                        description="Wrong optimizer or optimizer misconfiguration",
                        fault_category="optimization",
                        children=[
                            FaultTaxonomyNode(id="optimizer_type", label="Optimizer Type",
                                description="Wrong optimizer algorithm (e.g., SGD vs Adam)", fault_category="optimization"),
                            FaultTaxonomyNode(id="optimizer_params", label="Optimizer Parameters",
                                description="Optimizer momentum, beta, or epsilon incorrect", fault_category="optimization"),
                        ],
                    ),
                    FaultTaxonomyNode(
                        id="loss",
                        label="Loss Function",
                        description="Inappropriate loss function for the task",
                        fault_category="loss",
                        children=[
                            FaultTaxonomyNode(id="loss_type", label="Loss Type",
                                description="Wrong loss function (e.g., MSE for classification)", fault_category="loss"),
                            FaultTaxonomyNode(id="loss_reduction", label="Loss Reduction",
                                description="Incorrect reduction strategy for the loss", fault_category="loss"),
                        ],
                    ),
                ],
            ),
            FaultTaxonomyNode(
                id="model_faults",
                label="Model Faults",
                description="Faults in the model architecture or construction",
                children=[
                    FaultTaxonomyNode(
                        id="layer",
                        label="Layer",
                        description="Incorrect layer type, configuration, or missing layers",
                        fault_category="layer",
                        children=[
                            FaultTaxonomyNode(id="kernel_size", label="Kernel Size",
                                description="Convolutional kernel size incorrect", fault_category="layer"),
                            FaultTaxonomyNode(id="filter_size", label="Filter Size",
                                description="Number of filters incorrect", fault_category="layer"),
                            FaultTaxonomyNode(id="strides_size", label="Strides Size",
                                description="Stride configuration incorrect", fault_category="layer"),
                            FaultTaxonomyNode(id="padding", label="Padding",
                                description="Padding type or amount incorrect", fault_category="layer"),
                            FaultTaxonomyNode(id="pooling_size", label="Pooling Size",
                                description="Pooling window size incorrect", fault_category="layer"),
                            FaultTaxonomyNode(id="output_shape", label="Output Shape",
                                description="Output shape mismatch between layers", fault_category="layer"),
                            FaultTaxonomyNode(id="layer_type", label="Layer Type",
                                description="Wrong layer type used", fault_category="layer"),
                            FaultTaxonomyNode(id="layer_number", label="Layer Number",
                                description="Too few or too many layers", fault_category="layer"),
                            FaultTaxonomyNode(id="neurons", label="Neurons",
                                description="Neuron count in a layer is incorrect", fault_category="layer"),
                        ],
                    ),
                    FaultTaxonomyNode(
                        id="activation",
                        label="Activation Function",
                        description="Wrong activation function for the layer or task",
                        fault_category="activation",
                        children=[
                            FaultTaxonomyNode(id="activation_output", label="Output Activation",
                                description="Output layer activation incompatible with loss/task", fault_category="activation"),
                            FaultTaxonomyNode(id="activation_hidden", label="Hidden Activation",
                                description="Hidden layer activation causes vanishing/exploding", fault_category="activation"),
                        ],
                    ),
                    FaultTaxonomyNode(
                        id="weights",
                        label="Weight Initialization",
                        description="Poor or incorrect weight initialization strategy",
                        fault_category="weights",
                        children=[
                            FaultTaxonomyNode(id="weight_init", label="Initialization Strategy",
                                description="Wrong initializer (e.g., zeros, random_normal)", fault_category="weights"),
                            FaultTaxonomyNode(id="weight_scale", label="Weight Scale",
                                description="Weight magnitude scale incorrect", fault_category="weights"),
                        ],
                    ),
                ],
            ),
        ],
    )

    return FaultTaxonomyResponse(root=root)
