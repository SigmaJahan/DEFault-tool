from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import KERAS_ACTIVATIONS, KERAS_LAYER_TYPES


def _iter_dims(shape: Any) -> list[int]:
    values: list[int] = []
    if shape is None:
        return values
    if isinstance(shape, tuple):
        for item in shape:
            values.extend(_iter_dims(item))
        return values
    if isinstance(shape, list):
        for item in shape:
            values.extend(_iter_dims(item))
        return values
    if isinstance(shape, int):
        values.append(shape)
    return values


def _max_min_dimension(shapes: list[Any]) -> tuple[int, int]:
    dims: list[int] = []
    for shape in shapes:
        dims.extend([d for d in _iter_dims(shape) if d is not None and d >= 0])
    if not dims:
        return 0, 0
    return int(max(dims)), int(min(dims))


def extract_static_features_from_model(
    model: Any, *, model_name: str = "uploaded_model"
) -> pd.DataFrame:
    import tensorflow as tf

    layer_count = {f"Count{name}": 0 for name in KERAS_LAYER_TYPES}
    activation_count = {f"Count{name}": 0 for name in KERAS_ACTIVATIONS}
    regularization_count = {"CountL1": 0, "CountL2": 0, "CountL1L2": 0}

    total_params = 0
    trainable_params = 0
    neurons: list[int] = []
    input_shapes: list[Any] = []
    output_shapes: list[Any] = []
    dropout_rates: list[float] = []
    activation_presence = 0
    input_output_mismatch_count = 0

    for layer in model.layers:
        layer_type = layer.__class__.__name__
        key = f"Count{layer_type}"
        if key in layer_count:
            layer_count[key] += 1

        if hasattr(layer, "activation") and callable(layer.activation):
            activation_name = layer.activation.__name__
            activation_key = f"Count{activation_name}"
            if activation_key in activation_count:
                activation_count[activation_key] += 1
            activation_presence += 1

        kernel_regularizer = getattr(layer, "kernel_regularizer", None)
        if kernel_regularizer is not None:
            name = kernel_regularizer.__class__.__name__.lower()
            if "l1l2" in name:
                regularization_count["CountL1L2"] += 1
            elif "l1" in name and "l2" not in name:
                regularization_count["CountL1"] += 1
            elif "l2" in name:
                regularization_count["CountL2"] += 1

        params = int(layer.count_params())
        total_params += params
        if getattr(layer, "trainable", False):
            trainable_params += params

        units = getattr(layer, "units", None)
        filters = getattr(layer, "filters", None)
        if units is not None:
            neurons.append(int(units))
        elif filters is not None:
            neurons.append(int(filters))

        if isinstance(layer, tf.keras.layers.Dropout):
            dropout_rates.append(float(layer.rate))

        input_shape = getattr(layer, "input_shape", None)
        if input_shape is not None:
            input_shapes.append(input_shape)
        output_shape = getattr(layer, "output_shape", None)
        if output_shape is not None:
            output_shapes.append(output_shape)

    for idx in range(len(model.layers) - 1):
        current_output = getattr(model.layers[idx], "output_shape", None)
        next_input = getattr(model.layers[idx + 1], "input_shape", None)
        if current_output is not None and next_input is not None:
            if tuple(_iter_dims(current_output)) != tuple(_iter_dims(next_input)):
                input_output_mismatch_count += 1

    max_dim_input, min_dim_input = _max_min_dimension(input_shapes)
    max_dim_output, min_dim_output = _max_min_dimension(output_shapes)
    max_input_output_mismatch_count = int(max_dim_input != max_dim_output)

    row = {
        **layer_count,
        **activation_count,
        **regularization_count,
        "Model_File": model_name,
        "Total_Params": total_params,
        "Trainable_Params": trainable_params,
        "Num_Neurons": len(neurons),
        "Max_Neurons": int(max(neurons) if neurons else 0),
        "Min_Neurons": int(min(neurons) if neurons else 0),
        "Max_Dimension_Input": max_dim_input,
        "Min_Dimension_Input": min_dim_input,
        "Max_Dimension_Output": max_dim_output,
        "Min_Dimension_Output": min_dim_output,
        "Max_Input_Output_Dimension_Mismatch_Count": max_input_output_mismatch_count,
        "Max_Dropout_Rate": float(max(dropout_rates) if dropout_rates else 0.0),
        "Min_Dropout_Rate": float(min(dropout_rates) if dropout_rates else 0.0),
        "Number_of_Layers": int(len(model.layers)),
        "Layer_Diversity": int(sum(1 for v in layer_count.values() if v > 0)),
        "Activation_Presence": (
            float(activation_presence / len(model.layers)) if model.layers else 0.0
        ),
        "Avg_Params_Per_Layer": (
            float(total_params / len(model.layers)) if model.layers else 0.0
        ),
        "Input_Output_Mismatch_Count_Layer": int(input_output_mismatch_count),
    }

    return pd.DataFrame([row]).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def extract_static_features_from_keras(model_path: Path) -> pd.DataFrame:
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)
    return extract_static_features_from_model(model, model_name=Path(model_path).name)
