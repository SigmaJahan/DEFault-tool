"""
Streaming model training runner.

Accepts user-pasted Keras model code + dataset (uploaded or dummy),
trains the model in an isolated subprocess, streams per-epoch progress
via multiprocessing.Queue → FastAPI SSE, then runs the full 3-stage
DEFault diagnosis and returns the complete FullAnalysisResponse.

Key design decisions:
- TF is NEVER imported in the FastAPI parent process (avoids CUDA init)
- All TF code lives exclusively inside training_subprocess_worker
- mp.Queue is polled with run_in_executor so asyncio event loop is not blocked
- RuntimeFeatureCallback is redefined locally (same logic as project_runner.py)
"""
from __future__ import annotations

import ast
import asyncio
import json
import multiprocessing as mp
import sys
import textwrap
from pathlib import Path
from queue import Empty
from typing import Any, AsyncGenerator

import numpy as np
import pandas as pd

from default_tool.config import DYNAMIC_FEATURE_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── Data helpers (no TF — called from subprocess after TF import) ──────────

def _generate_dummy_data(model: Any, num_samples: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate random synthetic data matching the model's input/output shape.
    Handles Dense (tabular), Conv2D (image), and LSTM/GRU (sequence) models.
    """
    input_shape = model.input_shape    # e.g. (None, 28, 28, 1)
    output_shape = model.output_shape  # e.g. (None, 10)

    # Strip batch dim; replace remaining None → reasonable defaults
    def _fill_none(shape: tuple, fallback_seq: int = 50, fallback_spatial: int = 32) -> tuple:
        result = []
        ndim = len(shape)
        for i, d in enumerate(shape):
            if d is not None:
                result.append(d)
            elif ndim >= 3 and i == 0:   # sequence-length dim in (timesteps, features)
                result.append(fallback_seq)
            else:
                result.append(fallback_spatial)
        return tuple(result)

    x_shape = _fill_none(input_shape[1:])
    y_raw_shape = output_shape[1:]

    X = np.random.random((num_samples,) + x_shape).astype(np.float32)

    # Scale image-like inputs to [0, 1] — already done above
    output_units = int(np.prod([d for d in y_raw_shape if d is not None])) if y_raw_shape else 1

    # Detect output type from last layer activation
    try:
        last_layer = model.layers[-1]
        act = getattr(last_layer, "activation", None)
        act_name = getattr(act, "__name__", "") if act else ""
    except Exception:
        act_name = ""

    if act_name == "sigmoid" or output_units == 1:
        Y = np.random.randint(0, 2, (num_samples, 1)).astype(np.float32)
    elif act_name == "softmax" or output_units > 1:
        labels = np.random.randint(0, output_units, num_samples)
        Y = np.eye(output_units, dtype=np.float32)[labels]
    else:
        Y = np.random.random((num_samples,) + tuple(
            d if d is not None else 1 for d in y_raw_shape
        )).astype(np.float32)

    return X, Y


def _parse_uploaded_csv(file_path: str, model: Any) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError("Uploaded CSV has no rows.")
    X_raw = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = df.iloc[:, -1].values

    # Validate feature count against model input shape
    input_shape = model.input_shape[1:]
    expected = int(np.prod([d for d in input_shape if d is not None]))
    if X_raw.shape[1] != expected:
        raise ValueError(
            f"CSV has {X_raw.shape[1]} feature column(s) but model input shape "
            f"{input_shape} requires {expected} feature(s). "
            "Ensure all columns except the last are features, and the last column is the label."
        )

    output_units = model.output_shape[-1] if model.output_shape[-1] is not None else 1
    n_classes = len(np.unique(y_raw))
    if output_units > 1 and n_classes <= output_units:
        Y = np.eye(int(output_units), dtype=np.float32)[y_raw.astype(int)]
    else:
        Y = y_raw.reshape(-1, 1).astype(np.float32)

    return X_raw, Y


def _parse_uploaded_numpy(file_path: str, model: Any) -> tuple[np.ndarray, np.ndarray]:
    if file_path.lower().endswith(".npz"):
        data = np.load(file_path, allow_pickle=False)
        keys = list(data.keys())
        if "X" in data and "y" in data:
            X, Y = data["X"], data["y"]
        elif "x_train" in data and "y_train" in data:
            X, Y = data["x_train"], data["y_train"]
        elif len(keys) >= 2:
            X, Y = data[keys[0]], data[keys[1]]
        else:
            raise ValueError(f"NPZ must have X/y or x_train/y_train keys. Found: {keys}")
    else:
        arr = np.load(file_path, allow_pickle=False)
        if arr.ndim < 2:
            raise ValueError("NPY array must be at least 2D (samples × features+label).")
        X = arr[:, :-1]
        Y = arr[:, -1:]

    return X.astype(np.float32), Y.astype(np.float32)


# ── Subprocess worker ───────────────────────────────────────────────────────

def training_subprocess_worker(
    queue: mp.Queue,
    code: str,
    data_config: dict[str, Any],
    epochs: int,
    model_name: str,
) -> None:
    """
    Runs inside mp.Process. Imports TF, trains the model, collects all 20
    dynamic features via RuntimeFeatureCallback, then runs full 3-stage
    DEFault diagnosis and puts the complete result into the queue.
    """
    try:
        import os
        import time
        import traceback as tb

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU
        sys.path.insert(0, str(REPO_ROOT))

        import psutil
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")

        # ── 1. Parse and execute user code ─────────────────────────────────
        try:
            ast.parse(code)
        except SyntaxError as e:
            queue.put({"type": "error", "message": f"Syntax error in your code: {e}"}); return

        tree = ast.parse(code)
        has_build = any(
            isinstance(n, ast.FunctionDef) and n.name == "build_model"
            for n in ast.walk(tree)
        )
        if not has_build:
            indented = textwrap.indent(code.strip(), "    ")
            code = f"def build_model():\n{indented}\n    return model\n"

        ns: dict[str, Any] = {}
        exec(compile(code, "<user_code>", "exec"), ns)  # noqa: S102

        if "build_model" not in ns or not callable(ns["build_model"]):
            queue.put({"type": "error", "message": "No build_model() function found."}); return

        try:
            model = ns["build_model"]()
        except Exception as e:
            queue.put({"type": "error", "message": f"build_model() raised an error: {e}",
                       "traceback": tb.format_exc()}); return

        if not hasattr(model, "optimizer") or model.optimizer is None:
            queue.put({"type": "error",
                       "message": "build_model() must return a compiled Keras model "
                                  "(call model.compile() before returning)."}); return

        # ── 2. Load or generate training data ───────────────────────────────
        mode = data_config.get("mode", "dummy")
        num_samples = int(data_config.get("num_samples", 200))

        try:
            if mode == "dummy":
                X, Y = _generate_dummy_data(model, num_samples=num_samples)
            else:
                fp = data_config.get("file_path", "")
                if fp.lower().endswith(".csv"):
                    X, Y = _parse_uploaded_csv(fp, model)
                else:
                    X, Y = _parse_uploaded_numpy(fp, model)
        except Exception as e:
            queue.put({"type": "error", "message": f"Data preparation failed: {e}",
                       "traceback": tb.format_exc()}); return

        # 80/20 train/val split
        split = max(1, int(len(X) * 0.8))
        X_train, X_val = X[:split], X[split:]
        Y_train, Y_val = Y[:split], Y[split:]
        sample_x = tf.constant(X_train[:min(16, len(X_train))], dtype=tf.float32)
        sample_y = tf.constant(Y_train[:min(16, len(Y_train))], dtype=tf.float32)

        # ── 3. Define callbacks ─────────────────────────────────────────────

        class EpochStreamCallback(tf.keras.callbacks.Callback):
            """Streams per-epoch metrics to the parent process via queue."""
            def __init__(self) -> None:
                super().__init__()
                self._start = time.time()

            def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
                logs = logs or {}
                queue.put({
                    "type": "epoch",
                    "epoch": epoch + 1,
                    "total": epochs,
                    "loss":    float(logs.get("loss", 0.0)),
                    "val_loss": float(logs.get("val_loss", 0.0)),
                    "acc":     float(logs.get("accuracy", logs.get("acc", 0.0))),
                    "val_acc": float(logs.get("val_accuracy", logs.get("val_acc", 0.0))),
                    "elapsed_ms": int((time.time() - self._start) * 1000),
                })

        # Local helpers for RuntimeFeatureCallback (TF available here)
        def _read_lr(m: Any) -> float:
            lr = getattr(m.optimizer, "learning_rate", 0.0)
            try:
                return float(tf.keras.backend.get_value(lr))
            except Exception:
                return 0.0

        def _grad_stats(m: Any) -> dict[str, float]:
            try:
                with tf.GradientTape() as tape:
                    preds = m(sample_x, training=True)
                    loss_fn = tf.keras.losses.get(m.loss)
                    lv = tf.reduce_mean(loss_fn(sample_y, preds))
                grads = tape.gradient(lv, m.trainable_variables)
                vecs = [tf.reshape(g, [-1]).numpy() for g in grads
                        if g is not None and tf.size(g) > 0]
                if not vecs:
                    zero = {k: 0.0 for k in ["mean_grad", "std_grad", "gradient_std",
                            "gradient_max", "gradient_min", "gradient_median",
                            "gradient_vanish", "gradient_explode"]}
                    return zero
                vals = np.abs(np.concatenate(vecs))
                return {
                    "mean_grad":        float(np.mean(vals)),
                    "std_grad":         float(np.std(vals)),
                    "gradient_std":     float(np.std(vals)),
                    "gradient_max":     float(np.max(vals)),
                    "gradient_min":     float(np.min(vals)),
                    "gradient_median":  float(np.median(vals)),
                    "gradient_vanish":  float(np.mean(vals) < 1e-4),
                    "gradient_explode": float(np.max(vals) > 70.0),
                }
            except Exception:
                return {k: 0.0 for k in ["mean_grad", "std_grad", "gradient_std",
                        "gradient_max", "gradient_min", "gradient_median",
                        "gradient_vanish", "gradient_explode"]}

        def _activation_stats(m: Any) -> dict[str, float]:
            try:
                layer_outputs = []
                relu_flags = []
                for layer in m.layers:
                    try:
                        layer_outputs.append(layer.output)
                        act = getattr(layer, "activation", None)
                        relu_flags.append(getattr(act, "__name__", "") == "relu")
                    except AttributeError:
                        continue
                if not layer_outputs:
                    return {"mean_activation": 0.0, "std_activation": 0.0, "dying_relu": 0.0}
                probe = tf.keras.Model(inputs=m.inputs, outputs=layer_outputs)
                probed = probe(sample_x[:min(8, len(sample_x))], training=False)
                if not isinstance(probed, (tuple, list)):
                    probed = [probed]
                flat, dead = [], []
                for i, t in enumerate(probed):
                    arr = np.array(t).astype(float)
                    if arr.size == 0:
                        continue
                    flat.append(arr.reshape(-1))
                    if i < len(relu_flags) and relu_flags[i]:
                        dead.append(float(np.mean(arr <= 0.0)))
                if not flat:
                    return {"mean_activation": 0.0, "std_activation": 0.0, "dying_relu": 0.0}
                merged = np.concatenate(flat)
                merged = merged[np.isfinite(merged)]
                return {
                    "mean_activation": float(np.mean(merged)),
                    "std_activation":  float(np.std(merged)),
                    "dying_relu":      float((max(dead) if dead else 0.0) > 0.7),
                }
            except Exception:
                return {"mean_activation": 0.0, "std_activation": 0.0, "dying_relu": 0.0}

        class RuntimeFeatureCallback(tf.keras.callbacks.Callback):
            """Collects all 20 DYNAMIC_FEATURE_COLUMNS per epoch."""
            def __init__(self) -> None:
                super().__init__()
                self.rows: list[dict[str, float]] = []
                self._prev_loss: float | None = None
                self._prev_acc: float | None = None

            def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
                logs = logs or {}
                train_acc  = float(logs.get("accuracy",     logs.get("acc",     np.nan)))
                val_acc    = float(logs.get("val_accuracy", logs.get("val_acc", np.nan)))
                train_loss = float(logs.get("loss",    np.nan))
                val_loss_v = float(logs.get("val_loss", np.nan))

                try:
                    weights = np.concatenate(
                        [w.reshape(-1) for w in self.model.get_weights() if np.size(w) > 0]
                    )
                    large_w = float(np.sum(np.abs(weights) > 10.0))
                except Exception:
                    large_w = 0.0

                acc_gap = (
                    float(abs(train_acc - val_acc) > 0.1)
                    if np.isfinite(train_acc) and np.isfinite(val_acc) else 0.0
                )
                loss_osc = (
                    float(self._prev_loss is not None
                          and np.isfinite(train_loss)
                          and abs(train_loss - self._prev_loss) > 0.01)
                )
                self._prev_loss = train_loss if np.isfinite(train_loss) else self._prev_loss

                # GPU memory (0 on CPU-only)
                gpu_mem = 0.0
                try:
                    if tf.config.list_physical_devices("GPU"):
                        info = tf.config.experimental.get_memory_info("GPU:0")
                        gpu_mem = float(info.get("current", 0.0) / (1024 ** 2))
                except Exception:
                    pass

                grad = _grad_stats(self.model)
                act  = _activation_stats(self.model)

                row: dict[str, float] = {
                    "gpu_memory_utilization": gpu_mem,
                    "cpu_utilization":  float(psutil.cpu_percent(interval=None)),
                    "train_acc":        train_acc if np.isfinite(train_acc) else 0.0,
                    "val_acc":          val_acc   if np.isfinite(val_acc)   else 0.0,
                    "memory_usage":     float(psutil.virtual_memory().percent),
                    "loss_oscillation": loss_osc,
                    "acc_gap_too_big":  acc_gap,
                    "adjusted_lr":      _read_lr(self.model),
                    "dying_relu":       act["dying_relu"],
                    "gradient_vanish":  grad["gradient_vanish"],
                    "gradient_explode": grad["gradient_explode"],
                    "gradient_median":  grad["gradient_median"],
                    "std_grad":         grad["std_grad"],
                    "gradient_min":     grad["gradient_min"],
                    "mean_grad":        grad["mean_grad"],
                    "large_weight_count": large_w,
                    "mean_activation":  act["mean_activation"],
                    "std_activation":   act["std_activation"],
                    "gradient_std":     grad["gradient_std"],
                    "gradient_max":     grad["gradient_max"],
                    # Extra (not in DYNAMIC_FEATURE_COLUMNS) for TrainingSummary:
                    "_loss":     train_loss if np.isfinite(train_loss) else 0.0,
                    "_val_loss": val_loss_v  if np.isfinite(val_loss_v)  else 0.0,
                }
                self.rows.append(row)

        stream_cb  = EpochStreamCallback()
        runtime_cb = RuntimeFeatureCallback()

        # ── 4. Train ─────────────────────────────────────────────────────────
        try:
            model.fit(
                X_train, Y_train,
                validation_data=(X_val, Y_val),
                epochs=epochs,
                batch_size=min(32, max(1, len(X_train) // 4)),
                verbose=0,
                callbacks=[stream_cb, runtime_cb],
            )
        except Exception as e:
            queue.put({"type": "error", "message": f"Training failed: {e}",
                       "traceback": tb.format_exc()}); return

        queue.put({"type": "training_done", "epochs_completed": min(len(runtime_cb.rows), epochs)})
        queue.put({"type": "analyzing", "stage": 1})

        # ── 5. Build dynamic feature DataFrame from collected rows ───────────
        if not runtime_cb.rows:
            queue.put({"type": "error", "message": "No epoch data collected during training."}); return

        dynamic_df = pd.DataFrame(runtime_cb.rows)[DYNAMIC_FEATURE_COLUMNS].copy()

        # ── 6. Extract static features from trained model ───────────────────
        queue.put({"type": "analyzing", "stage": 2})
        try:
            from default_tool.static_features import extract_static_features_from_model
            static_df = extract_static_features_from_model(model, model_name=model_name)
        except Exception as e:
            queue.put({"type": "error", "message": f"Static feature extraction failed: {e}",
                       "traceback": tb.format_exc()}); return

        queue.put({"type": "analyzing", "stage": 3})

        # ── 7. Run full 3-stage diagnosis ────────────────────────────────────
        try:
            from default_tool.inference import load_models, predict_frame
            from webapp.inference_utils import (
                build_stage1_result,
                build_stage2_result,
                explain_static_signed,
            )
            from webapp.schemas import FullAnalysisResponse, TrainingSummary

            models = load_models()
            raw = predict_frame(dynamic_df, models, source_name=model_name)
            stage1 = build_stage1_result(raw["classifiers"])
            stage2 = build_stage2_result(raw["classifiers"], raw["detected_fault"])
            stage3 = explain_static_signed(static_df, models, top_n=7)
        except Exception as e:
            queue.put({"type": "error", "message": f"Diagnosis failed: {e}",
                       "traceback": tb.format_exc()}); return

        # ── 8. Build TrainingSummary from per-epoch rows ─────────────────────
        rows = runtime_cb.rows
        loss_series     = [r["_loss"]     for r in rows]
        val_loss_series = [r["_val_loss"] for r in rows]
        acc_series      = [r["train_acc"] for r in rows]
        val_acc_series  = [r["val_acc"]   for r in rows]
        loss_diffs = [loss_series[i] - loss_series[i - 1]
                      for i in range(1, len(loss_series))]

        training_summary = TrainingSummary(
            epochs=len(rows),
            final_train_acc=float(acc_series[-1]),
            final_val_acc=float(val_acc_series[-1]),
            final_loss=float(loss_series[-1]),
            final_val_loss=float(val_loss_series[-1]),
            loss_oscillation=float(np.std(loss_diffs)) if loss_diffs else 0.0,
            acc_gap=float(val_acc_series[-1] - acc_series[-1]),
            decrease_acc_count=sum(
                1 for i in range(1, len(acc_series)) if acc_series[i] < acc_series[i - 1]
            ),
            increase_loss_count=sum(
                1 for i in range(1, len(loss_series)) if loss_series[i] > loss_series[i - 1]
            ),
            available_dynamic_features=list(DYNAMIC_FEATURE_COLUMNS),
            missing_dynamic_features=[],
        )

        full_response = FullAnalysisResponse(
            analysis_mode="full_training",
            model_name=model_name,
            stage1_detection=stage1,
            stage2_categories=stage2,
            stage3_static=stage3,
            training_summary=training_summary,
            warnings=[],
        )

        queue.put({
            "type":      "complete",
            "result":    full_response.model_dump(),
            "data_mode": "dummy" if mode == "dummy" else "uploaded",
        })

    except Exception as exc:
        import traceback as tb2
        queue.put({"type": "error", "message": str(exc), "traceback": tb2.format_exc()})


# ── Async SSE generator ─────────────────────────────────────────────────────

async def stream_training(
    code: str,
    data_config: dict[str, Any],
    epochs: int,
    model_name: str,
) -> AsyncGenerator[str, None]:
    """
    Async generator that spawns a training subprocess and yields
    SSE-formatted lines ("data: {...}\\n\\n") as events arrive.
    """
    queue: mp.Queue = mp.Queue()
    process = mp.Process(
        target=training_subprocess_worker,
        args=(queue, code, data_config, epochs, model_name),
        daemon=True,
    )
    process.start()
    loop = asyncio.get_event_loop()

    try:
        while True:
            try:
                item: dict[str, Any] = await loop.run_in_executor(
                    None, lambda: queue.get(timeout=2.0)
                )
                yield f"data: {json.dumps(item)}\n\n"
                if item.get("type") in ("complete", "error"):
                    break
            except Empty:
                if not process.is_alive():
                    # Process finished without sending a complete/error — unusual
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Training process exited unexpectedly.'})}\n\n"
                    break
                # Still running, just no data yet — keep polling
                await asyncio.sleep(0.05)
    finally:
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
