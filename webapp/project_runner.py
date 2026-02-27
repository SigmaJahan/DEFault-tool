from __future__ import annotations

import ast
import importlib.util
import multiprocessing as mp
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from default_tool.config import DYNAMIC_FEATURE_COLUMNS
from default_tool.inference import explain_static_features, predict_frame
from default_tool.static_features import extract_static_features_from_model


def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            target = (dest_dir / member.filename).resolve()
            if not str(target).startswith(str(dest_dir.resolve())):
                raise ValueError("Zip contains an invalid path.")
        archive.extractall(dest_dir)


def _tensorflow_declared(requirements_file: Path) -> bool:
    for raw in requirements_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip().lower()
        if not line or line.startswith("#"):
            continue
        line = line.split(";")[0].strip()
        if line.startswith("tensorflow") or line.startswith("tf-nightly"):
            return True
    return False


def _has_required_adapter_functions(py_file: Path) -> bool:
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
    except Exception:
        return False
    names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    return {"build_model", "load_data"}.issubset(names)


def _resolve_adapter_path(extract_dir: Path, adapter_path: str | None) -> Path:
    if adapter_path:
        requested = (extract_dir / adapter_path).resolve()
        if requested.exists():
            return requested
        raise FileNotFoundError(f"Adapter file not found: {adapter_path}")

    adapter_named = sorted(extract_dir.rglob("adapter.py"))
    if adapter_named:
        return adapter_named[0]

    for py_file in sorted(extract_dir.rglob("*.py")):
        if _has_required_adapter_functions(py_file):
            return py_file

    raise FileNotFoundError(
        "Could not find adapter. Provide a Python file defining "
        "build_model() and load_data()."
    )


def _stage_project_files(
    *, zip_bytes: bytes, filename: str, temp_path: Path
) -> tuple[Path, str]:
    extract_dir = temp_path / "project"
    extract_dir.mkdir(parents=True, exist_ok=True)
    file_name = Path(filename).name or "project.zip"
    suffix = Path(file_name).suffix.lower()

    if suffix == ".zip":
        zip_path = temp_path / file_name
        zip_path.write_bytes(zip_bytes)
        _safe_extract_zip(zip_path, extract_dir)
        return extract_dir, "zip"
    if suffix == ".py":
        (extract_dir / "adapter.py").write_bytes(zip_bytes)
        return extract_dir, "py"
    raise ValueError("Upload must be a .zip project or a single .py adapter.")


def precheck_project_package(
    *,
    zip_bytes: bytes,
    filename: str,
    adapter_path: str | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="default_precheck_") as temp_dir:
        temp_path = Path(temp_dir)
        extract_dir, upload_kind = _stage_project_files(
            zip_bytes=zip_bytes, filename=filename, temp_path=temp_path
        )

        missing: list[str] = []
        warnings: list[str] = []
        adapter_detected = ""

        try:
            adapter = _resolve_adapter_path(extract_dir, adapter_path)
            adapter_detected = str(adapter.relative_to(extract_dir))
        except Exception as exc:
            missing.append(str(exc))

        tf_runtime_available = importlib.util.find_spec("tensorflow") is not None
        if not tf_runtime_available:
            missing.append("TensorFlow is not installed in the tool runtime.")

        tensorflow_declared = None
        if upload_kind == "zip":
            req_files = sorted(extract_dir.rglob("requirements.txt"))
            if not req_files:
                missing.append(
                    "Project zip must include requirements.txt with tensorflow dependency."
                )
            else:
                tensorflow_declared = _tensorflow_declared(req_files[0])
                if not tensorflow_declared:
                    missing.append(
                        "requirements.txt does not declare tensorflow dependency."
                    )
        else:
            warnings.append(
                "Single .py upload cannot declare dependencies; use .zip + requirements.txt."
            )

        return {
            "can_run": len(missing) == 0,
            "upload_kind": upload_kind,
            "adapter_detected": adapter_detected,
            "tensorflow_runtime_available": tf_runtime_available,
            "tensorflow_declared": tensorflow_declared,
            "missing": missing,
            "warnings": warnings,
        }


def _normalize_data_payload(payload: Any) -> tuple[Any, Any, int]:
    batch_size = 32
    if isinstance(payload, dict):
        train = payload.get("train")
        val = payload.get("val")
        batch_size = int(payload.get("batch_size", 32))
        return train, val, batch_size
    if isinstance(payload, (tuple, list)):
        if len(payload) == 4:
            return (payload[0], payload[1]), (payload[2], payload[3]), 32
        if len(payload) == 2:
            return payload[0], payload[1], 32
    raise ValueError(
        "load_data() must return either "
        "dict(train=..., val=..., batch_size=...) or "
        "((x_train, y_train), (x_val, y_val)) or "
        "(x_train, y_train, x_val, y_val)."
    )


def _analysis_worker(
    queue: mp.Queue,
    extract_dir: str,
    adapter_rel_path: str | None,
    epochs: int,
    max_train_batches: int,
) -> None:
    try:
        import psutil
        import tensorflow as tf

        extract_root = Path(extract_dir)
        adapter_path = _resolve_adapter_path(extract_root, adapter_rel_path)

        spec = importlib.util.spec_from_file_location("default_user_adapter", adapter_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load adapter module: {adapter_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "build_model") or not hasattr(module, "load_data"):
            raise ValueError(
                "Adapter must define build_model() and load_data() functions."
            )

        model = module.build_model()
        train_payload = module.load_data()
        train_raw, val_raw, batch_size = _normalize_data_payload(train_payload)

        def _to_dataset(obj: Any, *, shuffle: bool) -> tf.data.Dataset:
            if isinstance(obj, tf.data.Dataset):
                return obj
            if (
                isinstance(obj, (tuple, list))
                and len(obj) == 2
                and hasattr(obj[0], "__len__")
                and hasattr(obj[1], "__len__")
            ):
                x_data, y_data = obj
                dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
                if shuffle:
                    dataset = dataset.shuffle(min(2048, len(x_data)))
                return dataset.batch(batch_size)
            raise ValueError(
                "Training and validation data must be tf.data.Dataset or (x, y)."
            )

        train_ds = _to_dataset(train_raw, shuffle=True).take(max_train_batches)
        val_ds = None
        if val_raw is not None:
            val_ds = _to_dataset(val_raw, shuffle=False).take(max_train_batches)

        if getattr(model, "optimizer", None) is None or getattr(model, "loss", None) is None:
            raise ValueError("build_model() must return a compiled Keras model.")

        sample_batch = next(iter(train_ds.take(1)), None)
        if sample_batch is None:
            raise ValueError("Training dataset is empty.")
        sample_x, sample_y = sample_batch

        def _read_lr(current_model: Any) -> float:
            lr_value = getattr(current_model.optimizer, "learning_rate", 0.0)
            try:
                if callable(lr_value):
                    return float(lr_value(current_model.optimizer.iterations).numpy())
                return float(tf.keras.backend.get_value(lr_value))
            except Exception:
                return 0.0

        def _grad_stats(current_model: Any) -> dict[str, float]:
            try:
                with tf.GradientTape() as tape:
                    predictions = current_model(sample_x, training=True)
                    loss_fn = tf.keras.losses.get(current_model.loss)
                    loss_value = loss_fn(sample_y, predictions)
                    loss_value = tf.reduce_mean(loss_value)
                    if current_model.losses:
                        loss_value = loss_value + tf.add_n(current_model.losses)
                grads = tape.gradient(loss_value, current_model.trainable_variables)
                vectors = [
                    tf.reshape(g, [-1]).numpy()
                    for g in grads
                    if g is not None and tf.size(g) > 0
                ]
                if not vectors:
                    return {
                        "mean_grad": 0.0,
                        "std_grad": 0.0,
                        "gradient_std": 0.0,
                        "gradient_max": 0.0,
                        "gradient_min": 0.0,
                        "gradient_median": 0.0,
                        "gradient_vanish": 0.0,
                        "gradient_explode": 0.0,
                    }
                values = np.abs(np.concatenate(vectors))
                return {
                    "mean_grad": float(np.mean(values)),
                    "std_grad": float(np.std(values)),
                    "gradient_std": float(np.std(values)),
                    "gradient_max": float(np.max(values)),
                    "gradient_min": float(np.min(values)),
                    "gradient_median": float(np.median(values)),
                    "gradient_vanish": float(np.mean(values) < 1e-4),
                    "gradient_explode": float(np.max(values) > 70),
                }
            except Exception:
                return {
                    "mean_grad": 0.0,
                    "std_grad": 0.0,
                    "gradient_std": 0.0,
                    "gradient_max": 0.0,
                    "gradient_min": 0.0,
                    "gradient_median": 0.0,
                    "gradient_vanish": 0.0,
                    "gradient_explode": 0.0,
                }

        def _activation_stats(current_model: Any) -> dict[str, float]:
            try:
                _ = current_model(sample_x[:1], training=False)
                outputs = []
                relu_outputs = []
                for layer in current_model.layers:
                    if not hasattr(layer, "output"):
                        continue
                    outputs.append(layer.output)
                    activation = getattr(layer, "activation", None)
                    activation_name = getattr(activation, "__name__", "")
                    is_relu = activation_name == "relu" or layer.__class__.__name__ == "ReLU"
                    relu_outputs.append(is_relu)
                if not outputs:
                    return {"mean_activation": 0.0, "std_activation": 0.0, "dying_relu": 0.0}
                probe_model = tf.keras.Model(inputs=current_model.inputs, outputs=outputs)
                probed = probe_model(sample_x[: min(16, len(sample_x))], training=False)
                if not isinstance(probed, (tuple, list)):
                    probed = [probed]
                flat_values = []
                relu_dead_ratios = []
                for idx, tensor in enumerate(probed):
                    array = np.array(tensor).astype(float)
                    if array.size == 0:
                        continue
                    flat_values.append(array.reshape(-1))
                    if idx < len(relu_outputs) and relu_outputs[idx]:
                        relu_dead_ratios.append(float(np.mean(array <= 0)))
                if not flat_values:
                    return {"mean_activation": 0.0, "std_activation": 0.0, "dying_relu": 0.0}
                merged = np.concatenate(flat_values)
                merged = merged[np.isfinite(merged)]
                if merged.size == 0:
                    return {"mean_activation": 0.0, "std_activation": 0.0, "dying_relu": 0.0}
                dead_ratio = max(relu_dead_ratios) if relu_dead_ratios else 0.0
                return {
                    "mean_activation": float(np.mean(merged)),
                    "std_activation": float(np.std(merged)),
                    "dying_relu": float(dead_ratio > 0.7),
                }
            except Exception:
                return {"mean_activation": 0.0, "std_activation": 0.0, "dying_relu": 0.0}

        class RuntimeFeatureCallback(tf.keras.callbacks.Callback):
            def __init__(self) -> None:
                self.rows: list[dict[str, float]] = []
                self.previous_loss: float | None = None

            def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
                logs = logs or {}
                train_acc = float(logs.get("accuracy", logs.get("acc", np.nan)))
                val_acc = float(logs.get("val_accuracy", logs.get("val_acc", np.nan)))
                train_loss = float(logs.get("loss", np.nan))
                weights = np.concatenate(
                    [w.reshape(-1) for w in self.model.get_weights() if np.size(w) > 0]
                )
                large_weight_count = float(np.sum(np.abs(weights) > 10.0)) if weights.size else 0.0
                acc_gap = (
                    float(abs(train_acc - val_acc) > 0.1)
                    if np.isfinite(train_acc) and np.isfinite(val_acc)
                    else 0.0
                )
                loss_oscillation = (
                    float(self.previous_loss is not None and np.isfinite(train_loss) and abs(train_loss - self.previous_loss) > 0.01)
                )
                self.previous_loss = train_loss if np.isfinite(train_loss) else self.previous_loss
                grad = _grad_stats(self.model)
                act = _activation_stats(self.model)
                try:
                    gpu_mem = 0.0
                    if tf.config.list_physical_devices("GPU"):
                        info = tf.config.experimental.get_memory_info("GPU:0")
                        gpu_mem = float(info.get("current", 0.0) / (1024 ** 2))
                except Exception:
                    gpu_mem = 0.0

                row = {
                    "gpu_memory_utilization": gpu_mem,
                    "cpu_utilization": float(psutil.cpu_percent(interval=None)),
                    "train_acc": train_acc if np.isfinite(train_acc) else 0.0,
                    "val_acc": val_acc if np.isfinite(val_acc) else 0.0,
                    "memory_usage": float(psutil.virtual_memory().percent),
                    "loss_oscillation": loss_oscillation,
                    "acc_gap_too_big": acc_gap,
                    "adjusted_lr": _read_lr(self.model),
                    "dying_relu": act["dying_relu"],
                    "gradient_vanish": grad["gradient_vanish"],
                    "gradient_explode": grad["gradient_explode"],
                    "gradient_median": grad["gradient_median"],
                    "std_grad": grad["std_grad"],
                    "gradient_min": grad["gradient_min"],
                    "mean_grad": grad["mean_grad"],
                    "large_weight_count": large_weight_count,
                    "mean_activation": act["mean_activation"],
                    "std_activation": act["std_activation"],
                    "gradient_std": grad["gradient_std"],
                    "gradient_max": grad["gradient_max"],
                }
                self.rows.append(row)

        callback = RuntimeFeatureCallback()
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=0,
            callbacks=[callback],
        )

        dynamic_df = pd.DataFrame(callback.rows)
        if dynamic_df.empty:
            raise RuntimeError("Dynamic feature extraction produced no rows.")
        static_df = extract_static_features_from_model(
            model, model_name=adapter_path.name
        )

        queue.put(
            {
                "ok": True,
                "dynamic_records": dynamic_df.to_dict(orient="records"),
                "static_records": static_df.to_dict(orient="records"),
                "metadata": {
                    "epochs": int(epochs),
                    "adapter": str(adapter_path.relative_to(extract_root)),
                    "dynamic_rows": int(len(dynamic_df)),
                },
            }
        )
    except Exception as exc:
        queue.put({"ok": False, "error": str(exc)})


def run_project_analysis(
    *,
    zip_bytes: bytes,
    filename: str,
    models: dict[str, dict[str, Any]],
    adapter_path: str | None = None,
    epochs: int = 5,
    timeout_seconds: int = 300,
    max_train_batches: int = 20,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="default_project_") as temp_dir:
        temp_path = Path(temp_dir)
        extract_dir, _ = _stage_project_files(
            zip_bytes=zip_bytes, filename=filename, temp_path=temp_path
        )

        queue: mp.Queue = mp.Queue()
        process = mp.Process(
            target=_analysis_worker,
            args=(
                queue,
                str(extract_dir),
                adapter_path,
                int(epochs),
                int(max_train_batches),
            ),
        )
        process.start()
        process.join(timeout=timeout_seconds)

        if process.is_alive():
            process.kill()
            process.join()
            raise TimeoutError(
                f"Project analysis timed out after {timeout_seconds} seconds."
            )
        if queue.empty():
            raise RuntimeError("Project analysis failed without any result.")

        payload = queue.get()
        if not payload.get("ok"):
            raise RuntimeError(payload.get("error", "Project analysis failed."))

        dynamic_df = pd.DataFrame(payload["dynamic_records"])
        static_df = pd.DataFrame(payload["static_records"])
        extracted_cols = dynamic_df.columns.tolist()
        missing_cols = [c for c in DYNAMIC_FEATURE_COLUMNS if c not in extracted_cols]

        prediction = predict_frame(dynamic_df, models, source_name=filename)
        static_explanation = explain_static_features(static_df, models, top_n=5)

        return {
            "metadata": payload["metadata"],
            "callback_validation": {
                "required_columns": DYNAMIC_FEATURE_COLUMNS,
                "extracted_columns": extracted_cols,
                "missing_required_columns": missing_cols,
                "row_count": int(len(dynamic_df)),
                "callback_ok": len(dynamic_df) > 0 and len(missing_cols) == 0,
            },
            "dynamic_features_preview": dynamic_df.head(5).to_dict(orient="records"),
            "stage1_detection": prediction["classifiers"]["detection"],
            "stage2_categories": prediction["predicted_categories"],
            "stage3_static_explanation": static_explanation,
            "full_prediction": prediction,
        }
