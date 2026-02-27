from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
import os
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from default_tool.inference import explain_static_features, load_models, predict_frame
from default_tool.static_features import extract_static_features_from_keras
from .project_runner import precheck_project_package, run_project_analysis
from .schemas import (
    AnalyzeCodeRequest,
    AnalyzeCodeResponse,
    FaultTaxonomyResponse,
    FullAnalysisResponse,
    TrainingHistoryRequest,
)
from .code_sandbox import extract_features_from_code
from .training_runner import stream_training
from .inference_utils import (
    build_stage1_result,
    build_stage2_result,
    derive_dynamic_features,
    explain_static_signed,
    get_fault_taxonomy,
)

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

MODELS: dict[str, dict[str, Any]] | None = None
ALLOW_UNTRUSTED_CODE = os.getenv("DEFAULT_ALLOW_UNTRUSTED_CODE", "0") == "1"


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    global MODELS
    MODELS = load_models()
    yield


app = FastAPI(title="DEFault Web Tool", version="2.0.0", lifespan=lifespan)

# Allow Next.js dev server (port 3000) to call the API during development.
# In production the frontend is served by FastAPI from the same origin,
# so no CORS header is needed for production traffic.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_csv_upload(file: UploadFile) -> pd.DataFrame:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name.")
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")
    try:
        raw = file.file.read()
        frame = pd.read_csv(BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc
    if frame.empty:
        raise HTTPException(status_code=400, detail="CSV has no rows.")
    return frame


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "models_loaded": bool(MODELS),
        "model_count": len(MODELS or {}),
        "project_execution_enabled": ALLOW_UNTRUSTED_CODE,
        "version": "2.0.0",
    }


# ── New endpoints for the professional UI ────────────────────────────────────

@app.post("/api/analyze-code", response_model=AnalyzeCodeResponse)
def analyze_code(req: AnalyzeCodeRequest) -> AnalyzeCodeResponse:
    """
    Stage 3 only: Accept pasted Keras model code, execute in a sandbox,
    extract 31 static features, and run SHAP root cause analysis.
    Instant results — no training required.
    """
    if MODELS is None:
        raise HTTPException(status_code=503, detail="Models are not loaded.")
    if not req.code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty.")

    try:
        features_dict, warnings = extract_features_from_code(req.code, req.model_name)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    features_df = pd.DataFrame([features_dict])
    try:
        static_result = explain_static_signed(features_df, MODELS, top_n=7)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Static analysis failed: {exc}") from exc

    return AnalyzeCodeResponse(
        analysis_mode="static_only",
        model_name=req.model_name,
        stage3_static=static_result,
        all_features=features_dict,
        warnings=warnings,
    )


@app.post("/api/analyze-history", response_model=FullAnalysisResponse)
def analyze_history(req: TrainingHistoryRequest) -> FullAnalysisResponse:
    """
    Full 3-stage diagnosis from Keras training history (loss/val_loss/acc/val_acc).
    Optionally also runs Stage 3 static analysis if model code is provided.
    """
    if MODELS is None:
        raise HTTPException(status_code=503, detail="Models are not loaded.")

    lengths = {len(req.loss), len(req.val_loss), len(req.train_acc), len(req.val_acc)}
    if len(lengths) > 1:
        raise HTTPException(
            status_code=400,
            detail="loss, val_loss, train_acc and val_acc must all have the same number of epochs.",
        )
    if not req.loss:
        raise HTTPException(status_code=400, detail="Training history must have at least 1 epoch.")

    warnings: list[str] = []

    # Derive dynamic features from training history
    try:
        dynamic_df, training_summary = derive_dynamic_features(
            req.loss, req.val_loss, req.train_acc, req.val_acc
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process training history: {exc}") from exc

    # Stage 1 + Stage 2 from dynamic features
    try:
        raw_result = predict_frame(dynamic_df, MODELS, source_name=req.model_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Dynamic analysis failed: {exc}") from exc

    stage1 = build_stage1_result(raw_result["classifiers"])
    stage2 = build_stage2_result(raw_result["classifiers"], raw_result["detected_fault"])

    # Stage 3: static analysis (only if code provided)
    static_result = None
    analysis_mode = "partial_dynamic"
    if req.code and req.code.strip():
        try:
            features_dict, code_warnings = extract_features_from_code(req.code, req.model_name)
            warnings.extend(code_warnings)
            features_df = pd.DataFrame([features_dict])
            static_result = explain_static_signed(features_df, MODELS, top_n=7)
            analysis_mode = "full"
        except RuntimeError as exc:
            warnings.append(f"Static analysis skipped: {exc}")
        except Exception as exc:
            warnings.append(f"Static analysis error: {exc}")

    if training_summary.missing_dynamic_features:
        warnings.append(
            f"{len(training_summary.missing_dynamic_features)} dynamic features "
            "unavailable (gradient/GPU stats require the DEFault SDK callback). "
            "Detection accuracy may be reduced."
        )

    return FullAnalysisResponse(
        analysis_mode=analysis_mode,
        model_name=req.model_name,
        stage1_detection=stage1,
        stage2_categories=stage2,
        stage3_static=static_result,
        training_summary=training_summary,
        warnings=warnings,
    )


@app.get("/api/fault-taxonomy", response_model=FaultTaxonomyResponse)
def fault_taxonomy() -> FaultTaxonomyResponse:
    """Return the DNN fault hierarchy tree (paper Fig. 3) for the frontend visualization."""
    return get_fault_taxonomy()


@app.post("/api/train-and-diagnose")
async def train_and_diagnose(
    code: str = Form(...),
    model_name: str = Form("pasted_model"),
    data_mode: str = Form("dummy"),
    epochs: int = Form(5),
    num_samples: int = Form(200),
    dataset_file: UploadFile = File(None),
) -> StreamingResponse:
    """
    Train the user's Keras model and stream per-epoch progress as SSE,
    then run the full 3-stage DEFault diagnosis.

    data_mode: "dummy"    — generate random data matching model input shape
               "uploaded" — use the uploaded CSV / .npy / .npz file
    """
    if MODELS is None:
        raise HTTPException(status_code=503, detail="Models are not loaded.")
    if not code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty.")
    if data_mode not in ("dummy", "uploaded"):
        raise HTTPException(status_code=400, detail="data_mode must be 'dummy' or 'uploaded'.")

    epochs      = max(5, min(int(epochs), 50))
    num_samples = max(50, min(int(num_samples), 2000))

    # Handle uploaded dataset file
    temp_dir_ctx = None
    temp_file_path: str | None = None

    if data_mode == "uploaded":
        if dataset_file is None or not dataset_file.filename:
            raise HTTPException(
                status_code=400,
                detail="A dataset file is required when data_mode='uploaded'.",
            )
        fname = dataset_file.filename
        if not any(fname.lower().endswith(ext) for ext in (".csv", ".npy", ".npz")):
            raise HTTPException(
                status_code=400,
                detail="Dataset must be .csv, .npy, or .npz.",
            )
        import tempfile as _tf
        temp_dir_ctx = _tf.TemporaryDirectory(prefix="default_train_")
        temp_file_path = str(Path(temp_dir_ctx.name) / Path(fname).name)
        Path(temp_file_path).write_bytes(await dataset_file.read())

    data_config = {
        "mode":        data_mode,
        "file_path":   temp_file_path,
        "num_samples": num_samples,
    }

    async def event_stream():
        try:
            async for chunk in stream_training(code, data_config, epochs, model_name):
                yield chunk
        finally:
            if temp_dir_ctx is not None:
                temp_dir_ctx.cleanup()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


# ── Serve Next.js static export ──────────────────────────────────────────────
# Must be LAST — API routes defined above take precedence over the mount.
# StaticFiles(html=True) serves index.html for any path not matched by a file,
# which enables client-side routing and serves _next/static/ CSS/JS correctly.
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="frontend")
else:
    @app.get("/", response_class=HTMLResponse)
    def home() -> str:  # pragma: no cover
        return (
            "<h1>DEFault API is running. Frontend not yet built.</h1>"
            "<p>Run: npm --prefix webapp/frontend run build</p>"
        )


@app.post("/api/predict")
def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if MODELS is None:
        raise HTTPException(status_code=503, detail="Models are not loaded.")
    frame = _read_csv_upload(file)
    try:
        result = predict_frame(frame, MODELS, source_name=file.filename or "uploaded.csv")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.post("/api/explain-static")
def explain_static(file: UploadFile = File(...)) -> dict[str, Any]:
    if MODELS is None:
        raise HTTPException(status_code=503, detail="Models are not loaded.")
    frame = _read_csv_upload(file)
    first_row = frame.head(1)
    try:
        report = explain_static_features(first_row, MODELS, top_n=5)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return report


@app.post("/api/explain-model")
def explain_model(file: UploadFile = File(...)) -> dict[str, Any]:
    if MODELS is None:
        raise HTTPException(status_code=503, detail="Models are not loaded.")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name.")
    if not (file.filename.lower().endswith(".h5") or file.filename.lower().endswith(".keras")):
        raise HTTPException(status_code=400, detail="Upload a .h5 or .keras model.")
    try:
        with tempfile.TemporaryDirectory(prefix="default_model_") as tmp:
            model_path = Path(tmp) / Path(file.filename).name
            model_path.write_bytes(file.file.read())
            features = extract_static_features_from_keras(model_path)
            report = explain_static_features(features, MODELS, top_n=5)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return report


@app.post("/api/analyze-project")
def analyze_project(
    file: UploadFile = File(...),
    epochs: int = 5,
) -> dict[str, Any]:
    if MODELS is None:
        raise HTTPException(status_code=503, detail="Models are not loaded.")
    if not ALLOW_UNTRUSTED_CODE:
        raise HTTPException(
            status_code=403,
            detail=(
                "Project execution is disabled. "
                "Set DEFAULT_ALLOW_UNTRUSTED_CODE=1 to enable it in a sandboxed environment."
            ),
        )
    if not file.filename:
        raise HTTPException(status_code=400, detail="Please upload a project file.")
    if not (
        file.filename.lower().endswith(".zip") or file.filename.lower().endswith(".py")
    ):
        raise HTTPException(status_code=400, detail="Upload .zip project or .py adapter.")
    raw = file.file.read()
    precheck = precheck_project_package(zip_bytes=raw, filename=file.filename, adapter_path=None)
    if not precheck["can_run"]:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Project pre-check failed.",
                "precheck": precheck,
            },
        )
    try:
        payload = run_project_analysis(
            zip_bytes=raw,
            filename=file.filename,
            models=MODELS,
            adapter_path=None,
            epochs=max(1, min(int(epochs), 50)),
            timeout_seconds=300,
            max_train_batches=20,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return payload


@app.post("/api/project-precheck")
def project_precheck(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Please upload a project file.")
    if not (
        file.filename.lower().endswith(".zip") or file.filename.lower().endswith(".py")
    ):
        raise HTTPException(status_code=400, detail="Upload .zip project or .py adapter.")
    try:
        report = precheck_project_package(
            zip_bytes=file.file.read(),
            filename=file.filename,
            adapter_path=None,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return report
