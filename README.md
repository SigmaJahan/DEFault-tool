---
title: DEFault Tool
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: Detect and explain faults in deep neural networks (ICSE 2025)
---

# DEFault: Detect and Explain Faults in Deep Neural Networks

A professional web tool for diagnosing faults in Keras/TensorFlow models, based on the ICSE 2025 paper:

> **DEFault: Detect and Explain Faults in Deep Neural Networks using Hierarchical Classification**
> Sigma Jahan, Dalhousie University

---

## What It Does

DEFault uses a 3-stage hierarchical classification pipeline to automatically detect and explain faults in DNN models:

| Stage | Question | Output |
|-------|----------|--------|
| **S1: Fault Detection** | Is the DNN faulty? | Probability + verdict |
| **S2: Fault Categorization** | What type of fault? | Flagged categories |
| **S3: Root Cause Analysis** | Why is it faulty? | SHAP waterfall chart |

Fault categories: Activation, Layer, Hyperparameter, Loss Function, Optimization, Regularization, Weights.

---

## Two Modes

### Check Model (instant, ~3-5 s)
Paste your Keras model code and get a static analysis with SHAP root-cause hints immediately. No training required.

### Train & Diagnose (full, ~30-120 s)
Upload your dataset (CSV, `.npy`, `.npz`) **or** use auto-generated dummy data. DEFault trains your model in an isolated subprocess, streams live per-epoch charts, then runs all 3 diagnostic stages with real gradient, activation, and overfitting signals.

---

## Quick Start (local)

```bash
git clone https://github.com/sigmajahan/DEFault-tool.git
cd DEFault-tool

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install TensorFlow for "Train & Diagnose" (optional but recommended)
pip install "tensorflow>=2.16,<3"

# Build the frontend (requires Node.js 18+)
npm --prefix webapp/frontend ci
npm --prefix webapp/frontend run build

# Start the server
uvicorn webapp.main:app --host 0.0.0.0 --port 8000
```

Open: `http://localhost:8000`

> **Dev mode** (hot-reload frontend):
> ```bash
> # Terminal 1: backend
> uvicorn webapp.main:app --host 0.0.0.0 --port 8001 --reload
> # Terminal 2: frontend
> npm --prefix webapp/frontend run dev
> ```
> Then open `http://localhost:3000`

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Server status and model count |
| `POST` | `/api/analyze-code` | Static analysis from Keras code |
| `POST` | `/api/analyze-history` | Full analysis from training history |
| `GET`  | `/api/fault-taxonomy` | DNN fault hierarchy tree (paper Fig. 3) |
| `POST` | `/api/train-and-diagnose` | SSE stream: live training + full 3-stage diagnosis |
| `POST` | `/api/predict` | Raw dynamic feature prediction (CSV upload) |
| `POST` | `/api/explain-static` | Raw static feature explanation (CSV upload) |
| `POST` | `/api/explain-model` | Static explanation from `.h5` / `.keras` file |
| `POST` | `/api/analyze-project` | Full pipeline from `.zip` project (requires `DEFAULT_ALLOW_UNTRUSTED_CODE=1`) |

---

## Docker Deployment

```bash
# Standard build (no TensorFlow; Train & Diagnose returns a friendly error)
docker build -t default-tool .
docker run --rm -p 8000:8000 default-tool

# Full build (includes TensorFlow, ~2 GB image)
docker build --build-arg INSTALL_TENSORFLOW=1 -t default-tool:full .
docker run --rm -p 8000:8000 default-tool:full
```

The Dockerfile uses a 2-stage build: Node.js builds the frontend, then Python serves everything.

---

## Render Deployment

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/SigmaJahan/DEFault-tool)

`render.yaml` is included for automatic Blueprint deployment. Click the button above or:

1. Go to [Render Dashboard](https://dashboard.render.com/select-repo) → **New Blueprint**
2. Connect your GitHub account and select **SigmaJahan/DEFault-tool**
3. Render will read `render.yaml` and configure the service automatically
4. Click **Apply** — the app will be live at a `*.onrender.com` URL within ~2 minutes

The frontend is pre-built in `webapp/static/`, so no Node.js build step is needed.

> **Note**: Free-tier Render (512 MB RAM) covers static analysis. The "Train & Diagnose" feature requires TensorFlow (~1 GB RAM); upgrade to Starter plan and add `pip install tensorflow` to the build command.

---

## Project Structure

```
DEFault-tool/
├── artifacts/default_models/   # 9 pre-trained ML models (.joblib)
├── default_tool/               # Core inference package
├── webapp/
│   ├── main.py                 # FastAPI app (all endpoints)
│   ├── schemas.py              # Pydantic request/response models
│   ├── code_sandbox.py         # Safe Keras code execution
│   ├── inference_utils.py      # Stage 1/2/3 builders, SHAP, fault taxonomy
│   ├── training_runner.py      # SSE streaming training subprocess
│   ├── project_runner.py       # ZIP project execution (advanced)
│   └── frontend/               # Next.js 16 + TypeScript web app
├── webapp/static/              # Pre-built frontend (committed for deployment)
├── Dockerfile                  # Multi-stage Docker build
├── render.yaml                 # Render deployment config
└── requirements.txt            # Python dependencies
```

---

## Security Notes

- Model code runs in an isolated subprocess with no network access.
- Project execution (`/api/analyze-project`) is **disabled by default**. Enable only in sandboxed environments:
  ```bash
  DEFAULT_ALLOW_UNTRUSTED_CODE=1 uvicorn webapp.main:app ...
  ```
