# ── Stage 1: Build Next.js frontend ───────────────────────────────────────
FROM node:20-slim AS frontend-builder

WORKDIR /app/webapp/frontend

COPY webapp/frontend/package.json webapp/frontend/package-lock.json ./
RUN npm ci --ignore-scripts

COPY webapp/frontend ./
# Produce a static export into webapp/static/
ENV NODE_ENV=production
RUN npm run build

# ── Stage 2: Python runtime ────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional TensorFlow for "Train & Diagnose" endpoint.
# Build with: docker build --build-arg INSTALL_TENSORFLOW=1 ...
ARG INSTALL_TENSORFLOW=0
RUN if [ "$INSTALL_TENSORFLOW" = "1" ]; then \
      pip install --no-cache-dir "tensorflow>=2.16,<3"; \
    fi

# Copy application code
COPY default_tool ./default_tool
COPY webapp/main.py \
     webapp/schemas.py \
     webapp/code_sandbox.py \
     webapp/inference_utils.py \
     webapp/training_runner.py \
     webapp/project_runner.py \
     webapp/__init__.py \
     ./webapp/
COPY artifacts ./artifacts
COPY README.md ./README.md

# Copy built frontend from Stage 1
COPY --from=frontend-builder /app/webapp/static ./webapp/static

EXPOSE 8000
ENV DEFAULT_ALLOW_UNTRUSTED_CODE=0

CMD ["uvicorn", "webapp.main:app", "--host", "0.0.0.0", "--port", "8000"]
