"""
Sandboxed execution of user-pasted Keras model code.

Mirrors the subprocess isolation pattern from project_runner.py:
- multiprocessing.Process for full isolation
- multiprocessing.Queue for result passing
- Explicit timeout via process.join(timeout)

The user pastes raw Python code. We:
1. AST-parse it to check for a build_model() function
2. If not found, auto-wrap the model code in a build_model() function
3. Write to a temp file, execute in subprocess
4. Subprocess extracts static features and returns them as JSON via stdout
"""
from __future__ import annotations

import ast
import multiprocessing as mp
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

# ── Subprocess script template ────────────────────────────────────────────────
# This script runs inside the isolated subprocess.
# It imports the repo's own static_features module and extracts features from
# whatever Keras model build_model() returns.
_SUBPROCESS_SCRIPT = '''\
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU

sys.path.insert(0, {tool_root!r})

import json
import traceback

try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
except ImportError:
    print(json.dumps({{"error": "TensorFlow is not installed in this environment."}}))
    sys.exit(1)

# ── User code ─────────────────────────────────────────────────────────────────
try:
{indented_user_code}
except Exception as e:
    print(json.dumps({{"error": f"Error in user code: {{e}}", "traceback": traceback.format_exc()}}))
    sys.exit(1)
# ── End user code ─────────────────────────────────────────────────────────────

try:
    import inspect
    # find build_model in the local scope
    _local_vars = {{k: v for k, v in locals().items()}}
    if "build_model" not in _local_vars or not callable(_local_vars["build_model"]):
        print(json.dumps({{"error": "No build_model() function found. Define a function named build_model() that returns a compiled tf.keras.Model."}}))
        sys.exit(1)

    model = _local_vars["build_model"]()

    from default_tool.static_features import extract_static_features_from_model
    df = extract_static_features_from_model(model, model_name={model_name!r})
    features = df.iloc[0].to_dict()
    # Remove non-numeric / metadata columns
    features.pop("Model_File", None)
    print(json.dumps(features))
except Exception as e:
    print(json.dumps({{"error": f"Feature extraction failed: {{e}}", "traceback": traceback.format_exc()}}))
    sys.exit(1)
'''


def _has_build_model(code: str) -> bool:
    """Return True if code defines a top-level build_model() function."""
    try:
        tree = ast.parse(code)
        return any(
            isinstance(n, ast.FunctionDef) and n.name == "build_model"
            for n in ast.walk(tree)
        )
    except SyntaxError:
        return False


def _auto_wrap_code(code: str) -> tuple[str, str]:
    """
    Wrap the user's model code in a build_model() function if not already present.
    Returns (wrapped_code, warning_message).
    """
    # Indent every line by 4 spaces
    indented = textwrap.indent(code.strip(), "    ")
    wrapped = (
        "def build_model():\n"
        f"{indented}\n"
        "    return model\n"
    )
    warning = (
        "No build_model() function detected — code was auto-wrapped. "
        "For best results, define: def build_model(): ... return model"
    )
    return wrapped, warning


def extract_features_from_code(
    code: str,
    model_name: str = "pasted_model",
    timeout_seconds: int = 45,
) -> tuple[dict[str, Any], list[str]]:
    """
    Execute pasted Keras model code in a sandboxed subprocess and extract
    31 static features from the resulting model.

    Returns:
        (features_dict, warnings_list)

    Raises:
        RuntimeError: if the subprocess fails or times out.
    """
    warnings: list[str] = []

    # Validate Python syntax first (fast, no subprocess)
    try:
        ast.parse(code)
    except SyntaxError as e:
        raise RuntimeError(f"Syntax error in your code: {e}") from e

    # Auto-wrap if no build_model() present
    if not _has_build_model(code):
        code, warn = _auto_wrap_code(code)
        warnings.append(warn)

    # Build the subprocess script
    tool_root = str(Path(__file__).resolve().parents[1])
    indented_user_code = textwrap.indent(code.strip(), "    ")
    script = _SUBPROCESS_SCRIPT.format(
        tool_root=tool_root,
        indented_user_code=indented_user_code,
        model_name=model_name,
    )

    with tempfile.TemporaryDirectory(prefix="default_sandbox_") as tmpdir:
        script_path = Path(tmpdir) / "sandbox_model.py"
        script_path.write_text(script, encoding="utf-8")

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=str(Path(__file__).resolve().parents[1]),
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Model execution timed out after {timeout_seconds}s. "
                "Ensure your model code compiles quickly (no training inside build_model)."
            )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if not stdout:
            detail = stderr[-2000:] if stderr else "No output from subprocess."
            raise RuntimeError(f"Could not build model: {detail}")

        import json
        try:
            payload = json.loads(stdout.split("\n")[-1])  # take last line (suppress TF output)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Unexpected subprocess output: {stdout[:500]}"
            ) from e

        if "error" in payload:
            tb = payload.get("traceback", "")
            raise RuntimeError(f"{payload['error']}\n{tb}".strip())

        # Collect any TF warnings from stderr, skipping internal file-path lines
        if stderr:
            for line in stderr.splitlines():
                stripped = line.strip()
                # Skip lines that are just Python source locations or internal paths
                if not stripped:
                    continue
                if "site-packages" in stripped or ".venv" in stripped:
                    continue
                if stripped.startswith("warnings.warn(") or stripped.startswith("warn("):
                    continue
                if any(kw in stripped.lower() for kw in ("userwarning:", "deprecationwarning:", "futurewarning:")):
                    if len(warnings) < 3:
                        warnings.append(stripped)

        # Ensure all values are floats (JSON may give ints for counts)
        features = {k: float(v) for k, v in payload.items() if isinstance(v, (int, float))}
        return features, warnings
