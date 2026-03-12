"""
DEFault Tool — Automated Screencast Demo
=========================================
Records a 3–5 minute Playwright video demo suitable for ICST 2026 tool paper.

Scenes:
  1. Landing page overview
  2. Static code analysis (buggy model → SHAP waterfall + fault gauge)
  3. History-paste analysis (overfitting → 3-stage pipeline)
  4. Live Train & Diagnose (SSE epoch stream → full diagnosis)

Usage:
  pip install playwright
  playwright install chromium
  python demo/record_demo.py

Output: demo/default_demo.mp4  (Playwright webm → ffmpeg converts to mp4 if available)
"""

import asyncio
import shutil
import subprocess
from pathlib import Path

from playwright.async_api import async_playwright

# ── Config ────────────────────────────────────────────────────────────────────
URL = "https://sigmajahan-default-tool.hf.space"
OUT_DIR = Path(__file__).parent / "recordings"
OUT_DIR.mkdir(exist_ok=True)

VIEWPORT = {"width": 1440, "height": 860}

# Typing speed: ms between characters (feels like ~60 wpm)
TYPE_DELAY = 18

# Pause helpers (ms)
PAUSE_SHORT  = 1_200
PAUSE_MEDIUM = 2_500
PAUSE_LONG   = 4_000
PAUSE_XLONG  = 6_000

# ── Demo code snippets ────────────────────────────────────────────────────────

# Scene 2: buggy classification model — relu on output (should be softmax)
BUGGY_CODE = """\
import tensorflow as tf
from tensorflow import keras

def build_model():
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='relu')   # BUG: should be softmax
    ])
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
"""

# Scene 4: cleaner model for live training demo
TRAIN_CODE = """\
import tensorflow as tf
from tensorflow import keras

def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(20,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='relu')   # BUG: sigmoid expected
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

async def human_type(page, text: str, delay: int = TYPE_DELAY):
    """Type text into whichever element currently has keyboard focus."""
    await page.keyboard.type(text, delay=delay)


async def slow_scroll(page, px: int = 300, steps: int = 8, delay_ms: int = 80):
    """Smooth scroll to simulate a human scanning results."""
    step = px // steps
    for _ in range(steps):
        await page.mouse.wheel(0, step)
        await asyncio.sleep(delay_ms / 1000)


async def wait_for_results(page, timeout: int = 60_000):
    """Wait until the results panel is no longer in skeleton/loading state."""
    await page.wait_for_selector(
        '[aria-label="SHAP explanations and fault insights"] .animate-fade-in',
        timeout=timeout,
    )


async def set_editor_code(page, code: str):
    """
    Set Monaco editor value via the Monaco JS API (reliable across OSes).
    Then simulate a brief human pause so the recording looks natural.
    """
    # Wait for Monaco to be ready
    await page.wait_for_function("typeof window.monaco !== 'undefined'", timeout=15_000)
    # Set value through Monaco API — bypasses keyboard typing issues with special chars
    await page.evaluate(
        "([code]) => { const ed = window.monaco.editor.getEditors()[0]; ed.setValue(code); ed.setPosition({lineNumber:1,column:1}); }",
        [code],
    )
    # Small pause so viewer sees the code appear
    await asyncio.sleep(1.2)


# ── Main demo ─────────────────────────────────────────────────────────────────

async def run():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=False,          # visible window — looks better on recording
            args=["--start-maximized"],
        )
        context = await browser.new_context(
            viewport=VIEWPORT,
            record_video_dir=str(OUT_DIR),
            record_video_size=VIEWPORT,
        )
        page = await context.new_page()

        # ── SCENE 1: Landing page ─────────────────────────────────────────────
        print("[Scene 1] Loading landing page…")
        await page.goto(URL, wait_until="networkidle")
        await asyncio.sleep(PAUSE_LONG / 1000)

        # Pan slowly over the three columns so viewer sees the layout
        await slow_scroll(page, px=100, steps=5)
        await asyncio.sleep(PAUSE_MEDIUM / 1000)
        await slow_scroll(page, px=-100, steps=5)
        await asyncio.sleep(PAUSE_MEDIUM / 1000)

        # ── SCENE 2: Static code analysis ────────────────────────────────────
        print("[Scene 2] Static analysis — buggy model…")

        # Load the buggy model into the Monaco editor
        await set_editor_code(page, BUGGY_CODE)
        await asyncio.sleep(PAUSE_MEDIUM / 1000)

        # Set a meaningful model name
        model_input = page.locator('[aria-label="Model name"]')
        await model_input.click(click_count=3)
        await page.keyboard.type("mnist_classifier", delay=25)
        await asyncio.sleep(PAUSE_SHORT / 1000)

        # Click "Check Model"
        check_btn = page.get_by_role("button", name="Check model: instant static analysis, no training required")
        await check_btn.click()
        print("  → Waiting for static results…")

        # Wait for SHAP waterfall heading to appear (chart is div-based, no canvas/svg)
        await page.get_by_text("SHAP Waterfall: Root Cause").wait_for(timeout=60_000)
        await asyncio.sleep(PAUSE_MEDIUM / 1000)

        # Scroll through the right panel to show SHAP waterfall + insights
        results_panel = page.locator('[aria-label="SHAP explanations and fault insights"]')
        await results_panel.hover()
        await slow_scroll(page, px=400, steps=12)
        await asyncio.sleep(PAUSE_LONG / 1000)
        await slow_scroll(page, px=-400, steps=12)
        await asyncio.sleep(PAUSE_MEDIUM / 1000)

        # Scroll centre panel to show pipeline + taxonomy
        pipeline_panel = page.locator('[aria-label="Analysis pipeline and training charts"]')
        await pipeline_panel.hover()
        await slow_scroll(page, px=500, steps=15)
        await asyncio.sleep(PAUSE_LONG / 1000)
        await slow_scroll(page, px=-500, steps=15)
        await asyncio.sleep(PAUSE_MEDIUM / 1000)

        # Reset before next scene
        reset_btn = page.get_by_role("button", name="Reset: clear all results and start over")
        if await reset_btn.is_visible():
            await reset_btn.click()
        await asyncio.sleep(PAUSE_SHORT / 1000)

        # ── SCENE 3: Training history paste ──────────────────────────────────
        # NOTE: the history-paste flow uses the /api/analyze-history endpoint
        # which the frontend calls when the user fills the history form fields.
        # If there is no dedicated "History" tab in the UI the cleanest demo
        # path is to skip to Scene 4 and do the full live training instead.
        # This block is kept as a placeholder — adapt if the UI adds a history tab.
        print("[Scene 3] Skipping to live training (no separate history-paste tab in UI)…")
        await asyncio.sleep(PAUSE_SHORT / 1000)

        # ── SCENE 4: Live Train & Diagnose ────────────────────────────────────
        print("[Scene 4] Live training + full diagnosis…")

        await set_editor_code(page, TRAIN_CODE)
        await asyncio.sleep(PAUSE_MEDIUM / 1000)

        # Update model name
        await model_input.click(click_count=3)
        await page.keyboard.type("binary_net", delay=25)
        await asyncio.sleep(PAUSE_SHORT / 1000)

        # Make sure "Dummy Data" is selected (it's the default)
        dummy_btn = page.get_by_role("button", name="Dummy Data")
        if await dummy_btn.is_visible():
            await dummy_btn.click()
        await asyncio.sleep(PAUSE_SHORT / 1000)

        # Click "Train & Diagnose"
        train_btn = page.get_by_role(
            "button",
            name="Train and diagnose: real model training with full 3-stage fault diagnosis",
        )
        await train_btn.click()
        print("  → Training started — watching epoch stream…")

        # While training, pan to the centre pipeline panel so epochs are visible
        await pipeline_panel.hover()
        await asyncio.sleep(PAUSE_MEDIUM / 1000)

        # Wait for training + full diagnosis to complete (up to 120 s on HF free tier)
        # The SHAP heading appears only once all 3 stages finish
        await page.get_by_text("SHAP Waterfall: Root Cause").wait_for(timeout=120_000)
        print("  → Training complete — showing results…")
        await asyncio.sleep(PAUSE_LONG / 1000)

        # Scroll centre panel: pipeline → live chart → detection gauge → categories → taxonomy
        await pipeline_panel.hover()
        await slow_scroll(page, px=800, steps=25)
        await asyncio.sleep(PAUSE_XLONG / 1000)

        # Scroll right panel: SHAP waterfall → insights
        await results_panel.hover()
        await slow_scroll(page, px=600, steps=20)
        await asyncio.sleep(PAUSE_XLONG / 1000)

        # Scroll back to top for clean ending frame
        await results_panel.hover()
        await slow_scroll(page, px=-600, steps=20)
        await asyncio.sleep(PAUSE_MEDIUM / 1000)

        print("[Done] Closing browser and saving video…")
        await context.close()
        await browser.close()

    # ── Convert webm → mp4 if ffmpeg is available ─────────────────────────────
    webm_files = sorted(OUT_DIR.glob("*.webm"), key=lambda p: p.stat().st_mtime, reverse=True)
    if webm_files:
        webm = webm_files[0]
        mp4 = Path(__file__).parent / "default_demo.mp4"
        if shutil.which("ffmpeg"):
            print(f"Converting {webm.name} → default_demo.mp4 …")
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(webm), "-c:v", "libx264", "-crf", "18",
                 "-preset", "slow", "-pix_fmt", "yuv420p", str(mp4)],
                check=True,
            )
            print(f"Saved: {mp4}")
        else:
            final = Path(__file__).parent / "default_demo.webm"
            webm.rename(final)
            print(f"ffmpeg not found — raw recording saved as: {final}")
            print("Install ffmpeg to convert: brew install ffmpeg")
    else:
        print("Warning: no recording found in", OUT_DIR)


if __name__ == "__main__":
    asyncio.run(run())
