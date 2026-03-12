"""
DEFault Tool — Automated Screencast Demo
=========================================
Records a 3–5 minute smooth demo suitable for ICST 2026 tool paper.

Recording strategy:
  - ffmpeg captures the macOS screen directly (avfoundation, 30 fps, crisp quality)
  - Playwright drives a VISIBLE Chrome window — no Playwright webm recording
  - Each step has generous pauses so nothing is skipped or rushed

Scenes:
  1. Landing page overview — shows the 3-column layout
  2. Static code analysis  — paste buggy model → Check Model → SHAP waterfall
  3. Live Train & Diagnose — load second model → Train & Diagnose → live epochs → full results

Usage:
  source .venv/bin/activate
  pip install playwright
  playwright install chromium
  python demo/record_demo.py

Output: demo/default_demo.mp4
"""

import asyncio
import subprocess
import time
from pathlib import Path

from playwright.async_api import async_playwright

# ── Config ────────────────────────────────────────────────────────────────────
URL        = "https://sigmajahan-default-tool.hf.space"
OUT_MP4    = Path(__file__).parent / "default_demo.mp4"
VIEWPORT   = {"width": 1440, "height": 860}

# ffmpeg screen-capture device index (macOS avfoundation)
# Run:  ffmpeg -f avfoundation -list_devices true -i ""   to confirm index
SCREEN_IDX = "1"

# ── Pause constants (seconds) ─────────────────────────────────────────────────
P_SHORT  = 2.0   # after a small action (click, type)
P_MEDIUM = 3.5   # after a section transition
P_LONG   = 5.5   # let viewer read a result panel
P_XLONG  = 8.0   # let viewer study a complex screen

# ── Demo code snippets ────────────────────────────────────────────────────────

# Scene 2: buggy 10-class classifier — relu on output (correct: softmax)
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

# Scene 3: binary classifier — relu on output (correct: sigmoid)
TRAIN_CODE = """\
import tensorflow as tf
from tensorflow import keras

def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(20,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='relu')   # BUG: should be sigmoid
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

async def pause(seconds: float):
    """Explicit pause — shows on screen, gives viewer time to read."""
    await asyncio.sleep(seconds)


async def smooth_scroll_element(page, selector: str, px: int, steps: int = 20):
    """
    Scroll a specific element smoothly by moving the mouse over it first,
    then using small mouse-wheel ticks. Much smoother than page.mouse.wheel().
    """
    el = page.locator(selector).first
    box = await el.bounding_box()
    if not box:
        return
    cx = box["x"] + box["width"] / 2
    cy = box["y"] + box["height"] / 2
    await page.mouse.move(cx, cy)
    await asyncio.sleep(0.2)
    step_px = px // steps
    for _ in range(steps):
        await page.mouse.wheel(0, step_px)
        await asyncio.sleep(0.06)   # ~17 fps scroll — very smooth


async def set_editor_code(page, code: str):
    """Inject code into Monaco editor via JS API — instant, reliable."""
    await page.wait_for_function("typeof window.monaco !== 'undefined'", timeout=15_000)
    await page.evaluate(
        "([c]) => { const ed = window.monaco.editor.getEditors()[0]; "
        "ed.setValue(c); ed.setPosition({lineNumber:1,column:1}); "
        "ed.revealLine(1); }",
        [code],
    )
    await pause(P_SHORT)   # pause after code appears so viewer can read it


async def hover_button(page, locator):
    """Move mouse to button centre and pause — makes the click visible."""
    box = await locator.bounding_box()
    if box:
        await page.mouse.move(
            box["x"] + box["width"] / 2,
            box["y"] + box["height"] / 2,
        )
        await pause(1.0)


# ── ffmpeg screen capture ─────────────────────────────────────────────────────

def start_recording(out_path: Path) -> subprocess.Popen:
    """Start ffmpeg capturing the macOS screen."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "avfoundation",
        "-capture_cursor", "1",
        "-capture_mouse_clicks", "1",
        "-framerate", "30",
        "-i", f"{SCREEN_IDX}:none",      # screen only, no audio
        "-vf", "scale=1440:860",
        "-c:v", "libx264",
        "-crf", "18",                    # high quality
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    print(f"  Starting screen capture → {out_path.name}")
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_recording(proc: subprocess.Popen):
    """Send 'q' to ffmpeg stdin to gracefully stop recording."""
    print("  Stopping screen capture…")
    proc.stdin.write(b"q")
    proc.stdin.flush()
    proc.wait(timeout=15)


# ── Main demo ─────────────────────────────────────────────────────────────────

async def run():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=False,
            args=[
                f"--window-size={VIEWPORT['width']},{VIEWPORT['height']}",
                "--window-position=0,0",
                "--disable-infobars",
                "--no-first-run",
            ],
        )
        context = await browser.new_context(
            viewport=VIEWPORT,
            # NO record_video_dir — we use ffmpeg instead
        )
        page = await context.new_page()

        # ── SCENE 1: Load the page ────────────────────────────────────────────
        print("\n[Scene 1] Loading DEFault…")
        await page.goto(URL, wait_until="networkidle", timeout=40_000)
        await pause(P_SHORT)   # brief settle

        # Give ffmpeg 1 second to warm up before we start moving
        await pause(1.0)

        # Start screen recording AFTER the page is loaded and visible
        rec_proc = start_recording(OUT_MP4)
        await pause(2.0)   # let ffmpeg buffer a clean first frame

        # Show the empty tool — let viewer see the 3-column layout
        await pause(P_LONG)

        # Move mouse slowly over each panel heading to introduce them
        for selector in [
            '[aria-label="Model code editor and training configuration"]',
            '[aria-label="Analysis pipeline and training charts"]',
            '[aria-label="SHAP explanations and fault insights"]',
        ]:
            el = page.locator(selector).first
            box = await el.bounding_box()
            if box:
                await page.mouse.move(box["x"] + 20, box["y"] + 20)
                await pause(P_SHORT)

        # ── SCENE 2: Static code analysis ────────────────────────────────────
        print("[Scene 2] Pasting buggy model…")

        # Click the editor area to focus it, then inject code
        editor_area = page.locator('[aria-label="Model code editor and training configuration"]').first
        await editor_area.click()
        await pause(0.5)
        await set_editor_code(page, BUGGY_CODE)
        await pause(P_MEDIUM)   # viewer reads the code

        # Type model name slowly so viewer sees it
        print("  Setting model name…")
        model_input = page.locator('[aria-label="Model name"]')
        await model_input.click(click_count=3)
        await pause(0.3)
        await page.keyboard.type("mnist_classifier", delay=60)
        await pause(P_SHORT)

        # Hover Check Model button, then click
        print("  Clicking Check Model…")
        check_btn = page.get_by_role(
            "button",
            name="Check model: instant static analysis, no training required",
        )
        await hover_button(page, check_btn)
        await check_btn.click()

        # Show the loading state briefly
        await pause(1.5)

        # Wait for SHAP waterfall to appear
        print("  Waiting for results…")
        await page.get_by_text("SHAP Waterfall: Root Cause").wait_for(timeout=60_000)
        await pause(P_LONG)   # viewer reads detection result

        # Scroll right panel: fault probability → SHAP bars → insights
        print("  Scrolling results panel…")
        await smooth_scroll_element(
            page,
            '[aria-label="SHAP explanations and fault insights"]',
            px=450, steps=25,
        )
        await pause(P_XLONG)   # viewer reads SHAP waterfall

        await smooth_scroll_element(
            page,
            '[aria-label="SHAP explanations and fault insights"]',
            px=-450, steps=25,
        )
        await pause(P_MEDIUM)

        # Scroll centre panel: pipeline stages → taxonomy tree
        print("  Scrolling pipeline panel…")
        await smooth_scroll_element(
            page,
            '[aria-label="Analysis pipeline and training charts"]',
            px=600, steps=30,
        )
        await pause(P_XLONG)   # viewer reads pipeline + taxonomy

        await smooth_scroll_element(
            page,
            '[aria-label="Analysis pipeline and training charts"]',
            px=-600, steps=30,
        )
        await pause(P_MEDIUM)

        # Reset before Scene 3
        reset_btn = page.get_by_role("button", name="Reset: clear all results and start over")
        if await reset_btn.is_visible():
            await hover_button(page, reset_btn)
            await reset_btn.click()
        await pause(P_MEDIUM)

        # ── SCENE 3: Live Train & Diagnose ────────────────────────────────────
        print("[Scene 3] Live training demo…")

        # Load the binary model
        await editor_area.click()
        await pause(0.5)
        await set_editor_code(page, TRAIN_CODE)
        await pause(P_MEDIUM)   # viewer reads new code

        # Update model name
        await model_input.click(click_count=3)
        await pause(0.3)
        await page.keyboard.type("binary_net", delay=60)
        await pause(P_SHORT)

        # Confirm Dummy Data is selected (show the choice explicitly)
        dummy_btn = page.get_by_role("button", name="Dummy Data")
        if await dummy_btn.is_visible():
            await hover_button(page, dummy_btn)
            await dummy_btn.click()
        await pause(P_SHORT)

        # Hover Train & Diagnose, then click
        print("  Clicking Train & Diagnose…")
        train_btn = page.get_by_role(
            "button",
            name="Train and diagnose: real model training with full 3-stage fault diagnosis",
        )
        await hover_button(page, train_btn)
        await train_btn.click()
        await pause(1.5)

        # Pan to the centre panel immediately so epochs stream is visible
        pipeline_panel = '[aria-label="Analysis pipeline and training charts"]'
        el = page.locator(pipeline_panel).first
        box = await el.bounding_box()
        if box:
            await page.mouse.move(box["x"] + box["width"] / 2, box["y"] + 30)
        await pause(P_LONG)     # watch first few epochs appear

        # Keep watching the live chart for a few more seconds
        await pause(P_XLONG)

        # Wait for full diagnosis to finish
        print("  Waiting for training + diagnosis to complete…")
        await page.get_by_text("SHAP Waterfall: Root Cause").wait_for(timeout=120_000)
        await pause(P_LONG)

        # Scroll centre panel to show live chart + detection gauge + category bars
        print("  Scrolling pipeline results…")
        await smooth_scroll_element(page, pipeline_panel, px=900, steps=40)
        await pause(P_XLONG)

        await smooth_scroll_element(page, pipeline_panel, px=-900, steps=40)
        await pause(P_MEDIUM)

        # Scroll right panel: SHAP waterfall → insights
        print("  Scrolling SHAP + insights…")
        results_panel = '[aria-label="SHAP explanations and fault insights"]'
        await smooth_scroll_element(page, results_panel, px=600, steps=30)
        await pause(P_XLONG)

        # Scroll back to top — clean ending frame
        await smooth_scroll_element(page, results_panel, px=-600, steps=30)
        await pause(P_LONG)

        print("[Done] Closing browser…")
        await context.close()
        await browser.close()

    # Stop ffmpeg after browser closes
    stop_recording(rec_proc)
    print(f"\nVideo saved: {OUT_MP4}")


if __name__ == "__main__":
    asyncio.run(run())
