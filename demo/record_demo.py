"""
DEFault Tool — Automated Screencast Demo
=========================================
Records a ~4-minute professional demo for the ICST 2026 tool paper.

Recording strategy:
  - ffmpeg captures the macOS screen at 30fps (crisp, no Playwright webm lag)
  - Playwright drives a VISIBLE Chrome window
  - Real dataset: Breast Cancer Wisconsin (569 samples, 20 features, binary)
  - Generous pauses on every result screen so the audience can read everything

Scenes (timed to ~4 min total):
  1. Landing page    — 3-column layout overview           (~20 s)
  2. Static analysis — buggy classifier → SHAP waterfall  (~80 s)
  3. Train & Diagnose— real dataset upload → live epochs
                       → full 3-stage results              (~130 s)

Usage:
  source .venv/bin/activate
  python demo/record_demo.py

Output: demo/default_demo.mp4
"""

import asyncio
import subprocess
from pathlib import Path

from playwright.async_api import async_playwright

# ── Config ────────────────────────────────────────────────────────────────────
URL         = "https://sigmajahan-default-tool.hf.space"
DEMO_DIR    = Path(__file__).parent
OUT_MP4     = DEMO_DIR / "default_demo.mp4"
DATASET_CSV = DEMO_DIR / "breast_cancer_20f.csv"
VIEWPORT    = {"width": 1440, "height": 860}

# ffmpeg screen index — run:
#   ffmpeg -f avfoundation -list_devices true -i ""
# to confirm. "1" = "Capture screen 0" on most Macs.
SCREEN_IDX  = "1"

# ── Timing (seconds) — generous so audience can read every screen ─────────────
P_TINY   = 0.8    # micro pause between sub-actions
P_SHORT  = 2.5    # after small action (click, type short text)
P_READ   = 5.0    # let audience read a panel or result
P_STUDY  = 8.0    # let audience study a complex visualisation
P_EPOCH  = 12.0   # extra watch time on live epoch chart during training

# ── Demo code snippets ────────────────────────────────────────────────────────

# Scene 2: buggy 10-class classifier — relu on output (should be softmax)
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

# Scene 3: binary classifier trained on Breast Cancer dataset (20 features)
# relu on output is the bug — should be sigmoid for binary classification
TRAIN_CODE = """\
import tensorflow as tf
from tensorflow import keras

def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(20,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
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

async def p(sec: float):
    """Pause and let the recording breathe."""
    await asyncio.sleep(sec)


async def move_to(page, locator):
    """Move mouse to centre of a locator."""
    box = await locator.bounding_box()
    if box:
        cx = box["x"] + box["width"] / 2
        cy = box["y"] + box["height"] / 2
        await page.mouse.move(cx, cy)


async def hover_then_click(page, locator, pre_hover_pause=1.2, post_click_pause=None):
    """Visibly hover over element, pause, then click — makes clicks obvious on screen."""
    await move_to(page, locator)
    await p(pre_hover_pause)
    await locator.click()
    if post_click_pause is not None:
        await p(post_click_pause)


async def scroll_panel(page, selector: str, px: int, steps: int = 30, tick_ms: int = 55):
    """
    Smoothly scroll a named panel by hovering over it and using fine wheel ticks.
    Much smoother than page.mouse.wheel() on the whole page.
    """
    el = page.locator(selector).first
    box = await el.bounding_box()
    if not box:
        return
    await page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    await asyncio.sleep(0.15)
    step_px = px // steps
    for _ in range(steps):
        await page.mouse.wheel(0, step_px)
        await asyncio.sleep(tick_ms / 1000)


async def set_editor_code(page, code: str):
    """Set Monaco editor content via the JS API — instant and reliable."""
    await page.wait_for_function("typeof window.monaco !== 'undefined'", timeout=15_000)
    await page.evaluate(
        "([c]) => { const ed = window.monaco.editor.getEditors()[0]; "
        "ed.setValue(c); ed.setPosition({lineNumber:1,column:1}); ed.revealLine(1); }",
        [code],
    )
    await p(P_SHORT)  # pause so code is visible before moving on


async def type_into(page, locator, text: str, char_delay_ms: int = 65):
    """Click a field and type visibly, character by character."""
    await locator.click(click_count=3)   # select-all first
    await p(P_TINY)
    await page.keyboard.type(text, delay=char_delay_ms)
    await p(P_SHORT)


# ── Screen capture ────────────────────────────────────────────────────────────

def start_capture(out_path: Path) -> subprocess.Popen:
    cmd = [
        "ffmpeg", "-y",
        "-f", "avfoundation",
        "-capture_cursor", "1",
        "-capture_mouse_clicks", "1",
        "-framerate", "30",
        "-i", f"{SCREEN_IDX}:none",
        "-vf", f"scale={VIEWPORT['width']}:{VIEWPORT['height']}",
        "-c:v", "libx264",
        "-crf", "17",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    print(f"  ▶ Recording → {out_path.name}")
    return subprocess.Popen(
        cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def stop_capture(proc: subprocess.Popen):
    print("  ■ Stopping capture…")
    try:
        proc.stdin.write(b"q")
        proc.stdin.flush()
        proc.wait(timeout=15)
    except Exception:
        proc.terminate()


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
                "--noerrdialogs",
            ],
        )
        ctx  = await browser.new_context(viewport=VIEWPORT)
        page = await ctx.new_page()

        # ── Load page first, then start recording so first frame is clean ────
        print("\n[Scene 1] Loading DEFault tool…")
        await page.goto(URL, wait_until="networkidle", timeout=45_000)
        await p(2.0)   # settle

        rec = start_capture(OUT_MP4)
        await p(2.5)   # let ffmpeg buffer a clean opening frame

        # ── SCENE 1: Tool overview (~20 s) ───────────────────────────────────
        print("  Showing 3-column layout…")
        await p(P_READ)   # audience sees the empty tool

        # Pan mouse slowly across each panel to introduce them
        PANELS = [
            '[aria-label="Model code editor and training configuration"]',
            '[aria-label="Analysis pipeline and training charts"]',
            '[aria-label="SHAP explanations and fault insights"]',
        ]
        for sel in PANELS:
            el = page.locator(sel).first
            box = await el.bounding_box()
            if box:
                await page.mouse.move(box["x"] + 30, box["y"] + 30)
                await p(P_SHORT)

        # ── SCENE 2: Static code analysis (~80 s) ────────────────────────────
        print("\n[Scene 2] Static code analysis…")

        # Click editor region to focus, then load code
        await page.locator(PANELS[0]).first.click()
        await p(P_TINY)
        await set_editor_code(page, BUGGY_CODE)
        await p(P_READ)   # audience reads the buggy code

        # Set model name visibly
        print("  Setting model name…")
        await type_into(page, page.locator('[aria-label="Model name"]'), "mnist_classifier")

        # Hover and click Check Model
        print("  Clicking Check Model…")
        check_btn = page.get_by_role(
            "button",
            name="Check model: instant static analysis, no training required",
        )
        await hover_then_click(page, check_btn, pre_hover_pause=1.5, post_click_pause=2.0)

        # Wait for SHAP waterfall to appear
        print("  Waiting for analysis results…")
        await page.get_by_text("SHAP Waterfall: Root Cause").wait_for(timeout=60_000)
        await p(P_READ)   # audience reads the detection probability

        # ── Right panel: fault probability + SHAP waterfall + insights ───────
        print("  Showing SHAP waterfall…")
        await scroll_panel(page, PANELS[2], px=0)   # hover to focus
        await p(P_STUDY)  # SHAP waterfall — let audience study it

        await scroll_panel(page, PANELS[2], px=350, steps=20)
        await p(P_STUDY)  # insights panel — let audience read recommendations

        await scroll_panel(page, PANELS[2], px=-350, steps=20)
        await p(P_SHORT)

        # ── Centre panel: pipeline stages + taxonomy tree ─────────────────────
        print("  Showing pipeline visualisation…")
        await scroll_panel(page, PANELS[1], px=0)   # hover to focus
        await p(P_STUDY)  # detection gauge + pipeline diagram

        await scroll_panel(page, PANELS[1], px=500, steps=25)
        await p(P_STUDY)  # fault taxonomy tree highlighted

        await scroll_panel(page, PANELS[1], px=-500, steps=25)
        await p(P_SHORT)

        # Reset before Scene 3
        reset_btn = page.get_by_role("button", name="Reset: clear all results and start over")
        if await reset_btn.is_visible():
            await hover_then_click(page, reset_btn, post_click_pause=P_SHORT)

        # ── SCENE 3: Live Train & Diagnose with real dataset (~130 s) ─────────
        print("\n[Scene 3] Live training on real dataset…")

        # Load training code
        await page.locator(PANELS[0]).first.click()
        await p(P_TINY)
        await set_editor_code(page, TRAIN_CODE)
        await p(P_READ)   # audience reads the binary classifier code

        # Update model name
        print("  Setting model name…")
        await type_into(page, page.locator('[aria-label="Model name"]'), "breast_cancer_net")

        # Switch to Upload Dataset mode
        print("  Switching to Upload Dataset…")
        upload_mode_btn = page.get_by_role("button", name="Upload Dataset")
        await hover_then_click(page, upload_mode_btn, post_click_pause=P_SHORT)

        # Upload the real CSV via the hidden file input
        print("  Uploading breast_cancer_20f.csv…")
        file_input = page.locator('input[type="file"][accept=".csv,.npy,.npz"]')
        await file_input.set_input_files(str(DATASET_CSV))
        await p(P_READ)   # show the uploaded filename

        # Hover and click Train & Diagnose
        print("  Clicking Train & Diagnose…")
        train_btn = page.get_by_role(
            "button",
            name="Train and diagnose: real model training with full 3-stage fault diagnosis",
        )
        await hover_then_click(page, train_btn, pre_hover_pause=1.5, post_click_pause=2.0)

        # Move focus to the centre panel to watch epochs stream
        el = page.locator(PANELS[1]).first
        box = await el.bounding_box()
        if box:
            await page.mouse.move(box["x"] + box["width"] / 2, box["y"] + 40)
        await p(P_EPOCH)   # watch the live epoch chart fill in
        await p(P_EPOCH)   # keep watching — training takes ~40-60 s on HF

        # Wait for full diagnosis to complete
        print("  Waiting for training + full diagnosis…")
        await page.get_by_text("SHAP Waterfall: Root Cause").wait_for(timeout=150_000)
        await p(P_READ)    # audience sees training complete

        # ── Centre panel: live chart → detection gauge → category bars ────────
        print("  Scrolling pipeline results…")
        await scroll_panel(page, PANELS[1], px=0)
        await p(P_STUDY)   # live training chart

        await scroll_panel(page, PANELS[1], px=350, steps=20)
        await p(P_STUDY)   # Stage 1 detection gauge

        await scroll_panel(page, PANELS[1], px=350, steps=20)
        await p(P_STUDY)   # Stage 2 category chart

        await scroll_panel(page, PANELS[1], px=350, steps=20)
        await p(P_STUDY)   # taxonomy tree with flagged nodes

        await scroll_panel(page, PANELS[1], px=-1050, steps=40)
        await p(P_SHORT)   # scroll back to top

        # ── Right panel: SHAP waterfall → insights ────────────────────────────
        print("  Scrolling SHAP + insights…")
        await scroll_panel(page, PANELS[2], px=0)
        await p(P_STUDY)   # SHAP waterfall for training run

        await scroll_panel(page, PANELS[2], px=400, steps=25)
        await p(P_STUDY)   # insights and recommendations

        await scroll_panel(page, PANELS[2], px=-400, steps=25)
        await p(P_READ)    # clean final frame showing tool header

        print("\n[Done] Closing browser…")
        await ctx.close()
        await browser.close()

    stop_capture(rec)
    print(f"\n✓ Saved: {OUT_MP4}")
    print(f"  Duration: ~4 minutes | Upload to YouTube (unlisted) for ICST paper")


if __name__ == "__main__":
    asyncio.run(run())
