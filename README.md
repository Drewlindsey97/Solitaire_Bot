# Solitaire Stash Automation Bot

This bot integrates a computer vision board reader, Klondike move planning, and an automation bridge to play "Solitaire Stash" directly on an Android device or emulator. The current model uses 7 tableau columns, 4 foundation piles, a draw-3 stock, and a waste pile. There are no free cells; only Kings can move to empty columns.

## Features
- **Computer Vision Card Reader**: Uses OpenCV template matching to read cards from screenshots.
- **Move Planning**: Supports best-first search, Monte Carlo rollouts, and a fast race policy. Hidden cards stay unknown until a fresh screenshot reveals them; planned paths are not a guarantee of a full solve.
- **Gesture Timing**: Taps and swipes incorporate coordinate jitter and randomized swipe durations. Inter-gesture pauses default to zero; `--human` enables randomized pacing.
- **Multiple Execution Backends**: Supports PC-to-Android ADB, rooted on-device execution (Pydroid 3 / Termux), wireless local debugging (LADB), and Tasker/AutoInput intent relays.
- **Simulation Mode**: Includes a dry-run feature (`--sim`) to test the pipeline on static mock images without requiring a connected device.

---

## File Structure
- [solitaire_auto_bot.py](solitaire_auto_bot.py): Main bot loop and gesture mapping.
- [bridge.py](bridge.py): Device commands, screenshot capture, and gesture timing.
- [board_reader_lib.py](board_reader_lib.py): CV board parser and fixed screen coordinates.
- [solver_state.py](solver_state.py): Shared conversion and validation of card reads.
- [freecell_solver.py](freecell_solver.py): Current Klondike search engine (historical filename).
- [race_policy.py](race_policy.py) and [monte_carlo_solver.py](monte_carlo_solver.py): Alternative move policies.

The `pipeline/`, `*_WORKING.py`, and `archive/` files include older experiments. Start with the main script above. The reader uses fixed coordinates and image templates; a different screen layout needs calibration.

---

## Installation & Setup

1. **Python Dependencies**:
   Install OpenCV, NumPy, Pillow, and Requests:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements-dev.txt
   ```

2. **Android Setup**:
   Ensure your Android device has **USB Debugging** enabled and is connected via ADB.

3. **Running Modes**:
   `bridge.py` selects `PC_ADB` on desktop and `LOCAL_ROOT` or `LOCAL_LADB` on Android. The detection block sets `RUN_MODE` during import, overriding the initial value at the top of the file. `LOCAL_LADB` expects an existing ADB connection at `localhost:5555`.

---

## Running the Bot

### 1. Dry-Run / Simulation Mode (Highly Recommended first step)
You can test the entire pipeline on a pre-captured game screenshot (e.g. from the `Gameplay` folder) without connecting any devices:
```bash
python3 solitaire_auto_bot.py --sim Gameplay/frame_0100.png
```
This will:
- Read the cards from the image file.
- Print out the detected board layout.
- Solve the board and calculate a move sequence.
- Print the exact pixel coordinates it *would* swipe on the device.

### 2. Live Bot Execution
To run the bot live on a connected device:
```bash
python3 solitaire_auto_bot.py
```
This will loop continuously: capture screen -> analyze state -> compute moves -> execute gesture -> wait for UI update.

For initial device verification, use `--moves-per-cycle 1` to read the board after each move. The default batches up to 5 moves using a predicted layout between screenshots.

Race mode is now the default for the timed game. Explicit `search` and
`monte-carlo` runs use a 0.5-second decision budget per cycle; use
`--solver-time-limit 2` (or another positive number of seconds) for a longer
diagnostic comparison. Search retries share that budget rather than each
starting a new one.

`--fast` lowers the capture interval to 0.3 seconds and keeps the established
720-860 ms drag duration. It no longer silently shortens drags to 400 ms.
Use `--swipe-ms` explicitly when calibrating a device and verify with
`--moves-per-cycle 1` before batching. Routine live captures remain in memory;
`--save-screenshots` or `--debug-draws` writes the diagnostic PNG.

Duplicate exposed card identities are reported as parsing issues and both
conflicting reads become unknown, preserving their physical slots. The live
stock tracker synchronizes at an empty waste and learns exposed identities
only after fresh captures confirm the dispatched moves. It retains the full
waste history behind the visible fan and reuses learned draw order on redeals.
Untrusted or contradictory captures discard history; starting in the middle
of a stock pass leaves its unseen history unknown until an empty-waste
boundary is observed. Redeals return only the remaining waste cards, never
replenishing cards already moved to tableau or foundations.

Race mode can batch known tableau moves while refreshing immediately after a
waste-card move, a draw/redeal, or a hidden-card reveal. This avoids drawing
past the newly exposed waste card based on a partly covered read. For a live
run with short pauses and the existing drag timing:

```bash
python3 solitaire_auto_bot.py --solver race --moves-per-cycle 5 --interval 0.3 --logcat
```

Live screenshots use validated raw ADB capture (with PNG fallback) and are
passed directly to the calibrated reader. Saved diagnostics use lossless PNG
compression level 1. Waste detections within 20 pixels are merged; covered
cards remain unknown until exposed. These changes preserve move scoring and
swipe duration. Capture latency still varies with USB and phone load. New foundation suits are
confirmed from the card just played, or from a second consistent capture when
that evidence is unavailable. Ambiguous reads retry after a short pause;
established foundation history is retained through obscured frames. The bot
stops after three captures showing a covered gameplay area or an unsupported
resolution, including the score panel, and never submits the score itself.

Run the maintained tests from the project root:

```bash
python -m pytest tests test_logging.py -q
```

---

## Session Logging and Android Logcat

Solvitaire can now write two complementary logs:

- A structured JSONL session log containing OCR timing, unresolved-card counts, solver timing, Monte Carlo statistics, selected moves, and gesture coordinates.
- A raw Android `logcat` capture for diagnosing app, input, rendering, and device-side behavior.

### Structured logging only

```bash
python3 solitaire_auto_bot.py \
  --sim Gameplay/frame_0108.png \
  --solver monte-carlo \
  --log-file logs/test_session.jsonl
```

Each line in the JSONL file is a complete JSON object, making the file easy to inspect with `jq`, Python, or a spreadsheet import.

Summarize one session's timing and capture-confirmed transitions:

```bash
python3 session_metrics.py logs/my_session.jsonl
```

The report separates gesture dispatches, confirmed moves, confirmed unchanged
boards, unverified moves, and moves still awaiting a capture. Confirmation
uses the synchronized history and recognized state; it is not a ground-truth
OCR accuracy or a measured final game score. Unsynchronized or contradictory
reads are unverified, not counted as successful gestures. It also reports
median capture/OCR/planning/cycle times and observed foundation progress.

Example:

```bash
jq . logs/test_session.jsonl | less
```

### Find the Android package name

Open the game on the connected Android device, then try:

```bash
adb shell dumpsys window | grep -E 'mCurrentFocus|mFocusedApp'
```

Or search installed package names:

```bash
adb shell pm list packages | grep -i solitaire
```

The package name will look similar to `com.example.solitaire`.

### Live bot with logcat

```bash
python3 solitaire_auto_bot.py \
  --solver monte-carlo \
  --logcat \
  --clear-logcat \
  --logcat-package com.example.solitaire
```

When `--logcat` is enabled, Solvitaire automatically creates:

```text
logs/session_YYYYMMDD_HHMMSS.jsonl
logs/logcat_YYYYMMDD_HHMMSS.log
```

Use explicit paths when preferred:

```bash
python3 solitaire_auto_bot.py \
  --solver search \
  --logcat \
  --log-file logs/my_session.jsonl \
  --logcat-file logs/my_android.log
```

Restrict the raw capture with one or more regular-expression filters:

```bash
python3 solitaire_auto_bot.py \
  --logcat \
  --logcat-filter 'InputDispatcher|InputReader' \
  --logcat-filter 'solitaire|card|move'
```

Package filtering uses the app's current process ID when it can be resolved. If the package is not running or Android cannot provide a PID, Solvitaire falls back to capturing the available logcat stream rather than silently producing no diagnostics.

### Capture logcat without running the bot

```bash
python3 logcat_monitor.py \
  --clear \
  --package com.example.solitaire \
  --output logs/manual_logcat.log
```

Stop it with `Ctrl+C`, or capture for a fixed number of seconds:

```bash
python3 logcat_monitor.py \
  --duration 20 \
  --output logs/manual_logcat.log
```

Android applications do not always emit useful game-specific messages. Even when the app is quiet, system tags such as input dispatch, activity lifecycle, crashes, and rendering warnings may still help diagnose failed gestures or UI timing issues.
