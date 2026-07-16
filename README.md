# Solitaire Stash Automation Bot

This bot integrates a computer vision board reader, a FreeCell-style solver, and a human-like automation bridge to play and solve "Solitaire Stash" (7 tableau columns, 4 free cells/stash slots, and foundation piles) directly on an Android device or emulator.

## Features
- **Computer Vision Card Reader**: Uses OpenCV template matching to read cards from screenshots.
- **State Solver**: Uses best-first search and heuristics to compute card-clearing paths.
- **Human Gesture Emulation**: Taps and swipes incorporate Gaussian coordinate jittering, randomized hold durations, and dynamic pauses to simulate a human user.
- **Multiple Execution Backends**: Supports PC-to-Android ADB, rooted on-device execution (Pydroid 3 / Termux), wireless local debugging (LADB), and Tasker/AutoInput intent relays.
- **Simulation Mode**: Includes a dry-run feature (`--sim`) to test the pipeline on static mock images without requiring a connected device.

---

## File Structure
- [solitaire_auto_bot.py](file:///Users/mastercontrol/.gemini/antigravity/scratch/Solitaire_Bot/solitaire_auto_bot.py): Main bot automation script (main loop, coordinate mapper, suit mapping, gesture execution).
- [bridge.py](file:///Users/mastercontrol/.gemini/antigravity/scratch/Solitaire_Bot/bridge.py): Multi-mode automation bridge (handles direct shell inputs, Tasker intents, and human click dynamics).
- [board_reader_lib.py](file:///Users/mastercontrol/.gemini/antigravity/scratch/Solitaire_Bot/board_reader_lib.py): CV board state parser.
- [freecell_solver.py](file:///Users/mastercontrol/.gemini/antigravity/scratch/Solitaire_Bot/freecell_solver.py): Card-clearing algorithm engine.

---

## Installation & Setup

1. **Python Dependencies**:
   Install OpenCV, NumPy, Pillow, and Requests:
   ```bash
   pip3 install opencv-python numpy pillow requests
   ```

2. **Android Setup**:
   Ensure your Android device has **USB Debugging** enabled and is connected via ADB.

3. **Running Modes**:
   By default, `bridge.py` auto-detects if it is running on a PC (defaults to `HTTP_BRIDGE` or `PC_ADB`) or locally on Android inside Pydroid 3 (defaults to `LOCAL_ROOT` or `LOCAL_LADB`). You can configure the `RUN_MODE` at the top of `bridge.py`.

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
