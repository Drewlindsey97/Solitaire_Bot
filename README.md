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

On ADB, live capture uses a native-size 720×1600, 20 Mbps H.264 screen stream
decoded by `ffmpeg` in a background worker during OCR, planning, and gestures.
Only the newest decoded frame is retained. It must agree with its predecessor,
and a 250 ms encoding/transport allowance is required after the last gesture.
H.264 is compressed, so reader checks remain necessary. Moving, stale,
disconnected, or already-consumed frames never reuse an older stable frame.
Timer and score changes are excluded from comparison. `ffmpeg` must be installed
for this ADB capture path. Other backends retain their screenshot capture.
This reduces animation misreads but does not eliminate recognition errors
on a stable frame. Routine captures stay in memory; `--save-screenshots`
saves accepted frames and `--debug-draws` saves selected diagnostic frames.
`--fast` uses a 0.05-second pause between batches, preserves the reliable
720–860 ms drag duration, and relies on the frame check to wait for animations.
`--interval 0` removes that fixed pause and waits only for a fresh settled frame.
Frame sequence, age, wait time, and motion fraction are recorded in the session log.

Race mode can continue independent known-card moves after a hidden-card reveal.
The revealed card remains unknown until the next capture, and drawing waits
for that read. A waste-card move or draw/redeal still ends the batch. For a live
run with up to ten moves per read and shorter drags:

```bash
python3 solitaire_auto_bot.py --solver race --moves-per-cycle 10 --swipe-ms 650 --fast --interval 0 --logcat
```

Live frames are passed directly from the background buffer to the calibrated
reader. Saved diagnostics use lossless PNG
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
