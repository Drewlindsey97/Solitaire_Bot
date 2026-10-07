import io
import os
import sys
import time
import random
import subprocess
import shlex
import struct
import requests

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
BASE_URL = "http://localhost:8080"

# Execution Mode Options:
#   "HTTP_BRIDGE" - Default. Calls local HTTP server endpoints (/tap, /swipe, /screenshot).
#   "PC_ADB"      - Calls PC ADB tool to control Android device.
#   "LOCAL_ROOT"  - Run directly in Pydroid 3 / Termux using `su -c`.
#   "LOCAL_LADB"  - Run directly in Pydroid 3 / Termux using local wireless debugging connection.
#   "INTENT_ONLY" - Direct Tasker intent broadcasts (useful for AutoInput integration).
RUN_MODE = "HTTP_BRIDGE"

# Enable/disable human-like automation properties (jitter, duration variations)
HUMAN_MODE = True

# Jitter config: standard deviation in pixels for target clicks
TAP_JITTER_RADIUS = 6
SWIPE_JITTER_RADIUS = 15

# Gesture pacing.
#
# Two independent knobs, previously conflated:
#
# 1. Swipe DURATION (SWIPE_MS_*): how long the drag itself takes. This is
#    load-bearing - a swipe much faster than ~700ms registers as a fling,
#    not a card drag, and the game rejects it. Lower it only with live
#    verification (watch for "Board unchanged after attempting" warnings).
#
# 2. Inter-gesture PAUSE (POST_TAP_S / POST_SWIPE_S): a blind sleep AFTER
#    the gesture's own blocking input event has already completed. This was
#    a "human-like" affectation (0.9-1.5s), not a correctness requirement:
#    the drag has physically finished by the time `input swipe` returns, and
#    every cycle re-reads the real board, so a stale/interrupted move is
#    caught by feedback rather than prevented by waiting. Competitive play
#    keeps it at zero - the dominant per-move cost was this dead time.
#    Foundation plays are taps to independent piles and need no settle at
#    all; a small POST_SWIPE_S can be dialed in only if batched column
#    swipes are seen snapping back before the next gesture (set via
#    --gesture-delay). HUMAN_PACING restores the old randomized profile for
#    stealth runs (see --human).
# Swipe duration is randomized per gesture (randint across this range), so
# no two drags fire at exactly the same speed - varied cadence both reads
# as human and avoids a fixed fingerprint. The 560-780 range previously used
# here was not comfortably above the ~700ms fling threshold: a live run on
# 2026-10-06 saw repeated "Board unchanged after attempting" failures on the
# same move at 754ms, 760ms, and 636ms, stalling the bot via max-stuck-cycles.
# Raised so the floor clears 700ms with margin; lower it again only with live
# verification (watch for "Board unchanged after attempting" warnings that
# mean a swipe flung instead of dragged).
SWIPE_MS_MIN, SWIPE_MS_MAX = 720, 860
POST_TAP_S = 0.0
POST_SWIPE_S = 0.0
HUMAN_PACING = False
# Retained for --human stealth pacing (see wait_human_delay); unused on the
# default competitive path.
DELAY_MIN_S, DELAY_MAX_S = 0.9, 1.5
SCAN_PAUSE_CHANCE = 0.05

def configure_timing(swipe_ms=None, delay_min=None, delay_max=None,
                     scan_pause_chance=None, post_tap=None, post_swipe=None,
                     human_pacing=None):
    """Override gesture pacing at runtime (used by the bot's --fast /
    --swipe-ms / --gesture-delay / --human flags)."""
    global SWIPE_MS_MIN, SWIPE_MS_MAX, DELAY_MIN_S, DELAY_MAX_S, SCAN_PAUSE_CHANCE
    global POST_TAP_S, POST_SWIPE_S, HUMAN_PACING
    if swipe_ms is not None:
        # keep the human-mode randomness as a ~±12% spread around the request
        spread = max(1, swipe_ms // 8)
        SWIPE_MS_MIN, SWIPE_MS_MAX = swipe_ms - spread, swipe_ms + spread
    if delay_min is not None:
        DELAY_MIN_S = delay_min
    if delay_max is not None:
        DELAY_MAX_S = delay_max
    if scan_pause_chance is not None:
        SCAN_PAUSE_CHANCE = scan_pause_chance
    if post_tap is not None:
        POST_TAP_S = post_tap
    if post_swipe is not None:
        POST_SWIPE_S = post_swipe
    if human_pacing is not None:
        HUMAN_PACING = human_pacing

# Detect if the runtime environment is Android (Pydroid 3, Termux, etc.)
IS_ANDROID = os.path.exists("/system/bin/app_process") or "ANDROID_ROOT" in os.environ

if IS_ANDROID:
    # If running on Android, default to LOCAL_ROOT if root binary is found, otherwise LOCAL_LADB
    is_rooted = os.path.exists("/system/xbin/su") or os.path.exists("/system/bin/su")
    RUN_MODE = "LOCAL_ROOT" if is_rooted else "LOCAL_LADB"
else:
    # Default to PC_ADB on desktop environments to control connected phone
    RUN_MODE = "PC_ADB"

print(f"[*] bridge.py environment: {'Android (Pydroid 3/Termux)' if IS_ANDROID else 'Desktop OS'}")
print(f"[*] bridge.py RUN_MODE set to: {RUN_MODE} (HUMAN_MODE: {HUMAN_MODE})")

# ==============================================================================
# 2. RUN COMMAND UTILITY
# ==============================================================================
def run_cmd(cmd_list):
    """Executes target commands locally or via ADB depending on the current RUN_MODE."""
    if RUN_MODE == "PC_ADB":
        full_cmd = ["adb"] + cmd_list
    elif RUN_MODE == "LOCAL_ROOT":
        # `shell` is an adb subcommand, not an Android executable.
        local_args = cmd_list[1:] if cmd_list and cmd_list[0] == "shell" else cmd_list
        cmd_str = shlex.join(local_args)
        full_cmd = ["su", "-c", cmd_str]
    elif RUN_MODE == "LOCAL_LADB":
        full_cmd = ["adb", "-s", "localhost:5555"] + cmd_list
    else:
        full_cmd = cmd_list

    try:
        result = subprocess.run(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"[Error] Command execution failed: {' '.join(full_cmd)}\nError: {e.stderr.strip()}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print(f"[Error] Executable command not found for mode '{RUN_MODE}'. Verify setup.", file=sys.stderr)
        return None

def _adb_prefix():
    """adb argv prefix for the current adb-backed mode (device selector included)."""
    if RUN_MODE == "LOCAL_LADB":
        return ["adb", "-s", "localhost:5555"]
    return ["adb"]

def decode_raw_screenshot(data):
    """Decode Android's 12/16-byte screencap header; reject malformed frames."""
    from PIL import Image
    if len(data) < 12:
        raise ValueError("truncated screencap header")
    width, height, pixel_format = struct.unpack("<3I", data[:12])
    formats = {1: ("RGBA", 4), 2: ("RGBX", 4), 3: ("RGB", 3)}
    if pixel_format not in formats or not (0 < width <= 8192 and 0 < height <= 8192):
        raise ValueError("unsupported screencap dimensions or pixel format")
    mode, channels = formats[pixel_format]
    offset = len(data) - width * height * channels
    if offset not in (12, 16):
        raise ValueError("truncated or oversized screencap pixel data")
    return Image.frombytes(mode, (width, height), data[offset:]).convert("RGB")


def adb_capture_raw_image():
    """Skip on-device PNG encoding while preserving all captured pixels."""
    try:
        result = subprocess.run(
            _adb_prefix() + ["exec-out", "screencap"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=15,
        )
        return decode_raw_screenshot(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"[Warn] Raw screenshot unavailable ({exc}); falling back to PNG.",
              file=sys.stderr)
        return None

def adb_capture_png_bytes():
    """Grab a screenshot as PNG bytes over `adb exec-out screencap -p`, with no
    on-device temp file and no separate pull - roughly 2x faster than the
    screencap-to-file-then-pull path (measured ~660ms vs ~1290ms), which is
    the slowest step in every read cycle. Returns raw PNG bytes, or None on
    failure. exec-out streams screencap's stdout straight back over the adb
    socket, so it's binary-safe (unlike `adb shell`, which mangles CRLF)."""
    try:
        result = subprocess.run(
            _adb_prefix() + ["exec-out", "screencap", "-p"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[Error] exec-out screencap failed: {e.stderr.decode(errors='replace').strip()}",
              file=sys.stderr)
        return None
    except FileNotFoundError:
        print("[Error] adb not found for screenshot capture.", file=sys.stderr)
        return None
    data = result.stdout
    if not data.startswith(b"\x89PNG"):
        # A transient device state (screen off, mid-reconnect) can yield a
        # truncated/empty stream; treat as a failed read so the caller retries.
        print(f"[Error] exec-out returned {len(data)} bytes, not a PNG; skipping this read.",
              file=sys.stderr)
        return None
    return data

# ==============================================================================
# 3. TASKER/AUTOINPUT BROADCAST
# ==============================================================================
def trigger_tasker_autoinput(task_name, parameter_dict=None):
    """
    Sends an intent broadcast to Tasker to run an AutoInput UI automation task.
    This works on non-rooted Android devices running Pydroid 3 without ADB setup.
    """
    print(f"[AutoInput] Triggering Tasker task '{task_name}' with parameters {parameter_dict or {}}")
    intent_args = [
        "shell", "am", "broadcast",
        "-a", "net.dinglisch.android.tasker.ACTION_TASK",
        "--es", "task_name", task_name
    ]
    if parameter_dict:
        for i, (key, value) in enumerate(parameter_dict.items(), start=1):
            intent_args += ["--es", f"var{i}", str(key), "--es", f"val{i}", str(value)]
    run_cmd(intent_args)

# ==============================================================================
# 3.5. GPS LOCATION SPOOFING
# ==============================================================================
def spoof_gps(latitude, longitude):
    """
    Sets a mock GPS location on the Android device via ADB shell.
    Requires 'Allow mock locations' (Developer options) enabled, or root, 
    or setting mock location app via settings:
       adb shell appops set <mock_app_package> android:mock_location allow
    """
    print(f"[*] Setting mock GPS location: Lat={latitude}, Lon={longitude}")
    
    # 1. Standard Android mock location setting (requires developer options / app setting enabled)
    # We send mock location values using Android's location provider command (geo fix)
    # Note: 'geo fix' requires telnet to emulator, or we can write mock locations using root settings provider.
    # Alternatively, we can trigger location mock apps via command intents.
    # Below uses the standard appops & location service method:
    run_cmd(["shell", "settings", "put", "secure", "mock_location", "1"])
    
    # Send broadcast intent to common location spoofer apps or set coordinates directly if rooted
    # Here, we run mock location provider update:
    run_cmd(["shell", "cmd", "location", "set-location-allow-mock", "true"])
    run_cmd(["shell", "cmd", "location", "providers", "set-test-provider-location", "gps", 
             "--latitude", str(latitude), "--longitude", str(longitude), "--accuracy", "5.0"])
    print("[*] GPS spoof coordinates updated.")

# ==============================================================================
# 4. HUMAN AUTOMATION MATHEMATICS
# ==============================================================================
def apply_jitter(val, radius):
    """Adds small Gaussian jitter to simulate human variance."""
    offset = int(random.gauss(0, radius / 2))
    return max(0, val + offset)

def wait_human_delay(min_d=None, max_d=None):
    """Sleeps a randomized delay. Adds an occasional scanning pause.

    Runs after every tap/swipe, including between moves within a
    multi-move batch (see solitaire_auto_bot.py's --moves-per-cycle) - the
    input event itself blocks until the touch/drag completes, but the
    game's own card-settle/snap animation keeps playing after that. A
    pause shorter than that animation lets the next gesture fire while a
    card is still mid-animation, which can interrupt it (snapping back to
    its origin) or land on a not-yet-stable layout - confirmed live: a
    6-move batch's first two moves landed, then every move after that
    silently failed once the pause was cut to 0.2-0.8s.
    """
    min_d = DELAY_MIN_S if min_d is None else min_d
    max_d = DELAY_MAX_S if max_d is None else max_d
    if random.random() < SCAN_PAUSE_CHANCE:  # occasional realistic 'screen scanning' pause
        scan_delay = random.uniform(1.2, 2.5)
        print(f"   [Human Mode] Pausing to scan screen for {scan_delay:.2f}s...")
        time.sleep(scan_delay)
    else:
        time.sleep(random.uniform(min_d, max_d))

# ==============================================================================
# 5. PUBLIC API (TAP, SWIPE, SCREENSHOT)
# ==============================================================================
def tap(x, y):
    """Executes a tap action. Applies human jitter and duration if enabled."""
    target_x = apply_jitter(x, TAP_JITTER_RADIUS) if HUMAN_MODE else x
    target_y = apply_jitter(y, TAP_JITTER_RADIUS) if HUMAN_MODE else y
    
    # average human hold duration is 75-125ms
    hold_duration_ms = random.randint(75, 125) if HUMAN_MODE else 100

    if RUN_MODE == "HTTP_BRIDGE":
        print(f"Tap at ({target_x}, {target_y}) via HTTP")
        try:
            requests.post(f"{BASE_URL}/tap", json={"x": target_x, "y": target_y}, timeout=5)
        except requests.exceptions.RequestException as e:
            print(f"Tap request failed: {e}")
    elif RUN_MODE == "INTENT_ONLY":
        trigger_tasker_autoinput("AutoInputTap", {"x": target_x, "y": target_y, "duration": hold_duration_ms})
    else:
        print(f"Tap at ({target_x}, {target_y}) via shell ({RUN_MODE}) for {hold_duration_ms}ms")
        # In Android shell, executing a short swipe on the same coordinate acts as a tap with custom duration
        run_cmd(["shell", "input", "swipe", str(target_x), str(target_y), str(target_x), str(target_y), str(hold_duration_ms)])

    # No blind pause on the competitive path: a foundation tap flies to an
    # independent pile and the next cycle re-reads the real board. The old
    # 0.9-1.5s human pause was dead time (HUMAN_PACING restores it).
    if HUMAN_PACING:
        wait_human_delay()
    elif POST_TAP_S > 0:
        time.sleep(POST_TAP_S)


def swipe(x1, y1, x2, y2):
    """Executes a swipe action. Applies path coordinate jitter and random swipe durations."""
    jx1 = apply_jitter(x1, SWIPE_JITTER_RADIUS) if HUMAN_MODE else x1
    jy1 = apply_jitter(y1, SWIPE_JITTER_RADIUS) if HUMAN_MODE else y1
    jx2 = apply_jitter(x2, SWIPE_JITTER_RADIUS) if HUMAN_MODE else x2
    jy2 = apply_jitter(y2, SWIPE_JITTER_RADIUS) if HUMAN_MODE else y2

    duration_ms = random.randint(SWIPE_MS_MIN, SWIPE_MS_MAX) if HUMAN_MODE \
        else (SWIPE_MS_MIN + SWIPE_MS_MAX) // 2

    if RUN_MODE == "HTTP_BRIDGE":
        print(f"Swipe from ({jx1}, {jy1}) to ({jx2}, {jy2}) via HTTP")
        try:
            # We append the duration_ms parameter. Even if the server ignores it, it remains compliant.
            requests.post(
                f"{BASE_URL}/swipe",
                json={"x1": jx1, "y1": jy1, "x2": jx2, "y2": jy2, "duration": duration_ms},
                timeout=5,
            )
        except requests.exceptions.RequestException as e:
            print(f"Swipe request failed: {e}")
    elif RUN_MODE == "INTENT_ONLY":
        trigger_tasker_autoinput("AutoInputSwipe", {"x1": jx1, "y1": jy1, "x2": jx2, "y2": jy2, "duration": duration_ms})
    else:
        print(f"Swipe from ({jx1}, {jy1}) to ({jx2}, {jy2}) via shell ({RUN_MODE}) over {duration_ms}ms")
        run_cmd(["shell", "input", "swipe", str(jx1), str(jy1), str(jx2), str(jy2), str(duration_ms)])

    # The drag has physically completed by the time input swipe returns.
    # Competitive default adds no settle; dial POST_SWIPE_S up (--gesture-delay)
    # only if batched column swipes are seen snapping back mid-animation.
    if HUMAN_PACING:
        wait_human_delay()
    elif POST_SWIPE_S > 0:
        time.sleep(POST_SWIPE_S)


def screenshot():
    """Fetch a screenshot from the bridge and return it as a PIL Image."""
    from PIL import Image

    if RUN_MODE == "HTTP_BRIDGE":
        try:
            resp = requests.get(f"{BASE_URL}/screenshot", timeout=10)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            print(f"[Error] Failed to fetch screenshot from HTTP bridge: {e}", file=sys.stderr)
            return None
    elif RUN_MODE in ("PC_ADB", "LOCAL_LADB"):
        raw_image = adb_capture_raw_image()
        if raw_image is not None:
            return raw_image
        # Fallback: stream the PNG straight over exec-out - no on-device
        # temp file, no separate pull (~2x faster; see adb_capture_png_bytes).
        data = adb_capture_png_bytes()
        if data is None:
            return None
        try:
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:
            print(f"[Error] Failed to decode exec-out screenshot: {e}", file=sys.stderr)
            return None
    else:
        # Running directly on Android (LOCAL_ROOT / on-device): capture to a
        # temp file and load it locally - no host to pull to.
        temp_path = "/sdcard/screen_tmp.png"
        run_cmd(["shell", "screencap", "-p", temp_path])
        if not os.path.exists(temp_path):
            print("[Error] Screenshot file not found on device.", file=sys.stderr)
            return None
        try:
            return Image.open(temp_path).convert("RGB")
        except Exception as e:
            print(f"[Error] Failed to open screenshot on device: {e}", file=sys.stderr)
            return None


# ==============================================================================
# 6. DEMO / VERIFICATION ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    # Test human touch features locally with logging
    print("\n--- Running bridge.py Test/Demo ---")
    print("Testing coordinate jitter generation:")
    for _ in range(3):
        original_x, original_y = 500, 800
        jx = apply_jitter(original_x, TAP_JITTER_RADIUS)
        jy = apply_jitter(original_y, TAP_JITTER_RADIUS)
        print(f"  Target: ({original_x}, {original_y}) -> Jittered: ({jx}, {jy})")
        
    print("\nTesting swipe duration generation:")
    for _ in range(3):
        duration = random.randint(250, 450)
        print(f"  Randomized human swipe duration: {duration}ms")
    
    print("\nNote: Make sure your run environment and port configurations are set up.")
