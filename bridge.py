import io
import os
import sys
import time
import random
import subprocess
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

# Detect if the runtime environment is Android (Pydroid 3, Termux, etc.)
IS_ANDROID = os.path.exists("/system/bin/app_process") or "ANDROID_ROOT" in os.environ

if IS_ANDROID:
    # If running on Android, default to LOCAL_ROOT if root binary is found, otherwise LOCAL_LADB
    is_rooted = os.path.exists("/system/xbin/su") or os.path.exists("/system/bin/su")
    RUN_MODE = "LOCAL_ROOT" if is_rooted else "LOCAL_LADB"
else:
    # Default to HTTP_BRIDGE on desktop environments (keeps backwards compatibility)
    RUN_MODE = "HTTP_BRIDGE"

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
        cmd_str = " ".join(cmd_list)
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
# 4. HUMAN AUTOMATION MATHEMATICS
# ==============================================================================
def apply_jitter(val, radius):
    """Adds small Gaussian jitter to simulate human variance."""
    offset = int(random.gauss(0, radius / 2))
    return max(0, val + offset)

def wait_human_delay(min_d=0.2, max_d=0.8):
    """Sleeps a randomized delay. Adds an occasional scanning pause."""
    if random.random() < 0.05:  # 5% chance of a realistic 'screen scanning' pause
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

    if HUMAN_MODE:
        wait_human_delay()


def swipe(x1, y1, x2, y2):
    """Executes a swipe action. Applies path coordinate jitter and random swipe durations."""
    jx1 = apply_jitter(x1, SWIPE_JITTER_RADIUS) if HUMAN_MODE else x1
    jy1 = apply_jitter(y1, SWIPE_JITTER_RADIUS) if HUMAN_MODE else y1
    jx2 = apply_jitter(x2, SWIPE_JITTER_RADIUS) if HUMAN_MODE else x2
    jy2 = apply_jitter(y2, SWIPE_JITTER_RADIUS) if HUMAN_MODE else y2

    duration_ms = random.randint(250, 450) if HUMAN_MODE else 300

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

    if HUMAN_MODE:
        wait_human_delay()


def screenshot():
    """Fetch a screenshot from the bridge and return it as a PIL Image."""
    from PIL import Image

    if RUN_MODE == "HTTP_BRIDGE":
        resp = requests.get(f"{BASE_URL}/screenshot", timeout=10)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    else:
        # For non-HTTP modes, capture screen directly using Android command and pull it
        temp_path = "/sdcard/screen_tmp.png"
        dest_path = "screen_tmp.png"
        print(f"Capturing screenshot via shell ({RUN_MODE})...")
        run_cmd(["shell", "screencap", "-p", temp_path])
        
        # Pull the file if in PC_ADB mode, otherwise load it locally
        if RUN_MODE == "PC_ADB":
            run_cmd(["pull", temp_path, dest_path])
            img = Image.open(dest_path).convert("RGB")
            # Cleanup local temp file
            if os.path.exists(dest_path):
                os.remove(dest_path)
            return img
        else:
            # Running directly on Android
            img = Image.open(temp_path).convert("RGB")
            return img


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
