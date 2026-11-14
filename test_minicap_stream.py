"""Standalone smoke test for minicap_stream against the current adb device."""
import sys, time
from minicap_stream import MinicapStream

serial = sys.argv[1] if len(sys.argv) > 1 else None
prefix = ["adb", "-s", serial] if serial else ["adb"]

s = MinicapStream(adb_prefix=prefix)
print("[*] starting minicap stream...")
banner = s.start()
print("[*] banner:", banner)

img = s.wait_for_frame(timeout=8)
if img is None:
    print("[FAIL] no frame received within timeout")
    s.stop(); sys.exit(1)
print("[*] first frame size (WxH):", img.size)

# Measure sustained fps over ~3s of pulling latest frames.
t0 = time.time()
n0 = s.frame_count
while time.time() - t0 < 3.0:
    _ = s.read_latest()
    time.sleep(0.01)
dt = time.time() - t0
produced = s.frame_count - n0
print("[*] producer fps (frames minicap delivered): %.1f" % (produced / dt))
print("[*] total frames:", s.frame_count)

img.save("minicap_test_frame.png")
print("[*] saved minicap_test_frame.png")
ok = img.size == (720, 1600)
print("[%s] frame dims match calibrated 720x1600" % ("PASS" if ok else "FAIL"))
s.stop()
sys.exit(0 if ok else 2)
