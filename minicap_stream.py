"""
minicap_stream.py - Continuous framebuffer streaming via minicap.

Why this exists
---------------
The live capture path in bridge.py grabs one frame per `adb exec-out screencap`
round-trip (~hundreds of ms each, so ~1.5-2 fps). For a screen-scraping bot the
capture round-trip is the dominant per-cycle cost. minicap streams JPEG frames
off SurfaceFlinger over a forwarded socket at 30-60 fps, so the bot reads the
*latest already-decoded* frame instead of blocking on a fresh grab.

What it does
------------
- Picks the matching prebuilt `minicap` binary + `minicap.so` for the device's
  ABI and SDK level, pushes them to /data/local/tmp, and starts minicap with a
  full-resolution projection.
- `adb forward`s minicap's localabstract socket to a local TCP port.
- A background thread reads the minicap banner + length-prefixed JPEG frames and
  keeps only the most recent JPEG (newest-wins; slow consumers never queue up).
- `read_latest()` decodes that JPEG to a PIL RGB Image - the same return shape as
  bridge.adb_capture_raw_image(), so it drops into the existing reader pipeline.

This is a prototype: it is wired in behind an explicit opt-in (bridge.enable_minicap())
and always falls back to screencap if a frame is not available.

minicap wire protocol (global header is 24 bytes, little-endian):
    byte  0     : version (1)
    byte  1     : header length (24)
    bytes 2-5   : pid
    bytes 6-9   : real display width
    bytes 10-13 : real display height
    bytes 14-17 : virtual (projected) width
    bytes 18-21 : virtual (projected) height
    byte  22    : display orientation
    byte  23    : quirk bitflags
Then, repeating: uint32 LE frame length, followed by that many JPEG bytes.
"""

import io
import os
import re
import socket
import struct
import subprocess
import threading
import time

from PIL import Image

# Where the vendored prebuilt binaries live: vendor/minicap/<abi>/bin/minicap
# and vendor/minicap/<abi>/lib/android-<sdk>/minicap.so
_VENDOR_DIR = os.environ.get(
    "MINICAP_VENDOR_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "minicap"),
)

_DEVICE_TMP = "/data/local/tmp"
_REMOTE_BIN = _DEVICE_TMP + "/minicap"
_REMOTE_LIB = _DEVICE_TMP + "/minicap.so"


class MinicapError(RuntimeError):
    pass


class MinicapStream:
    """A single live minicap capture stream for one device."""

    def __init__(self, adb_prefix=None, local_port=1313):
        # adb_prefix is the argv list up to and including device selection,
        # e.g. ["adb"] or ["adb", "-s", "emulator-5554"]. Mirrors bridge._adb_prefix().
        self._adb = list(adb_prefix) if adb_prefix else ["adb"]
        self._port = int(local_port)
        self._proc = None            # the `adb shell ... minicap` process
        self._sock = None
        self._reader = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest_jpeg = None     # newest raw JPEG bytes (newest-wins)
        self._decoded_cache = None   # (id(jpeg), PIL.Image) to skip re-decoding
        self._frame_count = 0
        self._started_at = None
        self._banner = {}            # parsed global header

    # -- adb helpers -------------------------------------------------------
    def _run(self, args, **kw):
        return subprocess.run(self._adb + args, capture_output=True, **kw)

    def _getprop(self, prop):
        out = self._run(["shell", "getprop", prop]).stdout.decode(errors="replace")
        return out.strip()

    def _display_size(self):
        """Return (width, height) of the physical display, e.g. (720, 1600)."""
        out = self._run(["shell", "wm", "size"]).stdout.decode(errors="replace")
        m = re.search(r"Physical size:\s*(\d+)x(\d+)", out)
        if not m:
            m = re.search(r"(\d+)x(\d+)", out)
        if not m:
            raise MinicapError("could not determine display size from `wm size`")
        return int(m.group(1)), int(m.group(2))

    # -- binary selection + push ------------------------------------------
    def _select_binaries(self):
        abi = self._getprop("ro.product.cpu.abi")
        sdk = self._getprop("ro.build.version.sdk")
        if not abi or not sdk:
            raise MinicapError("could not read device abi/sdk via getprop")
        bin_path = os.path.join(_VENDOR_DIR, abi, "bin", "minicap")
        lib_path = os.path.join(_VENDOR_DIR, abi, "lib", "android-%s" % sdk, "minicap.so")
        if not os.path.exists(bin_path):
            raise MinicapError("no vendored minicap binary for abi=%s (looked in %s)"
                               % (abi, bin_path))
        if not os.path.exists(lib_path):
            raise MinicapError("no vendored minicap.so for abi=%s sdk=%s (looked in %s)"
                               % (abi, sdk, lib_path))
        return abi, sdk, bin_path, lib_path

    def _push(self, local, remote):
        r = self._run(["push", local, remote])
        if r.returncode != 0:
            raise MinicapError("adb push failed: %s" % r.stderr.decode(errors="replace").strip())
        self._run(["shell", "chmod", "755", remote])

    # -- lifecycle ---------------------------------------------------------
    def start(self, projection=None, timeout=10.0):
        """Push binaries, start minicap, forward the socket, and begin reading.

        projection: "RWxRH@VWxVH/orientation". Defaults to full-res upright,
        i.e. real == virtual, orientation 0.
        """
        abi, sdk, bin_path, lib_path = self._select_binaries()
        self._push(bin_path, _REMOTE_BIN)
        self._push(lib_path, _REMOTE_LIB)

        w, h = self._display_size()
        if projection is None:
            projection = "%dx%d@%dx%d/0" % (w, h, w, h)

        # Forward a local TCP port to minicap's abstract socket.
        self._run(["forward", "tcp:%d" % self._port, "localabstract:minicap"])

        # Start minicap. It creates localabstract:minicap and streams frames.
        # -Q 95 sets JPEG quality; default is fine but explicit keeps it stable.
        shell_cmd = ("LD_LIBRARY_PATH=%s %s -P %s -Q 90"
                     % (_DEVICE_TMP, _REMOTE_BIN, projection))
        self._proc = subprocess.Popen(
            self._adb + ["shell", shell_cmd],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        # `adb forward` accepts the local TCP connection before minicap has
        # created its localabstract socket, then drops it - so a bare connect
        # "succeeds" and the banner read hits a closed socket. Retry the whole
        # connect+banner handshake until a valid banner arrives.
        deadline = time.time() + timeout
        last_err = None
        while time.time() < deadline and self._banner == {}:
            if self._proc.poll() is not None:
                err = self._proc.stderr.read().decode(errors="replace")
                raise MinicapError("minicap exited during startup: %s" % err.strip())
            try:
                s = socket.create_connection(("127.0.0.1", self._port), timeout=1.0)
                s.settimeout(2.0)
                self._sock = s
                self._read_banner()           # fills self._banner on success
            except (OSError, MinicapError) as e:
                last_err = e
                if self._sock is not None:
                    try:
                        self._sock.close()
                    except OSError:
                        pass
                    self._sock = None
                self._banner = {}
                time.sleep(0.15)
        if self._sock is None or self._banner == {}:
            self.stop()
            raise MinicapError("could not connect/handshake with minicap: %s" % last_err)

        # Streaming is change-driven: on a static screen minicap emits nothing,
        # so a read may block arbitrarily long. Use a blocking socket and rely
        # on stop() closing it to unblock; never let an idle gap kill the reader.
        self._sock.settimeout(None)
        self._started_at = time.time()
        self._reader = threading.Thread(target=self._read_loop, name="minicap-reader",
                                        daemon=True)
        self._reader.start()
        return self._banner

    def _recv_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise MinicapError("minicap socket closed while reading")
            buf.extend(chunk)
        return bytes(buf)

    def _read_banner(self):
        head = self._recv_exact(2)
        version, hdr_len = head[0], head[1]
        rest = self._recv_exact(hdr_len - 2)
        pid, rw, rh, vw, vh = struct.unpack_from("<IIIII", rest, 0)
        orientation = rest[20]
        quirks = rest[21]
        self._banner = dict(version=version, length=hdr_len, pid=pid,
                            real_width=rw, real_height=rh,
                            virtual_width=vw, virtual_height=vh,
                            orientation=orientation, quirks=quirks)

    def _read_loop(self):
        try:
            while not self._stop.is_set():
                (frame_len,) = struct.unpack("<I", self._recv_exact(4))
                jpeg = self._recv_exact(frame_len)
                with self._lock:
                    self._latest_jpeg = jpeg
                    self._frame_count += 1
        except (MinicapError, OSError):
            # Socket closed / stream ended: thread exits; read_latest() will
            # just keep returning the last frame (or None) and callers fall
            # back to screencap.
            pass

    # -- consumer API ------------------------------------------------------
    def read_latest(self):
        """Return the most recent frame as a PIL RGB Image, or None if no frame
        has arrived yet."""
        with self._lock:
            jpeg = self._latest_jpeg
            cache = self._decoded_cache
        if jpeg is None:
            return None
        # Skip re-decoding when polled faster than frames arrive (same bytes).
        if cache is not None and cache[0] is jpeg:
            return cache[1]
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        with self._lock:
            self._decoded_cache = (jpeg, img)
        return img

    def wait_for_frame(self, timeout=5.0):
        """Block until at least one frame is available; returns it or None."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            img = self.read_latest()
            if img is not None:
                return img
            time.sleep(0.02)
        return None

    @property
    def fps(self):
        if not self._started_at or self._frame_count == 0:
            return 0.0
        return self._frame_count / (time.time() - self._started_at)

    @property
    def frame_count(self):
        return self._frame_count

    @property
    def banner(self):
        return dict(self._banner)

    def is_alive(self):
        return self._reader is not None and self._reader.is_alive()

    def stop(self):
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        # Best-effort: drop the port forward and any lingering minicap on-device.
        self._run(["forward", "--remove", "tcp:%d" % self._port])
        self._run(["shell", "pkill", "-f", "minicap"])
