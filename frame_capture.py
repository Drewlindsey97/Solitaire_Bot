"""Capture fresh board frames continuously and check stability before reading."""
import numpy as np
import threading
import time
import os
import select
import struct
import subprocess
from dataclasses import dataclass
from PIL import Image


def board_motion_fraction(previous, current):
    if previous.size != current.size:
        return 1.0
    # Exclude the changing timer, score, and bottom controls. Retain the
    # foundations, stock/waste, tableau, and overlays over those regions.
    before = np.asarray(previous.convert("RGB"))[300:1400:4, ::4].astype(np.int16)
    after = np.asarray(current.convert("RGB"))[300:1400:4, ::4].astype(np.int16)
    if before.size == 0:
        return 1.0
    changed = np.max(np.abs(after - before), axis=2) > 24
    return float(changed.mean())


def capture_settled_frame(capture, max_captures=4, tolerance=0.0005):
    """Return (frame, capture count, motion); moving frames never reach OCR.

    ADB capture itself separates samples, so no additional blind sleep is
    needed. An unsettled board returns None for the caller to retry.
    """
    previous = None
    motion = None
    for count in range(1, max_captures + 1):
        current = capture()
        if current is None:
            previous = None
            continue
        if current.size != (720, 1600):
            # Let the existing resolution guard stop unsupported layouts.
            return current, count, None
        if previous is not None:
            motion = board_motion_fraction(previous, current)
            if motion <= tolerance:
                return current, count, motion
        previous = current
    return None, max_captures, motion


@dataclass(frozen=True)
class Frame:
    image: object
    sequence: int
    started: float
    completed: float
    motion: float | None
    settled: bool


class FrameBuffer:
    """Keep only the newest frame; an older stable frame is never a fallback."""

    def __init__(self, tolerance=0.0005):
        self.tolerance = tolerance
        self.condition = threading.Condition()
        self.latest = None
        self.sequence = 0
        self.closed = False

    def publish(self, image, started, completed):
        with self.condition:
            self.sequence += 1
            previous = self.latest
            motion = None
            settled = False
            if image is not None:
                if image.size != (720, 1600):
                    settled = True  # Existing resolution guard handles this.
                elif previous is not None:
                    motion = board_motion_fraction(previous.image, image)
                    settled = motion <= self.tolerance
            self.latest = (Frame(image, self.sequence, started, completed,
                                 motion, settled) if image is not None else None)
            self.condition.notify_all()

    def wait(self, *, not_before=0.0, after_sequence=0, timeout=4.0, max_age=0.5):
        deadline = time.monotonic() + timeout
        with self.condition:
            while not self.closed:
                now = time.monotonic()
                frame = self.latest
                if (frame is not None and frame.settled
                        and frame.sequence > after_sequence
                        and frame.started >= not_before
                        and now - frame.completed <= max_age):
                    return frame
                remaining = deadline - now
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)
        return None

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


class BackgroundCapture:
    """Capture while OCR, planning, and gestures run; never queue old frames."""

    def __init__(self, capture):
        self.capture = capture
        self.buffer = FrameBuffer()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="board-capture", daemon=True)
        self.error = None

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                image = self.capture()
                started = getattr(self.capture, "last_started", started)
                self.error = None if image is not None else "capture returned no frame"
            except Exception as exc:
                image = None
                self.error = str(exc)
            completed = time.monotonic()
            if self.stop_event.is_set():
                break
            self.buffer.publish(image, started, completed)
            delay = (0.0 if getattr(self.capture, "continuous", False) else 0.02)
            self.stop_event.wait(delay if image is not None else 0.3)

    def stop(self):
        self.stop_event.set()
        self.buffer.close()
        close = getattr(self.capture, "close", None)
        if close is not None:
            close()
        self.thread.join(timeout=16.0)


def decode_stream_header(header):
    # Android FrameOutput: packet length, width, height, RGB stride, format.
    # https://android.googlesource.com/platform/frameworks/av/+/master/cmds/screenrecord/FrameOutput.cpp
    size, width, height, stride, pixel_format = struct.unpack("<5I", header)
    if (not 0 < width <= 4096 or not 0 < height <= 4096
            or pixel_format != 3 or stride != width * 3
            or size != 16 + stride * height):
        raise ValueError("invalid screenrecord frame header")
    return width, height, size - 16


class ScreenrecordCapture:
    """Persistent native-resolution RGB stream; no video compression or files."""

    def __init__(self, adb_prefix):
        self.command = list(adb_prefix) + ["exec-out", "screenrecord", "--output-format=frames", "-"]
        self.process = None
        self.last_started = 0.0
        self.previous_read_started = None
        self.closed = False

    def _read_exact(self, pipe, count, deadline):
        data = bytearray()
        while len(data) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([pipe], [], [], remaining)[0]:
                raise TimeoutError("screen stream stalled")
            chunk = os.read(pipe.fileno(), min(count - len(data), 262144))
            if not chunk:
                raise EOFError("screen stream disconnected")
            data.extend(chunk)
        return bytes(data)

    def __call__(self):
        if self.closed:
            return None
        if self.process is None:
            self.process = subprocess.Popen(self.command, stdout=subprocess.PIPE,
                                            stderr=subprocess.DEVNULL, bufsize=0,
                                            start_new_session=True)
            self.previous_read_started = None
        process = self.process
        read_started = time.monotonic()
        # Reserve one complete transfer interval for the frame in flight.
        # It prevents a frame buffered during a drag from passing the
        # post-gesture freshness barrier just because it arrived later.
        self.last_started = (self.previous_read_started
                             if self.previous_read_started is not None else read_started)
        self.previous_read_started = read_started
        try:
            deadline = read_started + 5.0
            header = self._read_exact(process.stdout, 20, deadline)
            width, height, size = decode_stream_header(header)
            pixels = self._read_exact(process.stdout, size, deadline)
            return Image.frombytes("RGB", (width, height), pixels)
        except (OSError, ValueError, EOFError, TimeoutError):
            self._terminate(process)
            if self.process is process:
                self.process = None
            raise

    @staticmethod
    def _terminate(process):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def close(self):
        self.closed = True
        if self.process is not None:
            self._terminate(self.process)


class VideoScreenrecordCapture(ScreenrecordCapture):
    """Native-size 20 Mbps H.264, continuously decoded without a frame queue."""
    continuous = True
    # Screenrecord's H.264 output has no device capture timestamps. Reserve
    # 250ms for encoding/transport/decoding before accepting a post-input frame.
    pipeline_margin = 0.25

    def __init__(self, adb_prefix):
        super().__init__(adb_prefix)
        self.command = list(adb_prefix) + [
            "exec-out", "screenrecord", "--output-format=h264",
            "--bit-rate", "20M", "--size", "720x1600", "-",
        ]
        self.decoder = None

    def __call__(self):
        if self.closed:
            return None
        if self.process is None:
            self.process = subprocess.Popen(self.command, stdout=subprocess.PIPE,
                                            stderr=subprocess.DEVNULL, start_new_session=True)
            try:
                self.decoder = subprocess.Popen([
                    "ffmpeg", "-loglevel", "error", "-flags", "low_delay",
                    "-probesize", "32", "-analyzeduration", "0", "-threads", "1",
                    "-f", "h264", "-i", "pipe:0", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-fps_mode", "passthrough", "pipe:1",
                ], stdin=self.process.stdout, stdout=subprocess.PIPE,
                   stderr=subprocess.DEVNULL, bufsize=0, start_new_session=True)
            except Exception:
                self._reset_video()
                raise
            self.process.stdout.close()
        self.last_started = time.monotonic() - self.pipeline_margin
        try:
            pixels = self._read_exact(self.decoder.stdout, 720*1600*3, time.monotonic()+5)
            return Image.frombytes("RGB", (720, 1600), pixels)
        except (OSError, ValueError, EOFError, TimeoutError):
            self._reset_video()
            raise

    def _reset_video(self):
        for process in (self.decoder, self.process):
            if process is not None:
                self._terminate(process)
        self.decoder = None
        self.process = None

    def close(self):
        self.closed = True
        self._reset_video()
