#!/usr/bin/env python3
"""Background Android logcat capture for Solvitaire."""

from __future__ import annotations

import re
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterable


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_logcat_path() -> Path:
    return Path("logs") / f"logcat_{timestamp_slug()}.log"


class LogcatMonitor:
    """
    Capture Android logcat in a background thread.

    When a package name is supplied, the monitor tries to resolve its current
    process ID and uses logcat's --pid option. If the process is not running,
    capture still starts without a PID restriction so the diagnostic session
    is not silently lost.
    """

    def __init__(
        self,
        output_path: str | Path,
        run_mode: str = "PC_ADB",
        package: str | None = None,
        include_patterns: Iterable[str] | None = None,
        clear_first: bool = False,
    ):
        self.output_path = Path(output_path)
        self.run_mode = run_mode
        self.package = package
        self.include_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in (include_patterns or [])
        ]
        self.clear_first = clear_first
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.error: str | None = None
        self.resolved_pids: list[str] = []
        self._stop_event = threading.Event()
        self._file = None

    def _adb_prefix(self) -> list[str]:
        if self.run_mode == "PC_ADB":
            return ["adb"]
        if self.run_mode == "LOCAL_LADB":
            return ["adb", "-s", "localhost:5555"]
        if self.run_mode == "LOCAL_ROOT":
            return ["su", "-c"]
        # HTTP_BRIDGE and INTENT_ONLY may still run on a desktop with ADB.
        return ["adb"]

    def _run_quiet(self, command: list[str]) -> subprocess.CompletedProcess[str] | None:
        try:
            if self.run_mode == "LOCAL_ROOT":
                command = ["su", "-c", " ".join(command[2:])]
            return subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    def _resolve_pids(self) -> list[str]:
        if not self.package:
            return []

        if self.run_mode == "LOCAL_ROOT":
            command = ["su", "-c", f"pidof {self.package}"]
        else:
            command = self._adb_prefix() + ["shell", "pidof", self.package]

        result = self._run_quiet(command)
        if result is None or result.returncode != 0:
            return []

        return [pid for pid in result.stdout.strip().split() if pid.isdigit()]

    def build_command(self, pids: Iterable[str] | None = None) -> list[str]:
        pids = list(pids or [])

        if self.run_mode == "LOCAL_ROOT":
            logcat_parts = ["logcat", "-v", "threadtime"]
            logcat_parts.extend(f"--pid={pid}" for pid in pids)
            return ["su", "-c", " ".join(logcat_parts)]

        command = self._adb_prefix() + ["logcat", "-v", "threadtime"]
        command.extend(f"--pid={pid}" for pid in pids)
        return command

    def _clear(self) -> None:
        if self.run_mode == "LOCAL_ROOT":
            command = ["su", "-c", "logcat -c"]
        else:
            command = self._adb_prefix() + ["logcat", "-c"]
        self._run_quiet(command)

    def _line_is_included(self, line: str) -> bool:
        if not self.include_patterns:
            return True
        return any(pattern.search(line) for pattern in self.include_patterns)

    def _reader_loop(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        assert self._file is not None

        try:
            for line in self.process.stdout:
                if self._stop_event.is_set():
                    break
                if self._line_is_included(line):
                    self._file.write(line)
                    self._file.flush()
        finally:
            self._file.flush()

    def start(self) -> bool:
        if self.process is not None:
            return True

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.clear_first:
            self._clear()

        self.resolved_pids = self._resolve_pids()
        command = self.build_command(self.resolved_pids)

        try:
            self._file = self.output_path.open("a", encoding="utf-8", buffering=1)
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except (FileNotFoundError, OSError) as exc:
            self.error = str(exc)
            if self._file is not None:
                self._file.close()
                self._file = None
            try:
                if self.output_path.exists() and self.output_path.stat().st_size == 0:
                    self.output_path.unlink()
            except OSError:
                pass
            self.process = None
            return False

        self.thread = threading.Thread(
            target=self._reader_loop,
            name="solvitaire-logcat",
            daemon=True,
        )
        self.thread.start()
        return True

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()

        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=timeout)

        if self.thread is not None:
            self.thread.join(timeout=timeout)

        if self._file is not None:
            self._file.close()
            self._file = None

        self.process = None
        self.thread = None

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def __enter__(self) -> "LogcatMonitor":
        if not self.start():
            raise RuntimeError(self.error or "Unable to start logcat")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()


def main() -> None:
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Capture Android logcat for Solvitaire diagnostics")
    parser.add_argument("--output", type=str, default=str(default_logcat_path()))
    parser.add_argument("--package", type=str, help="Android package to restrict by PID")
    parser.add_argument("--filter", action="append", default=[], help="Regex line filter; repeat as needed")
    parser.add_argument("--clear", action="store_true", help="Clear logcat before capture")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to capture; 0 runs until Ctrl+C")
    parser.add_argument(
        "--run-mode",
        choices=["PC_ADB", "LOCAL_LADB", "LOCAL_ROOT", "HTTP_BRIDGE", "INTENT_ONLY"],
        default="PC_ADB",
    )
    args = parser.parse_args()

    monitor = LogcatMonitor(
        output_path=args.output,
        run_mode=args.run_mode,
        package=args.package,
        include_patterns=args.filter,
        clear_first=args.clear,
    )

    if not monitor.start():
        raise SystemExit(f"Unable to start logcat: {monitor.error}")

    pids = ", ".join(monitor.resolved_pids) or "all processes"
    print(f"Capturing logcat to {monitor.output_path} ({pids}). Press Ctrl+C to stop.")

    try:
        if args.duration > 0:
            time.sleep(args.duration)
        else:
            while monitor.is_running:
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
        print(f"Saved logcat to {monitor.output_path}")


if __name__ == "__main__":
    main()
