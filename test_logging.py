#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from logcat_monitor import LogcatMonitor
from session_logger import SessionLogger


class SessionLoggerTests(unittest.TestCase):
    def test_writes_jsonl_event_and_serializes_tuples(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "session.jsonl"
            logger = SessionLogger(path, session_id="test-session")
            logger.event("move", move=("col_to_col", 0, 1, ("Q", "H")))
            logger.close()

            record = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(record["session_id"], "test-session")
            self.assertEqual(record["event"], "move")
            self.assertEqual(record["move"], ["col_to_col", 0, 1, ["Q", "H"]])


class LogcatMonitorTests(unittest.TestCase):
    def test_builds_pc_adb_command_with_pid_filters(self):
        monitor = LogcatMonitor("capture.log", run_mode="PC_ADB")
        command = monitor.build_command(["123", "456"])
        self.assertEqual(
            command,
            ["adb", "logcat", "-v", "threadtime", "--pid=123", "--pid=456"],
        )

    def test_builds_local_ladb_command(self):
        monitor = LogcatMonitor("capture.log", run_mode="LOCAL_LADB")
        command = monitor.build_command([])
        self.assertEqual(
            command,
            ["adb", "-s", "localhost:5555", "logcat", "-v", "threadtime"],
        )

    def test_resolves_package_pid(self):
        monitor = LogcatMonitor(
            "capture.log",
            run_mode="PC_ADB",
            package="com.example.game",
        )
        completed = type(
            "Completed",
            (),
            {"returncode": 0, "stdout": "1234 5678\n", "stderr": ""},
        )()
        with patch.object(monitor, "_run_quiet", return_value=completed):
            self.assertEqual(monitor._resolve_pids(), ["1234", "5678"])


if __name__ == "__main__":
    unittest.main()
