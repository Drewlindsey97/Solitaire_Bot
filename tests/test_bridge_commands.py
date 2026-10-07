import shlex
import unittest
from unittest.mock import patch

import bridge


class BridgeCommandTests(unittest.TestCase):
    def test_root_executes_android_command_without_adb_shell_prefix(self):
        with patch.object(bridge, "RUN_MODE", "LOCAL_ROOT"), \
                patch.object(bridge.subprocess, "run") as run:
            run.return_value.stdout = "ok\n"
            self.assertEqual(bridge.run_cmd(["shell", "input", "tap", "10", "20"]), "ok")
            self.assertEqual(run.call_args.args[0], ["su", "-c", "input tap 10 20"])

    def test_root_preserves_arguments_with_shell_metacharacters(self):
        args = ["am", "broadcast", "--es", "task_name", "a task; echo unwanted"]
        with patch.object(bridge, "RUN_MODE", "LOCAL_ROOT"), \
                patch.object(bridge.subprocess, "run") as run:
            bridge.run_cmd(["shell"] + args)
            self.assertEqual(shlex.split(run.call_args.args[0][2]), args)

    def test_pc_adb_retains_shell_prefix(self):
        with patch.object(bridge, "RUN_MODE", "PC_ADB"), \
                patch.object(bridge.subprocess, "run") as run:
            bridge.run_cmd(["shell", "input", "tap", "10", "20"])
            self.assertEqual(run.call_args.args[0], ["adb", "shell", "input", "tap", "10", "20"])
