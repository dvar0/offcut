import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import offcut_launch


class LauncherTests(unittest.TestCase):
    def test_saved_runtime_interpreter_and_explicit_override(self):
        with tempfile.TemporaryDirectory(prefix="offcut launch ") as directory:
            root = Path(directory)
            python = root / "ComfyUI/.venv/bin/python"
            python.parent.mkdir(parents=True)
            python.symlink_to(sys.executable)
            config = root / "settings.json"
            config.write_text(json.dumps({"comfy_root": str(root / "ComfyUI")}))
            with patch.dict(os.environ, {"OFFCUT_CONFIG": str(config), "OFFCUT_PYTHON": ""}):
                self.assertEqual(offcut_launch.python_executable(), str(python))
                with patch.dict(os.environ, {"OFFCUT_PYTHON": sys.executable}):
                    self.assertEqual(offcut_launch.python_executable(), sys.executable)

    def test_missing_comfy_environment_uses_bootstrap_python(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"OFFCUT_CONFIG": str(Path(directory) / "settings.json"), "OFFCUT_COMFY_ROOT": directory, "OFFCUT_PYTHON": ""}):
                self.assertEqual(offcut_launch.python_executable(), sys.executable)

    def test_shell_entrypoints_work_from_another_directory(self):
        with tempfile.TemporaryDirectory(prefix="offcut help ") as directory:
            env = {**os.environ, "OFFCUT_PYTHON": sys.executable, "OFFCUT_CONFIG": str(Path(directory) / "settings.json"),
                   "OFFCUT_DATABASE": str(Path(directory) / "test.sqlite3"), "OFFCUT_SECRETS": str(Path(directory) / "test.secrets.json")}
            for script in ("offcut", "offcut-cli"):
                with self.subTest(script=script):
                    result = subprocess.run([str(offcut_launch.offcut_cli.APP_ROOT / script), "--help"], cwd=directory, env=env, capture_output=True, text=True, timeout=20)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)

    def test_old_node_fails_before_dependency_install(self):
        with patch("offcut_launch.shutil.which", return_value="node"), patch("offcut_launch.subprocess.check_output", return_value="v20.19.0\n"), patch("offcut_launch.subprocess.run") as install:
            with self.assertRaisesRegex(RuntimeError, "22.19"):
                offcut_launch.ensure_agent()
            install.assert_not_called()

    def test_ui_preserves_allocator_override_and_arguments(self):
        with patch.object(sys, "argv", ["offcut_launch.py", "ui", "--port", "8765"]), patch.dict(os.environ, {"MALLOC_ARENA_MAX": "4"}), patch("offcut_launch.python_executable", return_value="python"), patch("offcut_launch.ensure_agent") as agent, patch("offcut_launch.os.execvp") as execute:
            self.assertEqual(offcut_launch.main(), 0)
            agent.assert_called_once()
            self.assertEqual(os.environ["MALLOC_ARENA_MAX"], "4")
            self.assertEqual(execute.call_args.args[1][-2:], ["--port", "8765"])
