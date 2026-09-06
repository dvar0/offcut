# SPDX-License-Identifier: GPL-3.0-only
"""Select the configured ComfyUI interpreter without importing inference or user data."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import offcut_cli


def python_executable() -> str:
    explicit = os.environ.get("OFFCUT_PYTHON")
    if explicit:
        return os.path.expanduser(explicit)
    root = Path(offcut_cli.load_settings()["comfy_root"]).expanduser()
    candidate = root / ".venv/bin/python"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return sys.executable


def ensure_agent() -> None:
    if not shutil.which("node"):
        raise RuntimeError("Node.js 22.19 or newer is required for the web app. See README.md.")
    version = subprocess.check_output(["node", "--version"], text=True).strip()
    if tuple(int(part) for part in version.lstrip("v").split(".")) < (22, 19, 0):
        raise RuntimeError(f"Node.js 22.19 or newer is required; found {version}.")
    agent = offcut_cli.APP_ROOT / "agent"
    packages = ("pi-agent-core", "pi-ai")
    if any(not (agent / "node_modules/@earendil-works" / name / "package.json").is_file() for name in packages):
        if not shutil.which("npm"):
            raise RuntimeError("npm is required to install the locked chat dependencies. See README.md.")
        subprocess.run(["npm", "--prefix", str(agent), "ci"], check=True)


def main() -> int:
    try:
        mode, *arguments = sys.argv[1:]
        if mode not in {"cli", "ui"}:
            raise ValueError("Launcher mode must be cli or ui")
        python = python_executable()
        if mode == "ui":
            if not any(arg in {"-h", "--help"} for arg in arguments):
                ensure_agent()
            # Must be set before the target interpreter's first allocation.
            os.environ.setdefault("MALLOC_ARENA_MAX", "2")
        script = "offcut_cli.py" if mode == "cli" else "offcut_server.py"
        os.execvp(python, [python, str(offcut_cli.APP_ROOT / script), *arguments])
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Offcut: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
