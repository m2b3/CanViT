"""Where a result came from: code revision, runtime, command line, SLURM job."""

import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

PACKAGE_DIR = Path(__file__).parent
PROJECT_DIR = PACKAGE_DIR.parent


def _git(*args: str) -> str | None:
    result = subprocess.run(["git", "-C", str(PACKAGE_DIR), *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def provenance(device: torch.device | None = None) -> dict[str, Any]:
    """Code revision of this package's checkout (None when installed from a wheel), versions, device, job."""
    commit = _git("rev-parse", "HEAD")
    return {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        # This project's files only (code, pyproject, lock): outputs written elsewhere in the repository,
        # such as the site's bundles, do not change the code that ran.
        "git_dirty": bool(_git("status", "--porcelain", "--", str(PROJECT_DIR))) if commit else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device) if device is not None else None,
        "cuda_device": torch.cuda.get_device_name(device) if device is not None and device.type == "cuda" else None,
        "command": " ".join(sys.argv),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
