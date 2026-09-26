"""Reproducibility metadata for eval sidecars.

Omits identity-revealing fields (hostname, cwd, absolute paths in cmdline) so
sidecars are shippable in anonymized submission bundles. Don't add anything
that names a user, host, or path outside the repo tree.
"""

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _git(*args: str) -> str:
    # OSError covers missing/non-executable git binary and subprocess pipe failures;
    # CalledProcessError covers non-zero exit (e.g. not in a git repo).
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return ""


def git_commit() -> str:
    return _git("rev-parse", "HEAD") or "unknown"


def git_branch() -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown"


def git_dirty() -> bool:
    return bool(_git("status", "--porcelain"))


def _safe_cmdline() -> str:
    """argv[0] rendered repo-root-relative (or basename if outside any repo); argv[1:] verbatim."""
    if not sys.argv:
        return ""
    head = Path(sys.argv[0])
    repo_root = _git("rev-parse", "--show-toplevel")
    if repo_root:
        try:
            head = head.resolve().relative_to(repo_root)
        except ValueError:
            head = Path(head.name)
    else:
        head = Path(head.name)
    return " ".join([str(head), *sys.argv[1:]])


def provenance() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "git_branch": git_branch(),
        "git_dirty": git_dirty(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cmdline": _safe_cmdline(),
    }


def device_info(device: torch.device | None) -> dict[str, Any]:
    if device is None:
        return {}
    info: dict[str, Any] = {"device": str(device)}
    if device.type == "cuda":
        info["cuda_device_name"] = torch.cuda.get_device_name(device)
    return info
