"""Estado do código (revisão Git e alterações locais) registrado em artefatos derivados."""

import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def git_state(repository: Path = REPOSITORY_ROOT) -> dict[str, object]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, stderr=subprocess.DEVNULL, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_revision": None, "git_dirty": None}
    return {"git_revision": revision, "git_dirty": bool(status.strip())}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
