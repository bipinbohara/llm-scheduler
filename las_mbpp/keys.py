from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from las_mbpp.config import PROJECT_ROOT


def _candidate_env_paths(path: str | Path | None) -> Iterable[Path]:
    if path is None:
        yield PROJECT_ROOT / ".env"
        return
    env_path = Path(path)
    if env_path.is_absolute():
        yield env_path
    else:
        yield PROJECT_ROOT / env_path


def load_env_file(path: str | Path | None = None, *, override: bool = False) -> Path | None:
    """Load KEY=VALUE pairs from a dotenv-style file if it exists."""

    for env_path in _candidate_env_paths(path):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            if override or key not in os.environ:
                os.environ[key] = value
        return env_path
    return None
