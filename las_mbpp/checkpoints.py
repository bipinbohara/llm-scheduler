"""Identify the gate weights before evaluation starts."""

import hashlib
from pathlib import Path


def checkpoint_identity(path: Path | None, seed: int) -> dict:
    if path is None:
        return {"condition": "untrained_gate", "seed": seed, "load_outcome": "base_model_only"}
    if not path.is_file():
        raise FileNotFoundError(f"Gate checkpoint {path} is missing. Supply --gate_checkpoint_path or explicitly select --untrained_gate.")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        chunk = stream.read(1024 * 1024)
        if chunk.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"Gate checkpoint {path} is a Git LFS pointer. Fetch the LFS payload or obtain the trained state dict from the maintainer.")
        while chunk:
            digest.update(chunk)
            size += len(chunk)
            chunk = stream.read(1024 * 1024)
    return {
        "condition": "trained_gate", "path": str(path.resolve()), "sha256": digest.hexdigest(),
        "bytes": size,
    }
