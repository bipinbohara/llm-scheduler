"""Durable JSON Lines records and conservative tail recovery."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4
from weakref import WeakSet


_WRITER_LOCKS = WeakSet()


def _close_inherited_locks():
    # Grading forks children that can outlive an interrupted experiment process.
    for stream in list(_WRITER_LOCKS):
        stream.close()
    _WRITER_LOCKS.clear()


os.register_at_fork(after_in_child=_close_inherited_locks)


def open_writer_lock(path):
    stream = path.open("a")
    _WRITER_LOCKS.add(stream)
    return stream


def json_line(record):
    return (json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def append(path: Path, record):
    with path.open("ab") as stream:
        stream.write(json_line(record))
        stream.flush()
        os.fsync(stream.fileno())


def recover(path: Path, *, repair=False, keep=True):
    """Read newline-terminated JSON objects. Reject corrupt interior rows."""
    if not path.exists():
        return [], 0
    rows = []
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        while True:
            offset = stream.tell()
            if offset >= size:
                return rows, 0
            line = stream.readline(size - offset)
            if not line:
                return rows, 0
            try:
                row = json.loads(line)
                if not line.endswith(b"\n") or not isinstance(row, dict):
                    raise ValueError("incomplete journal row")
            except (UnicodeDecodeError, ValueError) as exc:
                if stream.tell() < size:
                    raise ValueError(f"Interior JSONL corruption in {path} at byte {offset}") from exc
                if repair:
                    backup = path.with_name(f"{path.name}.tail-{uuid4().hex}.bak")
                    with backup.open("xb") as saved:
                        saved.write(line)
                        saved.flush()
                        os.fsync(saved.fileno())
                    sync_directory(path.parent)
                    with path.open("r+b") as journal:
                        journal.truncate(offset)
                        os.fsync(journal.fileno())
                return rows, len(line)
            if keep:
                rows.append(row)
