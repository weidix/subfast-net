from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=128)
def _sha256_for_stat(path: Path, size: int, modified_ns: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """Return a full-file content hash, cached for an unchanged path/stat tuple."""
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return _sha256_for_stat(resolved, stat.st_size, stat.st_mtime_ns)
