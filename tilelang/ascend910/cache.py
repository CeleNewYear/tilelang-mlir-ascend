"""Content-addressed cache for the Ascend 910 (A2) backend.

The cache key covers every input that can change the compiled artifact: the
MLIR text, pass configuration, mode (expert/developer), compiler executable and
its version banner.  The on-disk layout mirrors the artifact stages instead of
mixing them in one directory, which makes partial invalidation and debugging
straightforward.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .bishengir import find_bishengir_compiler

CACHE_ENV = "TILELANG_CACHE_DIR"
DEFAULT_CACHE_DIR = Path.home() / ".tilelang" / "cache" / "ascend910"


def _cache_root() -> Path:
    return Path(os.environ.get(CACHE_ENV, DEFAULT_CACHE_DIR)).expanduser() / "ascend910"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _compiler_stamp() -> str:
    compiler = find_bishengir_compiler()
    try:
        st = os.stat(compiler)
        return f"{compiler}:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return compiler


def make_key(
    mlir: str,
    *,
    pass_configs: Mapping[str, Any] | None = None,
    developer: bool = False,
) -> str:
    payload = {
        "format": 1,
        "mlir": _sha256_bytes(mlir.encode()),
        "pass_configs": dict(pass_configs or {}),
        "developer": bool(developer),
        "compiler": _compiler_stamp(),
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True).encode())


@dataclass(frozen=True)
class CacheEntry:
    key: str
    root: Path
    mlir_path: Path
    object_path: Path
    shared_object_path: Path
    metadata_path: Path

    @property
    def ready(self) -> bool:
        return self.object_path.is_file() and self.metadata_path.is_file()


def entry(key: str) -> CacheEntry:
    root = _cache_root() / key
    return CacheEntry(
        key=key,
        root=root,
        mlir_path=root / "kernel.npuir",
        object_path=root / "kernel.o",
        shared_object_path=root / "libkernel.so",
        metadata_path=root / "metadata.json",
    )


def load(
    mlir: str,
    *,
    pass_configs: Mapping[str, Any] | None = None,
    developer: bool = False,
) -> CacheEntry | None:
    key = make_key(mlir, pass_configs=pass_configs, developer=developer)
    e = entry(key)
    return e if e.ready else None


def save(
    key: str,
    mlir: str,
    object_path: str | os.PathLike[str],
    shared_object_path: str | os.PathLike[str] | None,
    *,
    pass_configs: Mapping[str, Any] | None = None,
    developer: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> CacheEntry:
    e = entry(key)
    e.root.mkdir(parents=True, exist_ok=True)
    e.mlir_path.write_text(mlir)
    shutil.copy2(object_path, e.object_path)
    if shared_object_path and Path(shared_object_path).is_file():
        shutil.copy2(shared_object_path, e.shared_object_path)
    metadata = {
        "format": 1,
        "key": key,
        "pass_configs": dict(pass_configs or {}),
        "developer": bool(developer),
        "compiler": _compiler_stamp(),
        **(dict(extra or {})),
    }
    e.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return e


def clear() -> None:
    root = _cache_root()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
