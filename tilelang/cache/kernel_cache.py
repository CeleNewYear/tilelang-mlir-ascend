# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The cache utils with class and database persistence - KernelCache Class

Provides process-safe, atomic disk caching for compiled NPU/GPU kernels.
Key properties:

* **Atomic writes** – files are written to a temporary staging directory,
  then atomically renamed into place.  No other process ever sees a partial
  cache entry.
* **Concurrent-write safety** – if two processes compile the same kernel
  simultaneously, one wins the ``os.rename`` race and the loser silently
  accepts the winner's complete result.
* **Human-readable paths** – the cache directory tree includes the kernel's
  ``global_symbol`` so users can locate entries manually.
* **Library-stamp invalidation** – the cache key includes a content hash of
  ``libtilelang.so``, so rebuilding the library automatically invalidates
  stale entries.
"""

from __future__ import annotations

import functools
import json
import logging
import errno
import os
import shutil
import threading
import uuid
import sys
import platform
from hashlib import sha256
from pathlib import Path
from typing import Callable, List, Literal, Union, Optional

import cloudpickle
from tvm.target import Target
from tvm.tir import PrimFunc

from tilelang.jit import JITKernel
from tilelang.jit.jit_npu import JitKernel_NPU
from tilelang.engine.param import KernelParam
from tilelang.utils.language import get_prim_func_name
from tilelang.utils.npu_utils import compute_sha256_hash
from tilelang import env
from tilelang.version import __version__

logger = logging.getLogger(__name__)

# ── file name constants ──────────────────────────────────────────────
# GPU artefacts
KERNEL_PATH = "kernel.cu"
WRAPPED_KERNEL_PATH = "wrapped_kernel.cu"
KERNEL_LIB_PATH = "kernel_lib.so"
PARAMS_PATH = "params.pkl"

# NPU artefacts
AUTOTUNE_KERNEL_MLIR_PATH = "kernel.mlir"
AUTOTUNE_WRAPPED_KERNEL_PATH = "wrapped_kernel.o"
AUTOTUNE_SO_LAUNCHER_PATH = "main.so"
AUTOTUNE_METADATA_PATH = "metadata.pkl"

# Tuning metadata
AUTOTUNE_SUBDIR = "autotune"
AUTOTUNE_BEST_CONFIG_PATH = "best_config.json"
AUTOTUNE_FUNCTION_PATH = "function.pkl"
AUTOTUNE_LATENCY_PATH = "latency.json"

# Directory names under the namespace root
CACHE_ROOT_DIR = "kernels"
STAGING_ROOT_DIR = ".staging"


# ══════════════════════════════════════════════════════════════════════
# KernelCache
# ══════════════════════════════════════════════════════════════════════

class KernelCache:
    """Process-safe, atomic disk cache for compiled TileLang kernels.

    Directory layout::

        <TILELANG_CACHE_DIR>/
            <namespace>/              ← version + platform
                kernels/              ← committed cache entries
                    <kernel_name>/
                        <hash>/
                            kernel.mlir
                            wrapped_kernel.o
                            main.so
                            metadata.pkl
                            …
                .staging/             ← in-progress writes (cleaned on startup)
                    <hash>_<pid>_<uuid>/
    """

    _instance = None
    _lock = threading.Lock()
    _memory_cache: dict[str, JITKernel] = {}
    _staging_cleanup_lock = threading.Lock()
    _last_cleaned_staging_root: Optional[str] = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    cls._create_dirs()
                    instance._memory_cache = {}
                    cls._instance = instance
        return cls._instance

    # ── directory helpers ────────────────────────────────────────────

    @staticmethod
    def _create_dirs():
        """Ensure all cache directory trees exist; clean stale staging."""
        os.makedirs(env.TILELANG_CACHE_DIR, exist_ok=True)
        os.makedirs(env.TILELANG_TMP_DIR, exist_ok=True)
        os.makedirs(KernelCache._get_namespace_root(), exist_ok=True)
        os.makedirs(KernelCache._get_cache_root(), exist_ok=True)
        os.makedirs(KernelCache._get_staging_root(), exist_ok=True)

        staging_root = KernelCache._get_staging_root()
        with KernelCache._staging_cleanup_lock:
            if KernelCache._last_cleaned_staging_root != staging_root:
                KernelCache._cleanup_stale_staging_dirs()
                KernelCache._last_cleaned_staging_root = staging_root

    @staticmethod
    def _get_namespace_root() -> str:
        return os.path.join(env.TILELANG_CACHE_DIR, KernelCache._get_cacheup
    @staticmethod
    def _get_staging_root() -> str:
        return os.path.join(KernelCache._get_namespace_root(), STAGING_ROOT_DIR)

    # ── namespace / stamp ────────────────────────────────────────────

    @staticmethod
    @functools.cache
    def _get_tilelang_lib_stamp() -> Optional[str]:
        """SHA-256 content hash of ``libtilelang.so`` for cache invalidation."""
        import importlib

        lib_dirs: list[str] = []
        try:
            env_mod = importlib.import_module("tilelang.env")
            lib_dirs.extend(getattr(env_mod, "TL_LIBS", []) or [])
        except Exception:
            pass

        if sys.platform == "win32":
            lib_names = ["tilelang.dll", "libtilelang.dll"]
        elif sys.platform == "darwin":
            lib_names = ["libtilelang.dylib", "libtilelang.so"]
        else:
            lib_names = ["libtilelang.so"]

        for lib_dir in lib_dirs:
            for name in lib_names:
                path = os.path.join(lib_dir, name)
                if os.path.exists(path):
                    file_hash = sha256()
                    with open(path, "rb") as f:
                        for chunk in iter(lambda: f.read(1 << 20), b""):
                            file_hash.update(chunk)
                    return f"{name}:{file_hash.hexdigest()}"
        return None

    @staticmethod
    @functools.cache
    def _get_base_key() -> dict:
        base = {"version": __version__, "platform": platform.machine()}
        lib_stamp = KernelCache._get_tilelang_lib_stamp()
        if lib_stamp:
            base["tilelang_lib"] = lib_stamp
        return base

    @staticmethod
    def _sanitize_path_component(component: str) -> str:
        sanitized = "".join(
            ch if ch.isalnum() or ch in "._-" else "_" for ch in component
        )
        sanitized = sanitized.strip("._-")
        return sanitized or "unknown"

    @staticmethod
    def _format_version_namespace(version: str) -> str:
        public, sep, local = version.partition("+")
        public = KernelCache._sanitize_path_component(public)
        if not sep:
            return public
        local = "".join(ch if ch.isalnum() else "_" for ch in local).strip("_")
        return f"{public}_{local}" if local else public

    @staticmethod
    @functools.cache
    def _get_cache_namespace() -> str:
        base_key = KernelCache._get_base_key()
        version = KernelCache._format_version_namespace(
            str(base_key.get("version", "unknown"))
        )
        platform_name = KernelCache._sanitize_path_component(
            str(base_key.get("platform", "unknown"))
        )
        return f"{version}-{platform_name}"

    # ── stale staging cleanup ────────────────────────────────────────

    @staticmethod
    def _cleanup_stale_staging_dirs(max_age_seconds: int = 3600):
        """Remove staging directories left by crashed processes."""
        import time

        try:
            now = time.time()
            staging_root = KernelCache._get_staging_root()
            if not os.path.isdir(staging_root):
                return
            for entry in os.scandir(staging_root):
                if entry.is_dir(follow_symlinks=False):
                    try:
                        if now - entry.stat().st_mtime > max_age_seconds:
                            shutil.rmtree(entry.path, ignore_errors=True)
                    except OSError:
                        pass
        except OSError:
            pass

    # ── file-level atomic helpers ────────────────────────────────────

    @staticmethod
    def _load_binary(path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    @staticmethod
    def _safe_write_file(path: str, mode: str, operation: Callable):
        """Write *path* atomically via a temp file + ``os.replace``."""
        tmp_path = os.path.join(
            env.TILELANG_TMP_DIR, f"{os.getpid()}_{uuid.uuid4().hex}"
        )
        with open(tmp_path, mode) as tmp_file:
            operation(tmp_file)
        os.replace(tmp_path, path)

    @staticmethod
    def _is_rename_collision(exc: OSError) -> bool:
        return exc.errno in {errno.EEXIST, errno.ENOTEMPTY}

    def _is_complete_cache_dir(self, cache_path: str) -> bool:
        """Return True if *cache_path* contains all required artefacts."""
        if not os.path.isdir(cache_path):
            return False
        required = [
            os.path.join(cache_path, AUTOTUNE_METADATA_PATH),
            os.path.join(cache_path, AUTOTUNE_SO_LAUNCHER_PATH),
            os.path.join(cache_path, AUTOTUNE_WRAPPED_KERNEL_PATH),
        ]
        return all(os.path.exists(f) for f in required)

    def _remove_incomplete_cache_dir(self, cache_path: str) -> bool:
        if not os.path.isdir(cache_path) or self._is_complete_cache_dir(cache_path):
            return False
        shutil.rmtree(cache_path, ignore_errors=True)
        return True

    # ── key generation ───────────────────────────────────────────────

    def _generate_compile_key(
        self,
        func,
        out_idx,
        target,
        target_host,
        execution_backend,
        verbose,
        pass_configs,
    ) -> str:
        """Stable hash key for a ``compile()`` call targeting NPU IR."""
        func_binary = cloudpickle.dumps(func.script())
        key_data = {
            "func": compute_sha256_hash(func_binary),
            "out_idx": (
                list(out_idx)
                if isinstance(out_idx, (list, tuple))
                else ([out_idx] if out_idx is not None else [])
            ),
            "target": str(target),
            "target_host": str(target_host) if target_host else None,
            "execution_backend": execution_backend,
            "verbose": verbose,
            "pass_configs": (
                json.dumps(pass_configs, sort_keys=True) if pass_configs else None
            ),
            **self._get_base_key(),
        }
        return compute_sha256_hash(json.dumps(key_data, sort_keys=True))

    # ── path helpers ─────────────────────────────────────────────────

    @staticmethod
    def _get_named_cache_path(key: str, kernel_name: str) -> str:
        """Return ``<cache_root>/<kernel_name>/<key>`` for user-facing cache."""
        safe_name = KernelCache._sanitize_path_component(kernel_name)
        return os.path.join(KernelCache._get_cache_root(), safe_name, key)

    @staticmethod
    def _get_flat_cache_path(key: str) -> str:
        """Return ``<cache_root>/<key>`` for autotune / flat paths."""
        return os.path.join(KernelCache._get_cache_root(), key)

    # ── public API ───────────────────────────────────────────────────

    def cached(
        self,
        func: PrimFunc = None,
        out_idx: List[int] = None,
        workspace_idx: List[int] = None,
        *args,
        target: Union[str, Target] = "auto",
        target_host: Union[str, Target] = None,
        platform: Literal["A2", "A3", "A5"] = "A3",
        execution_backend: Optional[Literal["dlpack", "ctypes", "cython"]] = "cython",
        verbose: Optional[bool] = False,
        pass_configs: Optional[dict] = None,
    ) -> JITKernel:
        """GPU-target compile with optional caching (passthrough for npuir branch)."""
        from tilelang.jit import JITKernel as _JITKernel

        if not env.is_cache_enabled():
            return _JITKernel(
                func,
                out_idx=out_idx,
                workspace_idx=workspace_idx,
                execution_backend=execution_backend,
                target=target,
                target_host=target_host,
                platform=platform,
                verbose=verbose,
                pass_configs=pass_configs,
            )

        # GPU cache: simple key → memory → disk → compile flow
        func_binary = cloudpickle.dumps(func.script())
        key_data = {
            "func": compute_sha256_hash(func_binary),
            "out_idx": (
                tuple(out_idx) if isinstance(out_idx, (list, tuple)) else [out_idx]
            ),
            "workspace_idx": (
                tuple(workspace_idx)
                if isinstance(workspace_idx, (list, tuple))
                else [workspace_idx]
            ),
            "args_repr": tuple(repr(arg) for arg in args),
            "target": str(target),
            "target_host": str(target_host) if target_host else None,
            "platform": str(platform),
            "execution_backend": execution_backend,
            "pass_configs": pass_configs,
            **self._get_base_key(),
        }
        key = compute_sha256_hash(json.dumps(key_data, sort_keys=True))

        with self._lock:
            if key in self._memory_cache:
                return self._memory_cache[key]

            # No disk cache for GPU in npuir branch — compile directly
            pass

        kernel = _JITKernel(
            func,
            out_idx=out_idx,
            workspace_idx=workspace_idx,
            execution_backend=execution_backend,
            target=target,
            target_host=target_host,
            platform=platform,
            verbose=verbose,
            pass_configs=pass_configs,
        )
        self._memory_cache[key] = kernel
        return kernel

    def cached_npu(
        self,
        func,
        out_idx=None,
        execution_backend="cython",
        target="npuir",
        target_host=None,
        verbose=False,
        pass_configs=None,
    ):
        """Compile *func* for ``npuir`` target with process-safe caching.

        Lookup order: in-memory → disk cache → fresh compile.
        Uses human-readable paths: ``kernels/<kernel_name>/<key>/``.
        """
        from tilelang.jit.jit_npu import compiler_npu

        key = self._generate_compile_key(
            func=func,
            out_idx=out_idx,
            target=target,
            target_host=target_host,
            execution_backend=execution_backend,
            verbose=verbose,
            pass_configs=pass_configs,
        )
        kernel_name = get_prim_func_name(func, "unknown")

        if env.is_cache_enabled():
            # 1. in-memory cache
            mem_hit = self._memory_cache.get(key)
            if mem_hit is not None:
                if verbose:
                    logger.debug(
                        f"cached_npu(): memory cache hit for {kernel_name} "
                        f"(key {key[:8]}…)"
                    )
                return mem_hit

            # 2. disk cache (named path)
            cache_path = self._get_named_cache_path(key, kernel_name)
            disk_hit = self._load_npu_kernel_from_cache_path(
                cache_path, func=func, out_idx=out_idx, verbose=verbose
            )
            if disk_hit is not None:
                if verbose:
                    logger.debug(
                        f"cached_npu(): disk cache hit for {kernel_name} "
                        f"(key {key[:8]}…)"
                    )
                self._memory_cache[key] = disk_hit
                return disk_hit

            if verbose:
                logger.debug(
                    f"cached_npu(): cache miss for {kernel_name}, compiling…"
                )

        # 3. fresh compile
        kernel = compiler_npu().compile(func, out_idx)

        if env.is_cache_enabled():
            cache_path = self._get_named_cache_path(key, kernel_name)
            self._save_npu_kernel_to_cache_path(cache_path, kernel, verbose=verbose)
            self._memory_cache[key] = kernel

        return kernel

    def save_compile_result(
        self,
        key: str,
        kernel: JitKernel_NPU,
        kernel_name: str = "unknown",
        verbose: bool = False,
    ) -> None:
        """Persist a ``JitKernel_NPU`` to disk using named paths."""
        cache_path = self._get_named_cache_path(key, kernel_name)
        self._save_npu_kernel_to_cache_path(cache_path, kernel, verbose)

    def load_compile_result(
        self,
        key: str,
        func,
        out_idx,
        kernel_name: str = "unknown",
        verbose: bool = False,
    ) -> Optional[JitKernel_NPU]:
        """Load an NPU kernel from named path.  Returns ``None`` on miss."""
        cache_path = self._get_named_cache_path(key, kernel_name)
        return self._load_npu_kernel_from_cache_path(
            cache_path, func=func, out_idx=out_idx, verbose=verbose
        )

    # ── disk persistence (NPU) ───────────────────────────────────────

    def _save_npu_kernel_to_cache_path(
        self,
        cache_path: str,
        kernel: JitKernel_NPU,
        verbose: bool = False,
    ) -> None:
        """Atomically save an NPU kernel using a staging directory.

        All artefacts are written into a unique staging directory under
        ``.staging/``.  Once every file is in place, the whole directory
        is atomically renamed into *cache_path*.
        """
        KernelCache._create_dirs()

        # Another process already wrote a complete entry.
        if self._is_complete_cache_dir(cache_path):
            return

        # Create staging directory with a unique name
        staging_path = os.path.join(
            self._get_staging_root(),
            f"{os.path.basename(cache_path)}_{os.getpid()}_{uuid.uuid4().hex[:8]}",
        )
        os.makedirs(staging_path)

        try:
            # Write MLIR content
            if kernel.mlir_content is not None:
                self._try_save(
                    "kernel MLIR",
                    lambda: (Path(staging_path) / AUTOTUNE_KERNEL_MLIR_PATH).write_text(
                        kernel.mlir_content
                    ),
                )

            # Write wrapped kernel object
            self._try_save(
                "wrapped kernel",
                lambda: self._safe_write_file(
                    os.path.join(staging_path, AUTOTUNE_WRAPPED_KERNEL_PATH),
                    "wb",
                    lambda f: f.write(kernel.get_kernel_source()),
                ),
            )

            # Copy main.so
            self._try_save(
                "main.so",
                lambda: shutil.copy(
                    kernel.so_launcher_path,
                    os.path.join(staging_path, AUTOTUNE_SO_LAUNCHER_PATH),
                ),
            )

            # Write metadata
            metadata = {
                "symbolic": kernel.symbolic,
                "params": kernel.params,
                "param_info": kernel.param_info,
                "out_idx": kernel.out_idx,
                "signature": kernel.signature,
                "primfunc": kernel.prim_func,
                "mlir_content": kernel.mlir_content,
                "shared": kernel.utils_shared,
                "kernel_name": kernel.kernel_name,
                "gridfunc": kernel.gridfunc,
                "mix_mode": kernel.mix_mode,
                "name": kernel.utils_name,
                "tensor_kinds": kernel.tensor_kinds,
                "kernel_src": kernel.utils_kernel_src,
            }
            self._try_save(
                "metadata",
                lambda: self._safe_write_file(
                    os.path.join(staging_path, AUTOTUNE_METADATA_PATH),
                    "wb",
                    lambda f: cloudpickle.dump(metadata, f),
                ),
            )

            # Verify staging is complete
            if not self._is_complete_cache_dir(staging_path):
                missing = [
                    f
                    for f in [
                        os.path.join(staging_path, AUTOTUNE_METADATA_PATH),
                        os.path.join(staging_path, AUTOTUNE_SO_LAUNCHER_PATH),
                        os.path.join(staging_path, AUTOTUNE_WRAPPED_KERNEL_PATH),
                    ]
                    if not os.path.exists(f)
                ]
                raise RuntimeError(
                    f"Incomplete staging directory missing: {missing}"
                )

            # Remove any previously incomplete entry at the target
            self._remove_incomplete_cache_dir(cache_path)

            # Ensure the parent directory exists
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            # Atomic rename — makes the complete directory visible in one step.
            try:
                os.rename(staging_path, cache_path)
            except OSError as exc:
                if not self._is_rename_collision(exc):
                    raise
                # Another process won the race; discard our staging.
                shutil.rmtree(staging_path, ignore_errors=True)

            if verbose:
                logger.debug(f"NPU kernel saved to {cache_path}")

        except Exception:
            shutil.rmtree(staging_path, ignore_errors=True)
            logger.exception("Error during atomic NPU cache save")

    def _load_npu_kernel_from_cache_path(
        self,
        cache_path: str,
        func: Callable = None,
        out_idx: Optional[List[int]] = None,
        verbose: bool = False,
    ) -> Optional[JitKernel_NPU]:
        """Load an NPU kernel from *cache_path*.  Returns ``None`` on any issue."""

        if not self._is_complete_cache_dir(cache_path):
            return None

        kernel_source: Optional[str] = None
        kernel_global_source: Optional[bytes] = None
        metadata: Optional[dict] = None

        try:
            kernel_source = (Path(cache_path) / AUTOTUNE_KERNEL_MLIR_PATH).read_text()
        except Exception as exc:
            logger.error(f"Error loading kernel MLIR: {exc}")

        try:
            kernel_global_source = (
                Path(cache_path) / AUTOTUNE_WRAPPED_KERNEL_PATH
            ).read_bytes()
        except Exception as exc:
            logger.error(f"Error loading wrapped kernel: {exc}")

        try:
            metadata = cloudpickle.loads(
                (Path(cache_path) / AUTOTUNE_METADATA_PATH).read_bytes()
            )
        except Exception as exc:
            logger.error(f"Error loading metadata: {exc}")

        if not (kernel_global_source and metadata):
            logger.warning(
                f"Incomplete NPU kernel artefacts at {cache_path} — "
                f"will recompile."
            )
            return None

        # Pre-populate so_launcher_path in metadata so that
        # JitKernel_NPU.__init__ → _launch() uses the correct path
        # (some versions of from_database set this after __init__).
        metadata["so_launcher_path"] = str(Path(cache_path) / AUTOTUNE_SO_LAUNCHER_PATH)

        return JitKernel_NPU.from_database(
            mod=func,
            kernel_source=kernel_source,
            kernel_launcher_path=str(Path(cache_path) / AUTOTUNE_SO_LAUNCHER_PATH),
            kernel_utils_path=None,
            metadata=metadata,
            out_idx=out_idx,
        )

    # ── autotune persistence (wraps NPU save/load) ───────────────────

    def save_autotune_result(
        self,
        key: str,
        result,
        verbose: bool = False,
    ) -> None:
        """Persist an autotune result with tuning metadata.

        Uses flat key-only paths: ``kernels/<key>/`` (not named subpaths).
        """
        from tilelang.autotuner.param import AutotuneResult

        cache_path = Path(self._get_flat_cache_path(key))
        autotune_path = cache_path / AUTOTUNE_SUBDIR
        cache_path.mkdir(parents=True, exist_ok=True)
        autotune_path.mkdir(exist_ok=True)

        # NPU kernel artefacts (saved to flat path)
        self._save_npu_kernel_to_cache_path(str(cache_path), result.kernel, verbose)

        # Tuning metadata
        self._try_save(
            "best config",
            lambda: _write_json(
                autotune_path / AUTOTUNE_BEST_CONFIG_PATH, result.config
            ),
        )
        self._try_save(
            "function",
            lambda: (autotune_path / AUTOTUNE_FUNCTION_PATH).write_bytes(
                cloudpickle.dumps(result.func)
            ),
        )
        self._try_save(
            "latency",
            lambda: _write_json(
                autotune_path / AUTOTUNE_LATENCY_PATH,
                {"latency": result.latency, "ref_latency": result.ref_latency},
            ),
        )

    def load_autotune_result(
        self,
        key: str,
        out_idx: Optional[List[int]],
        kernel_name: str = "unknown",
        verbose: bool = False,
    ):
        """Load a previously saved autotune result.  Returns ``None`` on miss.

        Uses flat key-only paths: ``kernels/<key>/`` (consistent with save).
        """
        from tilelang.autotuner.param import AutotuneResult

        cache_path = Path(self._get_flat_cache_path(key))
        autotune_path = cache_path / AUTOTUNE_SUBDIR

        if not autotune_path.exists():
            return None

        try:
            config = _read_json(autotune_path / AUTOTUNE_BEST_CONFIG_PATH)
            func = cloudpickle.loads(
                (autotune_path / AUTOTUNE_FUNCTION_PATH).read_bytes()
            )
            latency_data = _read_json(autotune_path / AUTOTUNE_LATENCY_PATH)
        except Exception as exc:
            logger.error(f"Failed to load autotune metadata: {exc}")
            return None

        latency = latency_data["latency"]
        ref_latency = latency_data["ref_latency"]

        kernel = self._load_npu_kernel_from_cache_path(
            str(cache_path), func=func, out_idx=out_idx, verbose=verbose
        )
        if kernel is None:
            return None

        kernel.update_tuner_result(
            config=config, latency=latency, ref_latency=ref_latency
        )
        return AutotuneResult(
            config=config,
            func=func,
            kernel=kernel,
            libcode=kernel.get_kernel_source(),
            latency=latency,
            ref_latency=ref_latency,
        )

    # ── cache directory management ───────────────────────────────────

    def get_cache_dir(self) -> Path:
        return Path(env.TILELANG_CACHE_DIR)

    def set_cache_dir(self, cache_dir: str):
        env.TILELANG_CACHE_DIR = cache_dir

    def clear_cache(self):
        """Clear both in-memory and disk cache for the current namespace."""
        with self._lock:
            self._memory_cache.clear()
            try:
                shutil.rmtree(self._get_cache_root(), ignore_errors=True)
                shutil.rmtree(self._get_staging_root(), ignore_errors=True)
                KernelCache._create_dirs()
            except Exception:
                logger.exception("Error clearing disk cache")

    # ── utilities ────────────────────────────────────────────────────

    def _try_save(self, label: str, fn: Callable) -> None:
        try:
            fn()
        except Exception as exc:
            logger.error(f"Error saving {label}: {exc}")


# ══════════════════════════════════════════════════════════════════════
# JSON helpers
# ══════════════════════════════════════════════════════════════════════

def _write_json(path: Path, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f)


def _read_json(path: Path):
    with open(path) as f:
        return json.load(f)
