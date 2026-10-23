"""bishengir-compile integration for Ascend 910 (A2).

This module is the A2 counterpart of ``tilelang.contrib.bisheng`` (Ascend 950).
It only knows how to turn the MLIR produced by the A2 codegen into a device
object.  Kernel construction, launch wrappers, and cache layout remain in their
own modules.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

_ENV_COMPILER = "TILELANG_NPU_COMPILER_PATH"
_ENV_OPT = "TILELANG_NPU_OPT_PATH"


@dataclass(frozen=True)
class BishengirCompileResult:
    """Paths and exact command line of one bishengir-compile invocation."""

    command: tuple[str, ...]
    mlir_path: Path
    output_base: Path
    object_path: Path
    shared_object_path: Path
    stdout: str
    stderr: str


def find_bishengir_compiler() -> str:
    """Locate ``bishengir-compile`` without relying on A5's ``bisheng``."""

    explicit = os.environ.get(_ENV_COMPILER)
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return str(path)
        raise FileNotFoundError(f"{_ENV_COMPILER}={explicit!r} is not a file")

    # Prefer the compiler shipped with the active CANN toolkit.  A separately
    # built AscendNPU-IR install may be first on PATH for its MLIR tools while
    # its bishengir-compile is not paired with the toolkit's hivmc.
    for env_name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        root = os.environ.get(env_name)
        if not root:
            continue
        for rel in ("tools/bishengir/bin/bishengir-compile", "bin/bishengir-compile"):
            candidate = Path(root) / rel
            if candidate.is_file():
                return str(candidate)

    found = shutil.which("bishengir-compile")
    if found:
        return found

    raise RuntimeError(
        "Cannot find bishengir-compile. Set TILELANG_NPU_COMPILER_PATH or put "
        "the CANN tools/bishengir/bin directory on PATH."
    )


def _bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(value)


def compiler_options(
    pass_configs: Mapping[str, object] | None = None,
    *,
    developer: bool = False,
) -> list[str]:
    """Return the exact option sequence used by the legacy A2 JIT.

    The order is part of the golden command line and must not be sorted.
    """

    cfg = pass_configs or {}
    options = [
        f"--enable-auto-multi-buffer={str(_bool(cfg.get('npuir.enable_auto_multi_buffer'), True)).lower()}",
        f"--disable-hivm-auto-inject-sync={str(_bool(cfg.get('npuir.disable_hivm_auto_inject_sync'), False)).lower()}",
        "--enable-triton-kernel-compile=true",
        "--enable-hivm-compile=true",
    ]
    if not developer:
        options.append("--disable-hivm-tensor-compile=true")
    return options


def compiler_command(
    compiler: str,
    mlir_path: str | os.PathLike[str],
    output_base: str | os.PathLike[str],
    pass_configs: Mapping[str, object] | None = None,
    *,
    developer: bool = False,
) -> list[str]:
    return [compiler, os.fspath(mlir_path), *compiler_options(pass_configs, developer=developer), "-o", os.fspath(output_base)]


def compile_mlir(
    mlir: str,
    output_dir: str | os.PathLike[str],
    *,
    pass_configs: Mapping[str, object] | None = None,
    developer: bool = False,
    compiler: str | None = None,
    timeout: int = 900,
    verbose: bool = False,
) -> BishengirCompileResult:
    """Compile A2 MLIR text into ``<name>.o`` / ``lib<name>.so``.

    The output base is ``kernel`` to match the legacy JIT's temp-file layout;
    bishengir-compile appends ``.o`` and writes ``libkernel.so``.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mlir_path = out / "kernel.npuir"
    mlir_path.write_text(mlir)

    compiler = compiler or find_bishengir_compiler()
    output_base = out / "kernel"
    command = compiler_command(compiler, mlir_path, output_base, pass_configs, developer=developer)

    if verbose:
        print("bishengir-compile:", shlex.join(command))
    proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)

    object_path = out / "kernel.o"
    shared_object_path = out / "libkernel.so"
    if proc.returncode != 0 or not object_path.is_file():
        detail = proc.stderr or proc.stdout
        raise RuntimeError(
            "AscendNPU IR compile failed.\n"
            f"Command: {shlex.join(command)}\n"
            f"Return code: {proc.returncode}\n"
            f"Output:\n{detail}"
        )

    return BishengirCompileResult(
        command=tuple(command),
        mlir_path=mlir_path,
        output_base=output_base,
        object_path=object_path,
        shared_object_path=shared_object_path,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def find_bishengir_opt() -> str:
    """Locate ``bishengir-opt`` for post-codegen canonicalization."""

    explicit = os.environ.get(_ENV_OPT)
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return str(path)
        raise FileNotFoundError(f"{_ENV_OPT}={explicit!r} is not a file")

    for env_name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        root = os.environ.get(env_name)
        if not root:
            continue
        for rel in ("tools/bishengir/bin/bishengir-opt", "bin/bishengir-opt"):
            candidate = Path(root) / rel
            if candidate.is_file():
                return str(candidate)

    found = shutil.which("bishengir-opt")
    if found:
        return found
    raise RuntimeError(
        "Cannot find bishengir-opt. Set TILELANG_NPU_OPT_PATH or put the "
        "CANN tools/bishengir/bin directory on PATH."
    )


def canonicalize_mlir(mlir: str, *, opt: str | None = None, timeout: int = 300) -> str:
    """Run the canonicalizer used by the legacy A2 pipeline.

    The canonicalizer is what hoists constants and normalizes commutative
    operands; golden MLIR is compared after this step.
    """

    import tempfile

    opt = opt or find_bishengir_opt()
    with tempfile.TemporaryDirectory(prefix="tilelang-a2-canon-") as tmp:
        src = Path(tmp) / "kernel.npuir"
        dst = Path(tmp) / "kernel.canon.mlir"
        src.write_text(mlir)
        proc = subprocess.run(
            [opt, os.fspath(src), "--canonicalize", "-o", os.fspath(dst)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0 or not dst.is_file():
            raise RuntimeError(
                "bishengir-opt canonicalize failed.\n"
                f"Command: {shlex.join([opt, os.fspath(src), '--canonicalize', '-o', os.fspath(dst)])}\n"
                f"Output:\n{proc.stderr or proc.stdout}"
            )
        return dst.read_text()


def normalize_compiler_command(command: Sequence[str]) -> list[str]:
    """Normalize temp paths for golden-command comparison."""

    normalized: list[str] = []
    for token in command:
        if token.endswith((".npuir", ".mlir")):
            normalized.append("<MLIR>")
        elif token.startswith("/tmp/"):
            # Keep the basename, drop the random temporary directory.
            normalized.append("<TMP>/" + Path(token).name)
        else:
            normalized.append(token)
    return normalized
