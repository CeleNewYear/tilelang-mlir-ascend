"""Compile generated A2 launcher sources (host side)."""

from __future__ import annotations

import os
import shlex
import subprocess
import sysconfig
from pathlib import Path


def _get_cxx() -> str:
    return os.environ.get("CXX", "clang++")


def _ascend_path() -> str:
    for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        value = os.environ.get(key)
        if value:
            return value
    raise RuntimeError("ASCEND_HOME_PATH is not set; source the CANN set_env.sh first")


def _torch_npu_path() -> Path:
    import torch

    return Path(torch.__file__).resolve().parent.parent / "torch_npu"


def launcher_compile_command(
    src_path: str | os.PathLike[str],
    header_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> list[str]:
    """Build the host-side compile argv for a generated launcher.

    The ordering mirrors the legacy ``build_npu_ext`` call so the argv can be
    compared with the golden launch command.
    """

    import pybind11

    cxx = _get_cxx()
    src = os.fspath(src_path)
    header = os.fspath(header_path)
    out = os.fspath(output_path)
    ascend = _ascend_path()
    torch_npu = _torch_npu_path()

    cmd = [cxx, src, "-w"]
    cmd.append(f"-I{sysconfig.get_paths()['include']}")
    cmd.append(f"-I{Path(__file__).resolve().parent}")
    if header:
        cmd.append(f"-I{Path(header).resolve().parent}")
    rt_path = Path(ascend) / "include/experiment/runtime/runtime/rt.h"
    if not rt_path.exists():
        cmd += [
            f"-I{Path(ascend) / 'pkg_inc'}",
            f"-I{Path(ascend) / 'pkg_inc/profiling'}",
        ]
    cmd += [
        f"-I{Path(ascend) / 'include'}",
        f"-I{Path(ascend) / 'include/experiment'}",
        f"-I{Path(ascend) / 'include/experiment/msprof'}",
        f"-I{pybind11.get_include()}",
        f"-L{Path(ascend) / 'lib64'}",
        "-lruntime",
        "-lascendcl",
    ]
    cmd += [
        f"-I{torch_npu / 'include'}",
        f"-I{torch_npu.parent / 'torch/include'}",
        f"-I{torch_npu.parent / 'torch/include/torch/csrc/api/include'}",
        "-D_GLIBCXX_USE_CXX11_ABI=1",
        f"-L{torch_npu / 'lib'}",
        "-ltorch_npu",
    ]
    cmd += ["-std=c++17", "-shared", "-fPIC", "-Winvalid-pch", "-o", out]
    return cmd


def compile_launcher(
    wrapper_src: str,
    header_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    verbose: bool = False,
) -> tuple[str, tuple[str, ...]]:
    """Compile *wrapper_src* into a shared library and return its path + argv."""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    src = out.with_suffix(".cpp")
    src.write_text(wrapper_src)
    # The generated source includes the fixed header name ``npu_launcher.h``.
    staged_header = out.parent / "npu_launcher.h"
    staged_header.write_text(Path(header_path).read_text())
    cmd = launcher_compile_command(src, staged_header, out)
    if verbose:
        print("launcher compile:", shlex.join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out.is_file():
        raise RuntimeError(
            "Failed to compile A2 launcher.\n"
            f"Command: {shlex.join(cmd)}\n"
            f"Output:\n{proc.stderr or proc.stdout}"
        )
    return os.fspath(out), tuple(cmd)
