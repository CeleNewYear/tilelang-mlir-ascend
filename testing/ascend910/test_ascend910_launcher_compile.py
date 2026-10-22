"""End-to-end host-side launcher build test for A2."""
from __future__ import annotations

from pathlib import Path

import pytest

from tilelang.ascend910.launcher_compile import compile_launcher

ROOT = Path(__file__).resolve().parents[2] / "golden" / "a2" / "smoke"


def test_generated_launcher_compiles(tmp_path):
    wrapper = (ROOT / "vec_add_1d.launcher.cpp").read_text()
    header = ROOT / "vec_add_1d.launcher.h"
    import torch

    torch_npu = Path(torch.__file__).resolve().parent.parent / "torch_npu"
    if not torch_npu.is_dir():
        pytest.skip("torch_npu is not installed")
    so, cmd = compile_launcher(wrapper, header, tmp_path / "vec_add.so")
    assert Path(so).is_file() and Path(so).stat().st_size > 0
    assert "-Winvalid-pch" in cmd
    assert "-ltorch_npu" in cmd
