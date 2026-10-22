"""Tests for the Ascend 910 (A2) cast/compile plumbing.

These tests do not need an NPU: bishengir-compile and the cache are host-side.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from tvm.target import Target

from tilelang.ascend910 import bishengir, cache
from tilelang.backend.target import determine_target

GOLDEN = Path(__file__).resolve().parents[2] / "golden" / "a2" / "expert" / "elementwise__vec_add_1d" / "000_main.mlir"


def test_target_normalization():
    target = Target(determine_target("ascend910"))
    assert target.kind.name == "ascend910"
    assert Target(determine_target("A2")).kind.name == "ascend910"


def test_compiler_options_are_golden_order():
    assert bishengir.compiler_options() == [
        "--enable-auto-multi-buffer=true",
        "--disable-hivm-auto-inject-sync=false",
        "--enable-triton-kernel-compile=true",
        "--enable-hivm-compile=true",
        "--disable-hivm-tensor-compile=true",
    ]
    assert bishengir.compiler_options(developer=True) == [
        "--enable-auto-multi-buffer=true",
        "--disable-hivm-auto-inject-sync=false",
        "--enable-triton-kernel-compile=true",
        "--enable-hivm-compile=true",
    ]


@pytest.mark.parametrize("developer,expected", [(False, True), (True, False)])
def test_compile_and_cache(developer, expected, tmp_path, monkeypatch):
    if not GOLDEN.is_file():
        pytest.skip("golden corpus is not checked out")
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path))
    mlir = GOLDEN.read_text()
    assert cache.load(mlir, developer=developer) is None
    result = bishengir.compile_mlir(
        mlir,
        tmp_path / "build",
        developer=developer,
        compiler=bishengir.find_bishengir_compiler(),
    )
    assert result.object_path.is_file()
    assert result.object_path.stat().st_size > 0
    assert "-o" in result.command
    key = cache.make_key(mlir, developer=developer)
    entry = cache.save(
        key,
        mlir,
        result.object_path,
        result.shared_object_path,
        developer=developer,
    )
    assert entry.ready
    assert cache.load(mlir, developer=developer) is not None
    assert cache.load(mlir, developer=not developer) is None
    assert (expected and ("--disable-hivm-tensor-compile=true" in result.command)) or (
        not expected and ("--disable-hivm-tensor-compile=true" not in result.command)
    )
