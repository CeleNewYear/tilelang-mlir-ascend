"""A2 codegen registration and MLIR emission tests (host only)."""
from __future__ import annotations

import pytest

import tilelang
import tilelang.language as T
from tvm.target import Target

from tilelang.ascend910.codegen import ASCEND910_CODEGEN


@T.prim_func
def _copy_kernel(A: T.Buffer((256,), "float32"), B: T.Buffer((256,), "float32")):
    B[0] = A[0]


def _lowered():
    mod = tilelang.tvm.IRModule({"main": _copy_kernel})
    return tilelang.transform.LowerOpaqueBlock()(mod)


@pytest.mark.parametrize("mode", ["expert", "developer"])
def test_codegen_emits_mlir(monkeypatch, mode):
    monkeypatch.setenv("TILELANG_ASCEND_MODE", mode)
    out = ASCEND910_CODEGEN.lower(_lowered(), Target("ascend910"), compile_device=False)
    src = out.inspect_source()
    assert "hivm.hir.set_ffts_base_addr" in src
    assert "func.func" in src
    assert "hacc.entry" in src
    if mode == "expert":
        assert "#hivm.address_space<gm>" in src
    else:
        assert "#hivm.address_space<gm>" not in src
