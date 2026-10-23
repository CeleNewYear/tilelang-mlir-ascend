"""Compile-only JIT path for A2."""
from __future__ import annotations

import pytest

import tilelang
import tilelang.language as T


@T.prim_func
def _copy(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
    with T.Kernel(1) as (cid, _):
        s = T.alloc_ub((64,), "float32")
        T.copy(A[cid * 64 : cid * 64 + 64], s)
        T.copy(s, B[cid * 64 : cid * 64 + 64])


def test_compile_only():
    kernel = tilelang.compile(_copy, target="ascend910")
    src = kernel.get_kernel_source()
    assert "hivm.hir.set_ffts_base_addr" in src
    assert "func.func" in src
    with pytest.raises(RuntimeError, match="runtime bridge"):
        kernel(None, None)
