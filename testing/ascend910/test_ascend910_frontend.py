"""End-to-end A2 frontend -> lowering -> codegen golden test."""
from __future__ import annotations

from pathlib import Path

import pytest

import tilelang
import tilelang.language as T
from tilelang.ascend910.bishengir import canonicalize_mlir

ROOT = Path(__file__).resolve().parents[2] / "golden" / "a2" / "smoke"
N = 4096


def _build():
    @T.prim_func
    def main(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), "float32"),
        shape: T.int32,
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((N,), "float32")
            B_VEC = T.alloc_ub((N,), "float32")
            C_VEC = T.alloc_ub((N,), "float32")
            tail_size = T.min(N, shape - cid * N)
            T.copy(A[cid * N : cid * N + tail_size], A_VEC[0:tail_size])
            T.copy(B[cid * N : cid * N + tail_size], B_VEC[0:tail_size])
            T.vadd(A_VEC, B_VEC, C_VEC)
            T.copy(C_VEC[0:tail_size], C[cid * N : cid * N + tail_size])

    return main


@pytest.mark.parametrize("mode", ["expert", "developer"])
def test_vec_add_matches_golden(monkeypatch, mode):
    monkeypatch.setenv("TILELANG_ASCEND_MODE", mode)
    artifact = tilelang.lower(_build(), target="ascend910")
    got = canonicalize_mlir(str(artifact.kernel_source)).strip()
    gpath = ROOT / ("vec_add_1d.mlir" if mode == "expert" else "vec_add_1d_dev.golden.mlir")
    assert got == gpath.read_text().strip()
