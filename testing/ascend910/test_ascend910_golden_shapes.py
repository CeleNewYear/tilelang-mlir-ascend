"""Golden MLIR for 2-D and dynamic-shape elementwise kernels."""
from __future__ import annotations

from pathlib import Path

import pytest

import tilelang
import tilelang.language as T
from tilelang.ascend910.bishengir import canonicalize_mlir

ROOT = Path(__file__).resolve().parents[2] / "golden" / "a2" / "smoke"


def _make_2d():
    M, N, block_M, block_N = 512, 512, 128, 128
    m_num, n_num = M // block_M, N // block_N
    dtype = "float16"
    BLOCK_SIZE = 20
    globals().update(locals())

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE) as (cid, _):
            a_vec = T.alloc_ub((block_M, block_N), dtype)
            b_vec = T.alloc_ub((block_M, block_N), dtype)
            c_vec = T.alloc_ub((block_M, block_N), dtype)
            for i in T.serial(T.ceildiv(m_num * n_num, BLOCK_SIZE)):
                block_id = i * BLOCK_SIZE + cid
                if block_id < m_num * n_num:
                    block_m = block_id // n_num
                    block_n = block_id % n_num
                    bx = block_m * block_M
                    by = block_n * block_N
                    T.copy(A[bx, by], a_vec)
                    T.copy(B[bx, by], b_vec)
                    T.vadd(a_vec, b_vec, c_vec)
                    T.copy(c_vec, C[bx, by])

    return main


def _make_dynamic():
    M = T.symbolic("M")
    N = T.symbolic("N")
    block_M, block_N = 32, 32
    dtype = "float16"
    BLOCK_SIZE = 20
    globals().update(locals())

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE) as (cid, _):
            a_vec = T.alloc_ub((block_M, block_N), dtype)
            b_vec = T.alloc_ub((block_M, block_N), dtype)
            c_vec = T.alloc_ub((block_M, block_N), dtype)
            m_num = T.ceildiv(M, block_M)
            n_num = T.ceildiv(N, block_N)
            start = cid * T.ceildiv(m_num * n_num, BLOCK_SIZE)
            for i in T.serial(T.ceildiv(m_num * n_num, BLOCK_SIZE)):
                block_id = start + i
                if block_id < m_num * n_num:
                    bx = (block_id // n_num) * block_M
                    by = (block_id % n_num) * block_N
                    remain_m = T.min(M - bx, block_M)
                    remain_n = T.min(N - by, block_N)
                    T.copy(A[bx : bx + remain_m, by : by + remain_n], a_vec)
                    T.copy(B[bx : bx + remain_m, by : by + remain_n], b_vec)
                    T.vadd(a_vec, b_vec, c_vec)
                    T.copy(c_vec, C[bx : bx + remain_m, by : by + remain_n])

    return main


@pytest.mark.parametrize(
    "builder,golden",
    [
        (_make_2d, "vec_add_2d.mlir"),
        (_make_dynamic, "vec_add_2d_dynamic_shape.mlir"),
    ],
)
def test_shape_golden(builder, golden):
    art = tilelang.lower(builder(), target="ascend910")
    got = canonicalize_mlir(art.kernel_source).strip()
    assert got == (ROOT / golden).read_text().strip()
