"""A2 GEMM compatibility wrapper.

Legacy A2 examples use the old ``T.gemm`` signature with an optional
``initC``/``size`` argument.  The A2 backend emits ``tl.npuir_dot`` directly;
this wrapper maps the old surface onto it.
"""

from __future__ import annotations

from typing import Optional

from tvm import tirx as tir

from . import npuir


def _shape(x):
    if isinstance(x, tir.Buffer):
        return list(x.shape)
    if isinstance(x, tir.BufferRegion):
        return [r.extent for r in x.region]
    return list(x.shape)


def gemm(
    A,
    B,
    C,
    *args,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy=None,
    clear_accum: bool = False,
    initC: Optional[bool] = None,
    size=None,
    **kwargs,
):
    if "a_transpose" in kwargs:
        transpose_A = bool(kwargs["a_transpose"])
    if "b_transpose" in kwargs:
        transpose_B = bool(kwargs["b_transpose"])
    if args:
        if isinstance(args[0], (list, tuple)):
            size = args[0]
        else:
            vals = list(args)
            if len(vals) >= 1:
                transpose_A = bool(vals[0])
            if len(vals) >= 2:
                transpose_B = bool(vals[1])
    if size is None:
        c_shape = _shape(C)
        a_shape = _shape(A)
        b_shape = _shape(B)
        m, n = c_shape[-2], c_shape[-1]
        k = a_shape[-2] if transpose_A else a_shape[-1]
        size = [m, k, n]
    return npuir.npuir_dot(
        A,
        B,
        C,
        size=size,
        initC=clear_accum if initC is None else bool(initC),
        a_transpose=transpose_A,
        b_transpose=transpose_B,
    )
