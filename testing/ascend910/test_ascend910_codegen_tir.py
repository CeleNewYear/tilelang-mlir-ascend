"""Codegen on a realistic A2 TIR snapshot (host only)."""
from __future__ import annotations

from pathlib import Path

import pytest

import tilelang
import tvm.tirx.script.parser as parser_ns
from tvm import ffi, tirx
from tvm.script import from_source
from tvm.target import Target

from tilelang.ascend910.codegen import ASCEND910_CODEGEN

ROOT = Path(__file__).resolve().parents[2] / "golden" / "a2" / "smoke"


def _op(name, *args):
    return tirx.call_intrin("handle", tirx.op.Op.get(name), *args)


@pytest.fixture(autouse=True)
def _install_dsl_extensions():
    parser_ns.region = lambda buf, mask, *ext: _op("tl.region", buf, mask, *ext)
    parser_ns.copy = lambda src, dst: _op("tl.copy", src, dst)
    parser_ns.npuir_add = lambda a, b, c: _op("tl.npuir_add", a, b, c)


@pytest.mark.parametrize("mode", ["expert", "developer"])
def test_vec_add_tir_emits_hivm(monkeypatch, mode):
    monkeypatch.setenv("TILELANG_ASCEND_MODE", mode)
    mod = from_source((ROOT / "vec_add_1d.tir").read_text())
    out = ASCEND910_CODEGEN.lower(mod, Target("ascend910"), compile_device=False)
    src = out.inspect_source()
    assert "hivm.hir.set_ffts_base_addr" in src
    assert "hivm.hir.vadd" in src
    if mode == "expert":
        assert src.count("memref.copy") == 3
        assert "#hivm.address_space<gm>" in src
    else:
        assert "bufferization.materialize_in_destination" in src
        assert "#hivm.address_space<gm>" not in src
