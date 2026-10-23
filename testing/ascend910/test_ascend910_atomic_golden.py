"""Golden parity for the migrated atomic-add example."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import tilelang
from tilelang.ascend910.bishengir import canonicalize_mlir

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "elementwise" / "atomic_add.py"
GOLDEN = ROOT / "golden" / "a2" / "smoke" / "atomic_add.mlir"


def _load_example():
    spec = importlib.util.spec_from_file_location("atomic_add_example", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_atomic_add_1d_golden():
    example = _load_example()
    func = example.atomic_add_1d(64, block_size=32)
    art = tilelang.lower(func, target="ascend910")
    got = canonicalize_mlir(art.kernel_source).strip()
    assert got == GOLDEN.read_text().strip()
