"""Golden launcher generation test (A2)."""
from __future__ import annotations

import json
from pathlib import Path

from tilelang.ascend910.runtime import generate_npu_wrapper_src

ROOT = Path(__file__).resolve().parents[2] / "golden" / "a2" / "smoke"


def test_launcher_is_byte_identical():
    meta = json.loads((ROOT / "vec_add_1d.json").read_text())
    signature = {int(k): v for k, v in meta["signature"].items()}
    generated = generate_npu_wrapper_src(
        constants={},
        signature=signature,
        workspace_size=meta["workspace_size"],
        mix_mode=meta["mix_mode"],
        lock_num=meta["lock_num"],
        lock_ini_val=meta["lock_ini_val"],
        need_debug=meta["need_debug"],
        force_simt_only=meta["force_simt_only"],
        shared_mem_dynamic_size=meta["shared_mem_dynamic_size"],
    )
    assert generated == (ROOT / "vec_add_1d.launcher.cpp").read_text()
