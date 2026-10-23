"""Compatibility helpers for the legacy A2 ``NPUUtils`` singleton.

The old tree queried device core counts through a CANN extension.  On a
CPU-only checkout there is no device, so report the 910B topology (24 AI cores,
48 vector cores) unless ``TILELANG_A2_AICORE_NUM`` is set.
"""

from __future__ import annotations

import os


class NPUUtils:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_aicore_num(self) -> int:
        return int(os.environ.get("TILELANG_A2_AICORE_NUM", "24"))

    def get_aivector_core_num(self) -> int:
        return int(os.environ.get("TILELANG_A2_AIVECTOR_CORE_NUM", "48"))
