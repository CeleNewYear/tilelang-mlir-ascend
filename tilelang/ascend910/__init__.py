"""Ascend 910 (A2) backend, kept parallel to the Ascend 950 backend.

Importing this package registers the ``ascend910`` target.  The actual A2
lowering/codegen lives in this namespace and must not depend on
``tilelang.ascend`` (950) internals.
"""

from __future__ import annotations

from . import cache  # noqa: F401
from . import target  # noqa: F401  (registers target normalizer)
from .bishengir import (  # noqa: F401
    BishengirCompileResult,
    compile_mlir,
    find_bishengir_compiler,
    normalize_compiler_command,
)

__all__ = [
    "BishengirCompileResult",
    "cache",
    "compile_mlir",
    "find_bishengir_compiler",
    "normalize_compiler_command",
]
