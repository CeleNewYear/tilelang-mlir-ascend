"""Ascend 910 (A2) backend, kept parallel to the Ascend 950 backend.

Importing this package registers the ``ascend910`` target.  The actual A2
lowering/codegen lives in this namespace and must not depend on
``tilelang.ascend`` (950) internals.
"""

from __future__ import annotations

from . import cache  # noqa: F401
from . import runtime  # noqa: F401
from .launcher_compile import compile_launcher, launcher_compile_command  # noqa: F401
from . import target  # noqa: F401  (registers target normalizer)
from .bishengir import (  # noqa: F401
    BishengirCompileResult,
    compile_mlir,
    find_bishengir_compiler,
    normalize_compiler_command,
)
from .runtime import generate_npu_wrapper_src  # noqa: F401

__all__ = [
    "BishengirCompileResult",
    "cache",
    "compile_launcher",
    "launcher_compile_command",
    "compile_mlir",
    "find_bishengir_compiler",
    "generate_npu_wrapper_src",
    "normalize_compiler_command",
    "runtime",
]
