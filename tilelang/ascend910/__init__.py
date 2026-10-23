"""Ascend 910 (A2) backend, kept parallel to the Ascend 950 backend.

Importing this package registers the ``ascend910`` target.  The actual A2
lowering/codegen lives in this namespace and must not depend on
``tilelang.ascend`` (950) internals.
"""

from __future__ import annotations

from . import backend  # noqa: F401  (registers the backend manifest)
from . import cache  # noqa: F401
from . import runtime  # noqa: F401
from . import target  # noqa: F401  (registers target normalizer)
from .bishengir import (  # noqa: F401
    BishengirCompileResult,
    canonicalize_mlir,
    compile_mlir,
    find_bishengir_compiler,
    find_bishengir_opt,
    normalize_compiler_command,
)
from .codegen import ASCEND910_CODEGEN  # noqa: F401
from .launcher_compile import compile_launcher, launcher_compile_command  # noqa: F401
from .runtime import generate_npu_wrapper_src  # noqa: F401

__all__ = [
    "ASCEND910_CODEGEN",
    "BishengirCompileResult",
    "cache",
    "canonicalize_mlir",
    "compile_launcher",
    "compile_mlir",
    "find_bishengir_compiler",
    "find_bishengir_opt",
    "generate_npu_wrapper_src",
    "launcher_compile_command",
    "normalize_compiler_command",
    "runtime",
]
