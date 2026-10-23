"""Ascend 910 (A2) JIT adapter.

Compilation is fully supported: the adapter can emit MLIR, run
``bishengir-compile`` and build the host launcher.  Launching still requires the
runtime bridge (``NPUUtils.load_binary``), which is only available on an NPU
host, so a call raises a clear error instead of silently doing nothing.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path


_MLIR_DTYPE = {
    "float16": "fp16",
    "bfloat16": "bf16",
    "float32": "fp32",
    "float": "fp32",
    "float64": "fp64",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "uint32": "u32",
    "uint64": "u64",
    "bool": "i1",
}


class Ascend910KernelAdapter:
    def __init__(
        self,
        *,
        params=None,
        result_idx=None,
        target=None,
        func_or_mod=None,
        host_mod=None,
        device_mod=None,
        device_kernel_source: str = "",
        verbose: bool = False,
        pass_configs=None,
        compile_flags=None,
        **kwargs,
    ):
        self.params = params or []
        self.result_idx = result_idx or []
        self.target = target
        self.func_or_mod = func_or_mod
        self.host_mod = host_mod
        self.device_mod = device_mod
        self.device_kernel_source = device_kernel_source
        self.verbose = verbose
        self.pass_configs = pass_configs or {}
        self.compile_flags = compile_flags
        self.func = self._call
        self.object_path: str | None = None
        self.launcher_path: str | None = None
        self.wrapper_src: str | None = None
        self._compiled = False

    # -- signature / metadata -------------------------------------------------
    def _signature(self):
        if self.params:
            sig = {}
            for i, p in enumerate(self.params):
                try:
                    dtype = str(p.dtype)
                except Exception:
                    dtype = "float32"
                sig[i] = "*" + _MLIR_DTYPE.get(dtype, "fp32")
            return sig
        sig = {}
        f = self.func_or_mod
        if f is not None and hasattr(f, "params") and hasattr(f, "buffer_map"):
            for i, p in enumerate(f.params):
                if p in f.buffer_map:
                    sig[i] = "*" + _MLIR_DTYPE.get(
                        str(f.buffer_map[p].dtype), "fp32"
                    )
                else:
                    sig[i] = _MLIR_DTYPE.get(str(p.dtype), "i32")
        return sig

    def _metadata(self):
        src = self.device_kernel_source
        name = re.search(r"func\.func\s+@(\w+)", src)
        mix = re.search(r"#hivm\.module_core_type<([^>]+)>", src)
        return (
            name.group(1) if name else "main",
            (mix.group(1).lower() if mix else "aiv"),
        )

    # -- compilation ----------------------------------------------------------
    def compile_artifacts(self, output_dir: str | None = None):
        """Compile MLIR to an object and build the host launcher."""
        if self._compiled:
            return self.object_path, self.launcher_path
        from .bishengir import compile_mlir
        from .launcher_compile import compile_launcher
        from .runtime import (
            generate_npu_wrapper_src,
            get_npu_launcher_header,
        )

        out = Path(
            output_dir
            or os.environ.get("TILELANG_A2_BUILD_DIR")
            or tempfile.mkdtemp(prefix="tilelang_a2_")
        )
        out.mkdir(parents=True, exist_ok=True)
        result = compile_mlir(
            self.device_kernel_source,
            out,
            developer="dev" in str(self.target),
            verbose=self.verbose,
        )
        self.object_path = str(result.object_path)

        _, mix_mode = self._metadata()
        workspace = int(os.environ.get("TILELANG_ASCEND_WORKSPACE_SIZE", "0"))
        self.wrapper_src = generate_npu_wrapper_src(
            {},
            self._signature(),
            workspace,
            mix_mode,
            -1,
            0,
            False,
        )
        launcher = out / "libkernel_launcher.so"
        self.launcher_path, _ = compile_launcher(
            self.wrapper_src,
            get_npu_launcher_header(),
            launcher,
            verbose=self.verbose,
        )
        self._compiled = True
        return self.object_path, self.launcher_path

    # -- execution ------------------------------------------------------------
    def _call(self, *args, **kwargs):
        raise RuntimeError(
            "Ascend 910 device execution is not wired yet: the MLIR was compiled, "
            "but the launch/load runtime bridge has not been connected."
        )

    def get_kernel_source(self, kernel_only: bool = True) -> str:
        return self.device_kernel_source

    def get_profiler(self, *args, **kwargs):
        raise RuntimeError("Ascend 910 profiling requires the device runtime bridge")
