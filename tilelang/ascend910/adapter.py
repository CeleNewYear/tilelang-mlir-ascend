"""Compile-only adapter for the Ascend 910 (A2) frontend.

The device MLIR and cache are produced normally.  Launching requires the A2
runtime bridge, which is a separate work item; until then a call raises a
clear error instead of silently doing nothing.
"""

from __future__ import annotations


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

    def _call(self, *args, **kwargs):
        raise RuntimeError(
            "Ascend 910 device execution is not wired yet: the MLIR was compiled, "
            "but the launch/load runtime bridge has not been connected."
        )

    def get_kernel_source(self, kernel_only: bool = True) -> str:
        return self.device_kernel_source

    def get_profiler(self, *args, **kwargs):
        raise RuntimeError("Ascend 910 profiling requires the device runtime bridge")
