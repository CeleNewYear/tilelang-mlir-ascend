"""A2-specific kernel launch surface."""

from __future__ import annotations

import threading
from typing import Any

from tilelang.language.kernel import launch_kernel

_A2_ACTIVE = threading.local()


def _set_active(value: bool):
    _A2_ACTIVE.value = value


def is_active() -> bool:
    return bool(getattr(_A2_ACTIVE, "value", False))


class _A2KernelFrame:
    """Return the TLA-style ``(cid, subid)`` pair for a 1-D NPU launch."""

    def __init__(self, frame):
        self._frame = frame

    def __enter__(self):
        _set_active(True)
        value = self._frame.__enter__()
        if isinstance(value, (list, tuple)):
            values = list(value)
            return values[0] if len(values) == 1 else values
        return value, 0

    def __exit__(self, ptype, value, trace):
        try:
            return self._frame.__exit__(ptype, value, trace)
        finally:
            _set_active(False)


def Kernel(
    *blocks,
    threads: int | list[int] | tuple[int, ...] | None = None,
    is_cpu: bool = False,
    prelude: str | None = None,
    is_npu: bool = False,
    pipeline: bool = False,
    **kwargs: Any,
):
    attrs: dict[str, Any] = {}
    if is_npu:
        attrs["tilelang.is_npu_kernel_frame"] = True
    if prelude is not None:
        attrs["pragma_import_c"] = prelude
    attrs.update(kwargs)
    frame = launch_kernel(blocks, threads=threads, **attrs)
    return _A2KernelFrame(frame) if is_npu else frame
