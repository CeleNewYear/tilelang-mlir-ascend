"""Ascend 910 (A2) lowering pipeline.

The frontend already produces NPUIR calls and a kernel launch frame.  These
passes materialize the launch, expose allocations, and simplify the folded
1-D grid before the MLIR emitter runs.
"""

from __future__ import annotations

from tvm import IRModule, tirx
from tvm.ir import CallingConv
from tvm.target import Target

import tilelang
from tilelang.backend.pass_pipeline import PassPipeline


def Ascend910PassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    # The frontend builds one PrimFunc per kernel.  Mark it as a device kernel
    # so the host/device filter in engine.lower keeps it for codegen.
    for gv in list(mod.functions):
        func = mod[gv]
        if not isinstance(func, tirx.PrimFunc):
            continue
        # The eager frontend keeps dynamic shape symbols as free vars in buffer
        # shapes.  A2 codegen expects them as explicit scalar kernel arguments.
        known = set(func.params)
        extra = []
        for buf in func.buffer_map.values():
            for dim in buf.shape:
                for var in tirx.analysis.undefined_vars(dim, []):
                    if var not in known and var not in extra and str(var.dtype).startswith("int"):
                        extra.append(var)
        if extra:
            func = tirx.PrimFunc(
                list(func.params) + extra,
                func.body,
                func.ret_type,
                func.buffer_map,
                func.attrs,
            )
        mod[gv] = func.with_attr("calling_conv", CallingConv.DEVICE_KERNEL_LAUNCH)
    mod = tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch(
        lower_grid_binding=True,
        lower_thread_binding=False,
        default_threads=None,
        unsupported_annotations=["cluster_dims"],
        launch_dim_tags=["cthread"],
    )(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.Simplify()(mod)
    return mod


ascend910_pipeline = PassPipeline("ascend910", Ascend910PassPipelineBody)
