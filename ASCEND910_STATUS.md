# Ascend 910 (A2) rebase status

This branch starts from tilelang `v0.1.15` and rebuilds the A2 backend in a
parallel `ascend910` namespace, independent of the Ascend 950 backend.

## Implemented and verified on CPU

| area | module | verification |
|---|---|---|
| target | `src/ascend910/target.cc`, `tilelang/ascend910/target.py` | `Target("ascend910")` is a distinct kind. |
| compiler | `tilelang/ascend910/bishengir.py` | Exact legacy flag order + canonicalizer. |
| cache | `tilelang/ascend910/cache.py` | Content-addressed MLIR/object cache. |
| launch source/compile | `tilelang/ascend910/runtime.py`, `launcher_compile.py` | Launcher source byte-identical to golden; host `.so` builds. |
| codegen | `src/ascend910/target/` | Expert + developer MLIR emitters; registered FFI builders. |
| frontend | `tilelang/ascend910/language/` | Kernel/copy/alloc APIs, npuir ops, v-aliases (including vlog2), gemm, Scope/rs, sync. |
| pipeline | `tilelang/ascend910/pipeline.py` | BindTarget, MaterializeKernelLaunch, LowerOpaqueBlock, Simplify. |
| lowering | `tilelang/engine/lower.py` | A2 bypasses the CUDA host/device split and calls codegen directly. |
| JIT | `tilelang/ascend910/adapter.py` | `tilelang.compile(target="ascend910")` returns a JITKernel with MLIR; launch raises until the runtime bridge is connected. |
| examples | `examples/` | Legacy A2 examples migrated to `target="ascend910"`; `is_npu=True` removed. |

### End-to-end

- `vec_add_1d`: expert and developer MLIR byte-identical to golden.
- `vec_add_2d`: MLIR byte-identical to golden after canonicalization.
- `example_elementwise_add.py`, `example_elementwise_exp2.py`,
  `example_elementwise_log2.py`, `vec_add_auto_brc.py`, `example_gemv.py`,
  `example_mixcv.py`: frontend → codegen succeeds.
- `vec_add_2d_dynamic_shape.py`: frontend builds TIR; codegen cannot resolve
  the symbolic `M`/`N` vars yet.
- `flash_attn_npuir.py`: frontend builds TIR; codegen reports a `vcast` rank
  assertion.
- 15 unit tests pass, including compile-only JIT.

## Remaining

1. Bind dynamic symbolic shape vars as explicit kernel scalar args.
2. Fix the flash `vcast` rank assertion and mixed Cube/Vector scope lowering.
3. Connect `bishengir.py`, `runtime.py`, `launcher_compile.py` and `cache.py`
   into a real A2 execution backend (currently compile-only).
4. Port A2 autotune search configuration.
5. Full golden comparison and re-enable the original CI workflows.

## Environment

- WSL Ubuntu 22.04, a2 env (Python 3.11)
- torch 2.10.0+cpu, torch_npu 2.10.0
- CANN 9.3 weekly, `bishengir-compile`/`bishengir-opt`
- AscendNPU-IR `31f690369d1247fbd5529a3f88b758f7d470ae4f`
- 16 build jobs
