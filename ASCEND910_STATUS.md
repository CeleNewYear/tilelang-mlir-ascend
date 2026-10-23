# Ascend 910 (A2) rebase status

This branch starts from tilelang `v0.1.15` and rebuilds the A2 backend in a
parallel `ascend910` namespace, independent of the Ascend 950 backend.

## Implemented and verified on CPU

| area | module | verification |
|---|---|---|
| target | `src/ascend910/target.cc`, `tilelang/ascend910/target.py` | `Target("ascend910")` is a distinct kind. |
| compiler | `tilelang/ascend910/bishengir.py` | Exact legacy flag order + canonicalizer. |
| cache | `tilelang/ascend910/cache.py` | Content-addressed MLIR/object cache. |
| launch source/compile | `tilelang/ascend910/runtime.py`, `launcher_compile.py` | Launcher source is byte-identical to golden; host `.so` builds. |
| codegen | `src/ascend910/target/` | Expert + developer MLIR emitters; registered FFI builders. |
| frontend | `tilelang/ascend910/language/` | Kernel/copy/alloc_ub/alloc_L1/L0A/L0B/L0C, npuir ops, v-aliases, gemm, Scope/rs, sync flags/barriers. |
| pipeline | `tilelang/ascend910/pipeline.py` | BindTarget, MaterializeKernelLaunch, LowerOpaqueBlock, Simplify. |
| lowering | `tilelang/engine/lower.py` | A2 bypasses the CUDA host/device split and calls codegen directly. |
| examples | `examples/` | Legacy A2 examples migrated to `target="ascend910"`; `is_npu=True` removed. |

### End-to-end

- `vec_add_1d`: expert and developer MLIR byte-identical to golden.
- `example_elementwise_add.py`: frontend → codegen succeeds.
- `example_gemv.py`: frontend → codegen succeeds.
- `example_mixcv.py`: frontend → codegen succeeds.
- `vec_add_2d.py`: frontend → codegen succeeds (MLIR is semantically
  equivalent; dead linear-offset ops still differ from golden).
- `flash_attn_npuir.py`: frontend builds TIR; codegen reports a vector-rank
  assertion in `vcast`.
- 14 unit tests pass.

## Remaining

1. Remove the dead linear-offset ops in 2-D copies to match the 2-D golden.
2. Fix the flash `vcast` rank assertion and mixed Cube/Vector scope lowering.
3. A2 JIT adapter: `tilelang.compile(target="ascend910")` reaches the Cython
   adapter and stops at its CUDA-only target check. Bind `bishengir.py`,
   `runtime.py`, `launcher_compile.py` and `cache.py` into an A2 execution
   backend.
4. Port A2 autotune search configuration.
5. Full golden comparison, then re-enable the original CI workflows.

## Environment

- WSL Ubuntu 22.04, a2 env (Python 3.11)
- torch 2.10.0+cpu, torch_npu 2.10.0
- CANN 9.3 weekly, `bishengir-compile`/`bishengir-opt`
- AscendNPU-IR `31f690369d1247fbd5529a3f88b758f7d470ae4f`
- 16 build jobs
