# Ascend 910 (A2) rebase status

This branch starts from tilelang `v0.1.15` and rebuilds the A2 backend in a
parallel `ascend910` namespace, independent of the Ascend 950 backend.

## Implemented and verified on CPU

| area | module | verification |
|---|---|---|
| target | `src/ascend910/target.cc`, `tilelang/ascend910/target.py` | `ascend910`/`A2`/`npuir` normalize to one target kind. |
| compiler | `tilelang/ascend910/bishengir.py` | Exact legacy flags + canonicalizer. |
| cache | `tilelang/ascend910/cache.py` | MLIR/object content-addressed cache, used by the JIT adapter. |
| launch generation | `tilelang/ascend910/runtime.py`, `launcher_compile.py` | Byte-compatible wrapper source; host launcher `.so` builds. |
| codegen | `src/ascend910/target/` | Expert + developer MLIR emitters; bool/tirx arithmetic fixes. |
| frontend | `tilelang/ascend910/language/` | Kernel/copy/alloc APIs, npuir ops, v-aliases, gemm, Scope/rs, sync, reduce, clear/fill. |
| pipeline | `tilelang/ascend910/pipeline.py` | BindTarget, symbolic params, allocation placement, MaterializeKernelLaunch, LowerOpaqueBlock, Simplify. |
| lowering | `tilelang/engine/lower.py` | A2 bypasses the CUDA host/device split. |
| JIT | `tilelang/ascend910/adapter.py` | `tilelang.compile` → MLIR → `kernel.o` → `libkernel_launcher.so`, cached. Device load/launch is the remaining bridge. |
| examples | `examples/` | Legacy A2 examples migrated to `target="ascend910"`. |

### Golden suite

`work/run_golden_suite.py` runs every migrated example with a golden module:

- 34 examples with a matching source file
- 26 reach canonicalized MLIR
- 11 are byte-identical to golden after canonicalization:
  `vec_add_1d`, `vec_add_2d`, `vec_add_2d_dynamic_shape`,
  `vec_add_2d_multi_buffer`, `atomic_add`, `atomic_add_dev`,
  `elementwise_add`, `elementwise_exp2`, `elementwise_log2`,
  `ada_layer_norm_and_zero`, `rms_norm`, `fp8_lighting_indexer`
- Remaining differences are mostly allocation/attribute ordering and a few
  mixed-kernel/frontend semantics (flash, sparse-MLA, some TileOPs kernels).

Unit tests: **19 passed**, including compile-only JIT, cached object/launcher
compilation, and 2-D/dynamic golden MLIR.

## Remaining

1. Mixed Cube/Vector resource-scope splitting for flash/sparse-MLA.
2. A2 device load/launch bridge (`NPUUtils.load_binary`) — requires NPU host.
3. Autotune profiling on A2.
4. Remaining golden diffs (allocation order, O2 attribute ordering).
5. CI workflows (need the AscendNPU-IR/TVM build and NPU runner images).

## Environment

- WSL Ubuntu 22.04, a2 env (Python 3.11)
- torch 2.10.0+cpu, torch_npu 2.10.0
- CANN 9.3 weekly, `bishengir-compile`/`bishengir-opt`
- AscendNPU-IR `31f690369d1247fbd5529a3f88b758f7d470ae4f`
- 16 build jobs
