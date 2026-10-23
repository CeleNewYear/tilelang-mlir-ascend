# Ascend 910 (A2) rebase status

This branch starts from tilelang `v0.1.15` and rebuilds the A2 backend in a
parallel `ascend910` namespace, independent of the Ascend 950 backend.

## Implemented and verified on CPU

| area | module | verification |
|---|---|---|
| target | `src/ascend910/target.cc`, `tilelang/ascend910/target.py` | `Target("ascend910")` is a distinct kind; `A2`/`Ascend910B` normalize through `determine_target`. |
| compiler | `tilelang/ascend910/bishengir.py` | Exact legacy flag order; compiles golden MLIR; canonicalizer post-pass. |
| cache | `tilelang/ascend910/cache.py` | Content-addressed by MLIR/pass config/mode/compiler. |
| launch source | `tilelang/ascend910/runtime.py` | Byte-identical to the `vec_add_1d` launcher golden. |
| launch compile | `tilelang/ascend910/launcher_compile.py` | Builds the generated launcher `.so` on CPU. |
| codegen | `src/ascend910/target/` | Expert and developer MLIR emitters, registered as `target.build.tilelang_ascend910{,_dev}`. |
| frontend | `tilelang/ascend910/language/` | `Kernel`, `copy`, `alloc_ub`, npuir ops and v-prefix aliases; core `tilelang.language` dispatches while an A2 frame is active. |
| pipeline | `tilelang/ascend910/pipeline.py` | BindTarget, MaterializeKernelLaunch, LowerOpaqueBlock, Simplify; backend manifest registered. |
| golden data | `golden/a2` | 426 kernels / 70 examples, both modes, with MLIR, argv, launchers. |

### End-to-end result

`tilelang.lower(kernel, target="ascend910")` now works. For the legacy
`vec_add_1d` example, both expert and developer output match the golden MLIR
**byte-for-byte** after `bishengir-opt --canonicalize`. The same is true for
`example_elementwise_add.py`. 14 tests pass.

## Remaining

1. `T.gemm` / `T.Scope` compatibility for mixcv and flash-attention examples.
2. A2 JIT adapter: `tilelang.compile(target="ascend910")` currently reaches the
   Cython adapter and stops because it only knows CUDA targets. The next step is
   to bind `bishengir.py`, `runtime.py`, `launcher_compile.py` and `cache.py`
   into an A2 execution backend.
3. Port the A2 autotune search configuration.
4. Run the full golden comparison, then re-enable the original CI workflows.

## Environment

- WSL Ubuntu 22.04, a2 env (Python 3.11)
- torch 2.10.0+cpu, torch_npu 2.10.0
- CANN 9.3 weekly, `bishengir-compile`/`bishengir-opt`
- AscendNPU-IR `31f690369d1247fbd5529a3f88b758f7d470ae4f`
- 16 build jobs
