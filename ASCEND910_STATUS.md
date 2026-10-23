# Ascend 910 (A2) rebase status

This branch starts from tilelang `v0.1.15` and rebuilds the A2 backend in a
parallel `ascend910` namespace.  It intentionally does **not** reuse
`tilelang/ascend` (Ascend 950) internals.

## Implemented and verified on CPU

| area | module | verification |
|---|---|---|
| target | `src/ascend910/target.cc`, `tilelang/ascend910/target.py` | `Target("ascend910")` is a distinct kind; `A2`/`Ascend910B` normalize through `determine_target`. |
| compiler | `tilelang/ascend910/bishengir.py` | Exact legacy flag order; compiles the smoke golden MLIR into `kernel.o` and `libkernel.so`; `bishengir-opt --canonicalize` post-pass. |
| cache | `tilelang/ascend910/cache.py` | Content-addressed by MLIR, pass config, mode, and compiler stamp; tested for mode isolation. |
| launch source | `tilelang/ascend910/runtime.py` | Byte-identical to the legacy `vec_add_1d` launcher golden. |
| launch compile | `tilelang/ascend910/launcher_compile.py` | Compiles the generated launcher to a `.so` on CPU with torch/torch_npu from the a2 env. |
| codegen | `src/ascend910/target/`, `tilelang/ascend910/codegen.py` | Expert and developer MLIR emitters compiled against `tvm/tirx`; both registered as FFI builders and linked with the AscendNPU-IR MLIR libraries. |
| codegen golden | `testing/ascend910/test_ascend910_codegen_tir.py` | `vec_add_1d` TIR snapshot produces the golden MLIR **byte-for-byte** in both modes after canonicalization. |
| golden data | `golden/a2` | 426 kernels from 70 examples, expert + developer, with MLIR, argv, launcher source/header. |

### What changed for tirx

- `tvm/tir/*` → `tvm/tirx/*`; `tir::` → `tirx::`.
- `LetStmt` → `Bind`; `Allocate` → `AllocBuffer`; `DeclBuffer` no longer carries a body.
- `Array`/`Map`/`String` are `tvm::ffi` containers.
- `runtime::Module` → `ffi::Module`; `TVM_REGISTER_GLOBAL` → `TVM_FFI_STATIC_INIT_BLOCK` + `Function::SetGlobal`.
- `String2DLDataType` → `ffi::StringToDLDataType`; `tvm::StructuralEqual` → `tvm::ffi::StructuralEqual`.

## Not yet ported

The A2 **frontend and TIR pass pipeline** are not yet wired into the
v0.1.15 engine:

1. `tilelang/ascend910/language/`: port the npuir/v-prefix DSL and the
   `is_npu` kernel frame (the v0.1.15 `tilelang.language` facade is CUDA-first).
2. `tilelang/ascend910/transform/`: port the A2 TIR passes
   (`NpuLoopVectorize`, `LowerNpuirBlock`, `PlanAndUpdateBufferAllocationLocation`,
   A2 legalization) and the expert/developer lowering order.  The codegen
   golden test starts from a TIR snapshot until this exists.
3. `tilelang/ascend910/engine.py`: dispatch `target="ascend910"` in
   `tilelang.engine.lower` without touching the A5 pipeline.
4. Wire `codegen.py` + `bishengir.py` + `cache.py` + `runtime.py` into the
   v0.1.15 JIT and autotuner, then compare against `golden/a2` with
   `tools/compare_a2_golden.py`.
5. Port the A2 autotune search configuration and re-enable the original CI
   workflows.

## Environment used

- WSL Ubuntu 22.04, a2 conda env (Python 3.11)
- torch 2.10.0+cpu, torch_npu 2.10.0
- CANN 9.3 weekly, `bishengir-compile`/`bishengir-opt` from `tools/bishengir/bin`
- AscendNPU-IR `31f690369d1247fbd5529a3f88b758f7d470ae4f` installed at
  `/home/xinyang/xinyang/AscendNPU-IR/build/install`
- 16 build jobs (half of 32 cores)
