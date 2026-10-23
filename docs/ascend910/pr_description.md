# Rebase A2 (Ascend 910) backend onto tilelang v0.1.15

## Summary

This branch rebuilds the Ascend 910 (A2) backend on top of tilelang `v0.1.15`
in a parallel `ascend910` namespace, without mixing it with the Ascend 950
`ascend` backend.

- New `src/ascend910` codegen (expert + developer) ported to `tirx`/`tvm::ffi`.
- New `tilelang/ascend910` target, compiler, cache, runtime generator and JIT
  adapter.
- Legacy A2 frontend (`Kernel`, `copy`, `alloc_ub/L1/L0`, `npuir_*`, `v*`,
  `gemm`, `Scope`, sync ops, reduce) available through `tilelang.language`.
- Migrated examples use `target="ascend910"` and no longer pass `is_npu=True`.
- Golden corpus (`golden/a2`) is committed for offline verification.

## Verification

- `testing/ascend910`: 19 passed.
- `vec_add_1d` expert/developer: byte-identical golden MLIR.
- `vec_add_2d`, dynamic shape, multi-buffer, atomic add, exp2/log2, rms norm,
  fp8 indexer: byte-identical after `bishengir-opt --canonicalize`.
- 26/34 migrated examples reach canonicalized MLIR.
- `tilelang.compile(target="ascend910")` produces `kernel.o` and
  `libkernel_launcher.so` through the content-addressed cache.

## Not included

- Device load/launch (`NPUUtils.load_binary`) — requires an NPU host.
- Mixed Cube/Vector codegen for flash attention / sparse MLA is still being
  debugged.
- Autotune profiling and NPU CI runners.
