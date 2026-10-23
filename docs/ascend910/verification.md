# A2 rebase verification

## Environment

- WSL Ubuntu 22.04, conda env `a2`, Python 3.11
- torch 2.10.0+cpu, torch_npu 2.10.0 (no local Ascend 910)
- CANN 9.3 weekly (`bishengir-compile`, `bishengir-opt`)
- AscendNPU-IR `31f690369d1247fbd5529a3f88b758f7d470ae4f`
- tilelang v0.1.15 base `a35f8ddf`, branch `rebase-v0.1.15`

## Build

```bash
cmake -S . -B build -DUSE_ASCEND=ON -DUSE_ASCEND910=ON \
  -DBISHENGIR_ROOT_PATH=$HOME/xinyang/AscendNPU-IR/build/install
cmake --build build -j16
```

## Tests

```bash
cd /home/xinyang/xinyang/tilelang-0.1.15
source /home/xinyang/xinyang/work/env_a2.sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  python -m pytest testing/ascend910 -q
```

Expected: `19 passed`.

## Golden comparison

`golden/a2/a2_corpus.tar.gz` contains 426 kernels from 70 legacy examples
(expert + developer), with MLIR, bishengir argv, launcher source and headers.

A local batch harness runs each migrated example, canonicalizes the generated
MLIR with `bishengir-opt --canonicalize`, and compares it to the corpus entry.
At the current commit:

- 34 corpus examples have a matching migrated source file
- all 34 emit MLIR and canonicalize
- 11 are byte-identical

Byte-identical:

```text
vec_add_1d
vec_add_2d
vec_add_2d_dynamic_shape
vec_add_2d_multi_buffer
atomic_add
atomic_add_dev
elementwise_exp2
elementwise_log2
ada_layer_norm_and_zero
rms_norm
fp8_lighting_indexer
```

The remaining differences fall into three groups:

1. Allocation/attribute ordering that does not change semantics
   (`matmul`, `matmul_dynamic_shape`, `mixcv`).
2. A2 mixed Cube/Vector codegen: the frontend now emits `resource_scope`
   markers, but the split AIC/AIV path still crashes for
   `flash_attn_npuir` and the sparse-MLA kernels.
3. TileOPs kernels whose golden was produced with a different core-type choice
   (AIC/L1/L0 vs the current AIV/UB path).

## Compile pipeline

`tilelang.compile(..., target="ascend910")` performs:

```text
PrimFunc -> A2 codegen -> kernel.npuir -> bishengir-compile -> kernel.o
         -> generated NPU wrapper -> clang++ -> libkernel_launcher.so
```

Objects and launchers are cached by content hash. Device load/launch is the
only missing stage; it needs `NPUUtils.load_binary` from an NPU host.
