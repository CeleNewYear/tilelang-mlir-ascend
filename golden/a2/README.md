# A2 (Ascend 910) golden corpus

This directory contains the reference outputs produced by
`tilelang-mlir-ascend` **before** the rebase onto tilelang v0.1.15.  It is the
contract for the `ascend910` backend: codegen must be byte-identical, not merely
"functionally equivalent".

## Contents

- `a2_corpus.tar.gz` — 426 kernels from 70 examples, in both expert
  (`TILELANG_ASCEND_MODE=expert`) and developer (`=developer`) modes.
  Each kernel has:
  - `NNN_<kernel>.mlir` — MLIR handed to `bishengir-compile`
  - `NNN_<kernel>.cmd` — exact compiler argv (temp paths normalized to `{TMP}`)
  - `NNN_<kernel>.launcher.cpp` / `.launcher.h` — generated launch/load sources
- `MANIFEST.json` — SHA-256, byte count, paths, compile status, source commit.
- `smoke/` — a small in-tree subset used by unit tests.

Extract the archive with:

```bash
tar -C golden/a2 -xzf golden/a2_corpus.tar.gz
```

## Provenance

| field | value |
|---|---|
| source repo | `tile-ai/tilelang-mlir-ascend` |
| source commit | see `MANIFEST.json:source_commit` |
| AscendNPU-IR commit | `31f690369d1247fbd5529a3f88b758f7d470ae4f` |
| compiler | CANN 9.3 weekly `tools/bishengir/bin/bishengir-compile` |
| modes | expert, developer |

The corpus was generated without an NPU: the examples were executed with a
device-free `torch_npu` shim that intercepts kernel launches, while the real
frontend, lowering pipeline, codegen, and `bishengir-compile` ran normally.
Device-bound verification is therefore out of scope for the generation step;
the MLIR and compiler argv are real.

## Comparing a candidate implementation

```bash
python tools/compare_a2_golden.py \
  --candidate /path/to/new/corpus \
  --golden golden/a2
```

The candidate directory must have the layout
`<mode>/<example-slug>/NNN_<kernel>.{mlir,cmd}`.  The script normalizes
temporary paths and compares MLIR and compiler argv exactly.

## Regenerating

The generation harness is intentionally not part of the repository (it depends
on a locally installed, pinned `tilelang-mlir-ascend` checkout).  Keep the
original checkout until the new backend passes the comparison, then regenerate
the corpus from the new backend and review the diff.
