---
name: tilelang-cube-skill
description: TileLang npuir Cube 算子开发指南。用户提及 GEMM、matmul、batch gemm、L1/L0C、load_nd2nz、store_fixpipe、NZ 格式、Cube scope、矩阵分块与流水优化时必须使用本技能。
---

# TileLang Cube Skill

## Mandatory routing rule

Before answering, follow AGENTS.md section "Docs Auto Routing Rules (Mandatory)".

## Mode-asking rule (Mandatory)

When the user asks to write a new cube kernel or port a GPU GEMM to NPU without specifying Developer or Expert mode, you MUST ask the user which mode to use before generating any code. Never assume a default mode.

## Operator baseline rule (Mandatory)

- Before writing a new cube operator, first check examples/ and testing/npuir/.
- Prefer adapting an existing operator case rather than writing from scratch.

## Primary use cases

- matmul and batched matmul kernels
- cube-heavy stages in mixed kernels

## Mode-specific memory and data movement

| Aspect | Developer mode | Expert mode |
|--------|---------------|-------------|
| Input memory | `T.alloc_shared(shape, dtype)` | `T.alloc_L1(shape, dtype)` |
| Accumulator | `T.alloc_fragment(shape, accum_dtype)` | `T.alloc_L0C(shape, accum_dtype)` |
| Load data | `T.copy(src, dst)` | `T.load_nd2nz(src, dst, size)` |
| Store data | `T.copy(C_buf, C_out)` | `T.store_fixpipe(C_buf, C_out, size=[M,N], enable_nz2nd=True)` |
| Layout | ND tensors throughout | ND → NZ (load) → ND (store) |
| Scope | No explicit scope needed | `T.Scope("Cube")` required |

## Core APIs

- T.gemm(A, B, C, initC=True or False, b_transpose=True or False, size=[M, K, N])

## Minimal flow

1. Ask user for mode if not specified
2. Partition blocks for M and N
3. **Developer**: alloc_shared → T.copy in → T.gemm → T.copy out
4. **Expert**: alloc_L1 → load_nd2nz → T.gemm → store_fixpipe
5. K-loop with `initC=(k==0)` for accumulation
6. Validate against torch reference

## Data type safety (Mandatory)

- For fp16 input GEMM, destination/accumulation must use fp32.
- Setting destination to fp16 for fp16 input GEMM may cause runtime hang.

## NZ format rule

- NZ format path is Expert mode only.
- In Developer mode kernels, keep ND layout and use T.copy-based data movement.

## References

- references/api-cube.md
- references/examples-matmul.md
- references/nz-format.md

## Example entry points

- examples/gemm/example_gemm.py
- examples/gemm/example_gemm_int82int32.py
- examples/gemm/matmul.py
- examples/gemm/matmul_dynamic_shape.py

## Official docs to consult

- docs/Tilelang.language/线性代数操作/T.gemm.md
- docs/Tilelang.language/内存操作/T.alloc_shared.md
- docs/Tilelang.language/内存操作/T.load_nd2nz.md
- docs/Tilelang.language/内存操作/T.store_fixpipe.md
- docs/Tilelang.language/内存操作/T.alloc_L1.md
- docs/Tilelang.language/内存操作/T.alloc_L0C.md

## Related skills

- tilelang-vector-skill
- tilelang-mixcv-skill
- tilelang-debug-helper
