---
name: tilelang-error-fixer
description: TileLang npuir 错误诊断与修复技能。用户提及编译失败、运行错误、pass 异常、结果错误、性能回退、Core Dump、段错误、BishengIR 编译报错、sync 死锁、load/store 维度不一致时必须使用本技能。
---

# TileLang Error Fixer (npuir)

## Mandatory routing rule

Before answering, follow AGENTS.md section "Docs Auto Routing Rules (Mandatory)".

## Scope

- compile errors in npuir path
- runtime failures and invalid results
- pass pipeline divergence
- precision and performance regressions

## Diagnosis workflow

1. Confirm environment and target setting
2. Reproduce with smallest kernel
3. Classify issue type using the table below
4. Capture evidence: logs, IR snapshot, failing stage
5. Propose minimal patch and validate

For errors caused by known Ascend compiler/codegen limits, first check:
`references/present_limit.md` (and `tilelang-vector-skill/references/present_limit.md`).

## Error classification lookup

| Symptom | Likely cause | Fix pattern |
|---------|-------------|-------------|
| Compile failure | v-prefix API misuse, wrong dtype, shape mismatch | Check API signature; verify src/dst shapes; compare with existing example |
| Compile failure (Expert) | load_nd2nz/store_fixpipe size/layout inconsistency | Verify tile sizes match across load/gemm/store; check NZ layout assumptions |
| Runtime crash / Core dump | null buffer access, out-of-bounds, dtype mismatch | Add T.print to isolate crash point; check buffer allocation sizes |
| Runtime hang / timeout | sync deadlock (missing or mismatched sync_block_set/wait) | Verify every sync_block_set has a matching sync_block_wait with same id |
| Precision deviation | Wrong dtype in accumulation path, missing cast | Check fp16 input uses fp32 accumulation; verify cast round modes |
| Performance regression | Redundant copies, excessive casts, suboptimal tile size | Profile block sizes; remove unnecessary copy/cast pairs |
| BishengIR compile error | Pass pipeline failure, invalid MLIR after transform | Dump IR before/after each pass; isolate first failing transform |
| Pass failure | cv-split or vectorize pass misconfiguration | Check pass ordering; verify kernel structure matches pass expectations |

## NPUIR-specific checks

- verify default vector API style uses v-prefix ops
- verify alias callsites are semantically equivalent
- verify load_nd2nz and store_fixpipe size/layout consistency
- verify sync_block_set and sync_block_wait pairing

## Output template

## TileLang JIT Issue Report

### Summary
- Symptom:
- Repro script:
- Impact:

### Root Cause
- Layer: frontend or pass or codegen or runtime
- Fault pattern:

### Fix
- Minimal change:
- Why this fixes it:

### Verification
- Repro after fix:
- Numerical check:
- Regression risk:
