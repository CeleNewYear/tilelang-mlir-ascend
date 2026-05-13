# TileLang npuir Agent Guide

This repository uses AGENTS skills for TileLang NPUIR development.

## Scope

The skills in .agents/skills are designed for target="npuir" workflows.
They prioritize v-prefix vector APIs such as vadd, vmul, vexp, vcast, vbrc.
Legacy npuir_xxx APIs remain valid as compatibility aliases.

## API Convention (Mandatory)

- Prefer v-prefix APIs in new examples and generated code.
- Keep compatibility with npuir_xxx when reading existing code.
- If both forms are available, output should default to v-prefix.

Examples:
- Prefer: T.vmul(A, B, C)
- Compatible: T.npuir_mul(A, B, C)

## Mode Convention (Mandatory)

- Developer mode uses T.alloc_shared / T.alloc_fragment; compiler manages memory.
- Expert mode uses explicit memory (T.alloc_ub / T.alloc_L1 / T.alloc_L0C) and explicit Scope blocks.
- **When the user asks to write a kernel or port a GPU kernel to NPU without specifying Developer or Expert mode, the agent MUST ask the user which mode to use before generating any code. Never assume a default.**

## Reference Verification Rule (Mandatory)

When porting a GPU kernel without a torch reference:
1. Write a torch CPU/GPU reference implementation first
2. Present it to the user for correctness verification
3. Wait for user confirmation before proceeding to Ascend kernel
4. Once confirmed, the reference is locked — do NOT auto-change it without user approval

## Issue Tracking Rule

When encountering an Ascend compiler/codegen/API limitation:
- Record it in `.agents/ISSUE.md` (for compiler team)
- Also add workaround to the relevant skill's `references/present_limit.md`

## Iteration Checkpoint Rule

1. If a single compile/runtime error persists for 3 consecutive fix attempts, STOP and ask the user.
2. At debug-loop start, ask: "How many iterations before I check in?" (default: 3).
   User can override per task.

## Kernel Artifact Rule

After every successful iteration (compile+run+pass), the working kernel MUST be preserved on disk.
Do NOT overwrite a passing kernel without user approval.

## Skill Index

1. tilelang-npuir-overview
Purpose: architecture and compile pipeline for npuir branch.

2. tilelang-vector-skill
Purpose: vector operator generation with v-prefix API style. Covers both Developer and Expert mode.

3. tilelang-cube-skill
Purpose: cube operator generation with load_nd2nz and store_fixpipe. Covers both Developer and Expert mode.

4. tilelang-mixcv-skill
Purpose: mixed Cube+Vector kernels such as flash attention pipelines.

5. tilelang-mlir-skill
Purpose: TileLangIR and MLIR pass workflow and debugging.

6. tilelang-debug-helper
Purpose: step-by-step debug workflow: T.print → IR dump → GDB for npuir.

7. tilelang-error-fixer
Purpose: categorized diagnosis and repair for compile/runtime/pass/precision/performance failures.

8. tilelang-review-skill
Purpose: risk-first code review, format checks, commit/PR/issue workflow.

9. tilelang-remote-runner
Purpose: remote Ascend kernel verification via SSH.

## Trigger Guidance

Use the matching skill whenever the user asks for:
- npuir kernel writing, performance tuning, vector math, cube gemm, mixed kernels
- pass debugging, IR dump, MLIR transform troubleshooting
- compile/runtime error analysis on npuir branch
- code review, lint/format checks, PR readiness, commit/push/rebase
- remote kernel verification on Ascend hardware

Developer-mode MixCV trigger rule:
- If one kernel contains Cube-side T.gemm and Vector-side at least one v-prefix op (such as T.vadd/T.vmul/T.vexp/T.vcast/T.vbrc), treat it as MixCV and use tilelang-mixcv-skill.

## Operator Implementation Baseline (Mandatory)

For operator-writing tasks, always start from existing examples and tests:

- First consult examples/ and testing/npuir/ for the closest existing pattern.
- Prefer modifying an existing operator case instead of generating a brand-new kernel from scratch.
- If no close template exists, explicitly state that and then build the minimal new kernel.

## Pre-PR Formatting Rule (Mandatory)

Before creating or updating a PR, run format validation for changed files from repository root:

- bash format.sh --files changed_files

Notes:
- This is a required self-check for clean code and style consistency.
- The changed_files placeholder represents the file list modified in the current branch.

## Docs Auto Routing Rules (Mandatory)

When any skill answers technical questions, it must route references by docs directory first.

### Routing Priority

1. docs/Tilelang.language/ (API semantics and signatures)
2. docs/Tilelang算子调试指南.md (debug and issue localization)
3. docs/developer/ (runtime and environment variables)
4. docs/开发指南.md and docs/快速入门.md (workflow and onboarding)
5. docs/Tilelang-Ascend贡献指南.md (PR, issue, contribution process)

### Keyword to Docs Mapping

- Vector ops (vadd/vmul/vexp/vcast/vbrc/reduce/sigmoid/rmsnorm):
    docs/Tilelang.language/数学操作/
    docs/Tilelang.language/数据类型转换操作/
    docs/Tilelang.language/shape操作/
    docs/Tilelang.language/规约操作/

- Cube ops (gemm/load_nd2nz/store_fixpipe/L1/L0C/NZ):
    docs/Tilelang.language/线性代数操作/
    docs/Tilelang.language/内存操作/

- Pipeline and sync (sync_block_set/wait/pipe_barrier/set_flag/wait_flag):
    docs/Tilelang.language/同步管道操作/

- Debug, compile failure, runtime failure, precision issue:
    docs/Tilelang算子调试指南.md
    docs/Tilelang.language/调试操作/

- MLIR, pass, tilelangir, bishengir-compile:
    docs/Tilelang算子调试指南.md
    docs/developer/EnvironmentVariables.md

- Runtime target, mode switch, env setup:
    docs/developer/npu runtime.md
    docs/developer/EnvironmentVariables.md
    docs/安装指南.md

- PR, rebase, commit, issue, CI workflow:
    docs/Tilelang-Ascend贡献指南.md

### Conflict Resolution

- If multiple mappings match, select by priority and keep at most 3 primary doc references.
- Always include at least 1 concrete API doc under docs/Tilelang.language/ when the question is API-related.
- If API docs and examples differ, API docs are source of truth and examples are secondary.
