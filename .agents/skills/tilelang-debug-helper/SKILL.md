---
name: tilelang-debug-helper
description: TileLang npuir 调试辅助技能。用户提及调试 npuir kernel、GDB 附加、IR dump、精度异常定位、编译失败定位、pass 阶段定位、T.print 调试、最小复现缩减时必须使用本技能。
---

# TileLang Debug Helper (npuir)

## Mandatory routing rule

Before answering, follow AGENTS.md section "Docs Auto Routing Rules (Mandatory)".

## Debug escalation ladder

Always start from the top; only escalate when the current level is insufficient:

| Level | Tool | When to use |
|-------|------|-------------|
| 1 | `T.print(buf)` | First line: inspect intermediate values inside the kernel |
| 2 | IR dump | When print reveals wrong values: capture IR snapshots to find pass/transform issues |
| 3 | GDB attach | When IR is correct but runtime crashes: native debug for segfault/core dump |

## Level 1 — T.print debugging

- Insert `T.print(buffer)` at key points: after load, after compute, after store
- Compare printed values against torch reference at each stage
- Identify which operation produces the first wrong value

## Level 2 — IR dump debugging

- Set `os.environ["TILELANG_DUMP_IR"] = "1"` before JIT compilation
- Capture IR before and after major pass stages
- Compare operation-level diffs to isolate the first failing transformation
- Common checkpoints: after lower entry, after tilelangir pass application, before backend codegen

## Level 3 — GDB attach

- Use only when T.print and IR dump cannot resolve the issue
- Attach to the running process and inspect native codegen output

## For API debugging

- First verify v-prefix API usage
- Then verify alias compatibility if legacy npuir_xxx appears

## References

- references/mlir-dump-guide.md

## Official docs to consult

- docs/Tilelang算子调试指南.md
- docs/Tilelang.language/调试操作/T.print.md
- docs/developer/EnvironmentVariables.md

## Related skills

- tilelang-mlir-skill
- tilelang-error-fixer
