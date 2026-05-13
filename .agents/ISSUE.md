# Ascend Compiler / Codegen Issues

Tracked limitations for the TileLang-Ascend team.
Reference from skill `present_limit.md` files.

---

## 1. Dtype constants are FunctionHandle, not DataType
- **File**: `T.Tensor(shape, T.float32)` → `expected DLDataType but got FunctionHandle`
- **Workaround**: use strings `"float32"` / `"float16"`
- **Date**: 2026-05-13

## 2. Atomic_add cbuf→gm unsupported in Developer mode
- **File**: `hivm.hir.store` only supports ub→gm
- **Workaround**: per-block output buffer + host reduction
- **Date**: 2026-05-13
