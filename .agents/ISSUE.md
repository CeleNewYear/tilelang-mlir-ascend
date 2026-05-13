# Ascend Compiler / Codegen Issues

Tracked limitations for the TileLang-Ascend team.
Reference from skill `present_limit.md` files.

---

## 1. Dtype constants are FunctionHandle, not DataType
- **File**: `T.Tensor(shape, T.float32)` → `expected DLDataType but got FunctionHandle`
- **Workaround**: use strings `"float32"` / `"float16"`
- **Date**: 2026-05-13

## 2. Atomic_add cbuf→gm unsupported for single-element (Developer mode)
- **File**: `hivm.hir.store` only supports ub→gm for single-element store; contiguous tiles work
- **Workaround**: per-block output buffer + host reduction
- **Date**: 2026-05-13

## 3. Cross-block atomic_add to same element produces wrong results
- **File**: Multiple blocks doing `T.atomic_add` to same global address gives incorrect values
- **Workaround**: per-block output buffer + host reduction
- **Date**: 2026-05-13
