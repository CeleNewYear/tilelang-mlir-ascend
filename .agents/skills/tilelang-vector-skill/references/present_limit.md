# Present Ascend Limitations (Vector)

Delete entries here as the compiler/codegen catches up.

---

## 1. Dtype constants are FunctionHandle, not DataType

- **Limit**: `T.Tensor(shape, T.float32)` / `T.float16` / `T.int32` / `T.int64` fail with
  `expected DLDataType but got FunctionHandle`
- **Workaround**: Always use Python strings: `T.Tensor(shape, "float32")`, `"float16"`
- **Scope**: All Tensor declarations inside `@T.prim_func`
- **Date**: 2026-05-13

## 2. Ascend tensors only support float16/float32

- **Limit**: int32/int64 tensor buffers are not available on Ascend
- **Workaround**: Use float32 for all tensor data; cast to needed type inside kernel with `T.cast()`
- **Date**: 2026-05-13

## 3. `float()` / `int()` cannot cast TVM loop variables

- **Limit**: `float(e)` or `int(e)` on a TVM `Var` (loop iterator) fails with
  `float() argument must be a string or a real number, not 'Var'`
- **Workaround**: Use `T.cast(e, "float32")` or `T.cast(e, "int32")`
- **Date**: 2026-05-13

## 4. Atomic_add requires buffer-to-buffer, not scalar

- **Limit**: `T.atomic_add(dst[idx], 1.0)` fails — `'float' object has no attribute 'buffer'`
- **Workaround**: Create a 1-element buffer:
  ```python
  count = T.alloc_shared((1,), "float32")
  T.clear(count)
  count[0] += 1.0  # accumulate, then
  T.atomic_add(dst[idx], count)
  ```
- **Date**: 2026-05-13

## 5. Atomic_add from shared memory (cbuf) to global memory (gm) — contiguous tiles work; single-element fails

- **Limit**: In Developer mode, `T.atomic_add` from shared buffer to a single global element fails with
  `hivm.hir.store op only support store ub to gm currently`.
  Contiguous-tile atomic_add from shared memory **works** for 1D/2D/3D tiles.
- **Workaround for single-element**: Write per-block results to a 2D output buffer (`[num_blocks, ...]`),
  then reduce on host with `torch.sum(dim=0)`
- **Date**: 2026-05-13

## 6. Cross-block atomic_add to the same output element produces wrong results

- **Limit**: Multiple Ascend blocks doing `T.atomic_add` to the same global output element
  produces incorrect numerical results (even in Expert mode with UB→GM).
  Cross-block atomic_add to different output positions works correctly.
- **Workaround**: Per-block output buffer + host reduction
- **Date**: 2026-05-13
