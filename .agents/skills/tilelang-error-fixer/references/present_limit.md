# Error Patterns Caused by Ascend Realization Limits

Match a symptom below → apply the workaround.
Delete entries as the compiler fixes them.

---

| Symptom snippet | Limit | Fix |
|----------------|-------|-----|
| `expected DLDataType but got FunctionHandle` | Dtype constant not recognized | Use string `"float32"` / `"float16"` |
| `'float' object has no attribute 'buffer'` at `atomic_add` | Scalar src in atomic_add | Use 1-element buffer accumulator |
| `store only support ub to gm` at codegen | cbuf→gm single-element atomic_add | Per-block output + host reduction (contiguous tiles work) |
| `float() argument must be a string or real number, not 'Var'` | Python cast on TVM Var | Use `T.cast(var, "float32")` |
| Cross-block atomic_add gives wrong values | Multiple blocks → same output element | Per-block output + host reduction |
