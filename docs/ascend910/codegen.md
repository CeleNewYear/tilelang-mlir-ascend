# Ascend 910 (A2) codegen

The A2 code generator lives in `src/ascend910/target` and is independent of
the Ascend 950 backend in `src/ascend`.

## Build

```bash
cmake -S . -B build \
  -DUSE_ASCEND=ON \
  -DUSE_ASCEND910=ON \
  -DBISHENGIR_ROOT_PATH=/path/to/AscendNPU-IR/build/install
cmake --build build -j16
```

`USE_ASCEND910` gates only the A2 code generator.  The target kind and the
Python package are always available.  The build links the AscendNPU-IR MLIR
static libraries (`BISHENGIR_ROOT_PATH/lib/*.a`) into `libtilelang.so`.

## Entry points

| global function | mode |
|---|---|
| `target.build.tilelang_ascend910` | expert (`T.npuir_*`, `T.region`) |
| `target.build.tilelang_ascend910_dev` | developer |

The Python facade selects the mode from `TILELANG_ASCEND_MODE` (default
`expert`; `dev`/`developer` selects the other):

```python
from tvm.target import Target
from tilelang.ascend910 import ASCEND910_CODEGEN, bishengir

mod = ASCEND910_CODEGEN.lower(mod, Target("ascend910"), compile_device=False)
mlir = bishengir.canonicalize_mlir(mod.inspect_source())
```

## tirx migration notes

- `tvm/tir` headers and `tir::` names are now `tvm/tirx` / `tirx::`.
- `LetStmt` became `Bind`; `Allocate` became `AllocBuffer`; `DeclBuffer` no
  longer has a body.
- `Array`/`Map`/`String` come from `tvm::ffi`.
- Modules are `ffi::Module`; registration uses
  `TVM_FFI_STATIC_INIT_BLOCK` + `ffi::Function::SetGlobal`.
- `String2DLDataType` became `ffi::StringToDLDataType`.

## Verification

`testing/ascend910/test_ascend910_codegen_tir.py` feeds a captured legacy TIR
snapshot through both emitters, canonicalizes with `bishengir-opt`, and asserts
byte equality with the golden MLIR for `vec_add_1d` in both modes.
