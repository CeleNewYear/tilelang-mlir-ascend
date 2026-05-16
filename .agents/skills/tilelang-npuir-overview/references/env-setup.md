# Environment Setup Notes

## Recommended install entry

Use `install_npuir.sh` for npuir workflow.

### Fast build with prebuilt AscendNPU-IR

If AscendNPU-IR is already compiled and installed at a known path, use
`--bishengir-path` to skip the lengthy 3rdparty clone+compile:

```bash
bash install_npuir.sh --bishengir-path=/home/<user>/AscendNPUIR/AscendNPU-IR/build/install
```

This avoids cloning the full `3rdparty/AscendNPU-IR` tree and all its recursive
submodules (can save 30+ minutes).  **Always ask the user for their prebuilt
path before running the install script.**

### Full build (fallback)

When no prebuilt BishengIR is available:

```bash
bash install_npuir.sh
```

This will clone and build AscendNPU-IR from `3rdparty/` including all recursive
submodules (catlass, composable_kernel, cutlass, tvm, flashinfer, ...).

## Basic verification

- Python environment activated
- NPU toolchain paths available
- target set to npuir in JIT entry

## Runtime hygiene

- clear tilelang cache when validating kernel changes
- keep sample scripts minimal and reproducible
