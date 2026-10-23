// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file src/ascend910/target/rt_mod_ascend910.cc
 * \brief Runtime module builders for the Ascend 910 (A2) codegen.
 */

#include "codegen_npuir_api.h"
#include "codegen_npuir_dev.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/runtime/module.h>

namespace tvm {
namespace codegen {

namespace {

Array<String> CollectFunctions(IRModule mod) {
  Array<String> names;
  for (auto kv : mod->functions) {
    names.push_back(Downcast<GlobalVar>(kv.first)->name_hint);
  }
  return names;
}

} // namespace

ffi::Module BuildTileLangAscend910Expert(IRModule mod, Target target) {
  CodeGenTileLangNPUIRAPI cg;
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangNPUIRAPI: Can only take PrimFunc";
    cg.AddFunction(Downcast<GlobalVar>(kv.first), Downcast<PrimFunc>(kv.second));
  }
  std::string mlir = cg.Finish();
  return CSourceModuleCreate(mlir, "c", CollectFunctions(mod));
}

ffi::Module BuildTileLangAscend910Developer(IRModule mod, Target target) {
  CodeGenTileLangNPUIRDEV cg;
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangNPUIRDEV: Can only take PrimFunc";
    cg.AddFunction(Downcast<GlobalVar>(kv.first), Downcast<PrimFunc>(kv.second));
  }
  std::string mlir = cg.Finish();
  return CSourceModuleCreate(mlir, "c", CollectFunctions(mod));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::Function::SetGlobal(
      "target.build.tilelang_ascend910",
      ffi::Function::FromTyped(BuildTileLangAscend910Expert));
  ffi::Function::SetGlobal(
      "target.build.tilelang_ascend910_dev",
      ffi::Function::FromTyped(BuildTileLangAscend910Developer));
}

} // namespace codegen
} // namespace tvm
