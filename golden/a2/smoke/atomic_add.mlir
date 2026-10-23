module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @main(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf32, #hivm.address_space<gm>>, %arg4: memref<?xf32, #hivm.address_space<gm>>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    %c32_i32 = arith.constant 32 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %alloc = memref.alloc() : memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
    %2 = arith.muli %1, %c32_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %subview = memref.subview %reinterpret_cast[%3] [32] [1] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview, %alloc : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %reinterpret_cast_0[%3] [32] [1] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%alloc : memref<32xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_1 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) atomic = <add>
    return
  }
}