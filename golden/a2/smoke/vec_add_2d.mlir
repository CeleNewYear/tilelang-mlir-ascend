module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @main(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    %c128_i32 = arith.constant 128 : i32
    %c4_i32 = arith.constant 4 : i32
    %c16_i32 = arith.constant 16 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [512, 512], strides: [512, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<512x512xf16, strided<[512, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [512, 512], strides: [512, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<512x512xf16, strided<[512, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [512, 512], strides: [512, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<512x512xf16, strided<[512, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %alloc = memref.alloc() : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
    %alloc_3 = memref.alloc() : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
    %2 = arith.cmpi slt, %1, %c16_i32 : i32
    scf.if %2 {
      %3 = arith.divsi %1, %c4_i32 : i32
      %4 = arith.muli %3, %c128_i32 : i32
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.remsi %1, %c4_i32 : i32
      %7 = arith.muli %6, %c128_i32 : i32
      %8 = arith.index_cast %7 : i32 to index
      %subview = memref.subview %reinterpret_cast[%5, %8] [128, 128] [1, 1] : memref<512x512xf16, strided<[512, 1]>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<128x128xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %subview_4 = memref.subview %reinterpret_cast_1[%5, %8] [128, 128] [1, 1] : memref<512x512xf16, strided<[512, 1]>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_4, %alloc_2 : memref<128x128xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      hivm.hir.vadd ins(%alloc, %alloc_2 : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>, memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_3 : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>)
      %subview_5 = memref.subview %reinterpret_cast_0[%5, %8] [128, 128] [1, 1] : memref<512x512xf16, strided<[512, 1]>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %alloc_3, %subview_5 : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<128x128xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}