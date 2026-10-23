module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @main(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    %c1_i32 = arith.constant 1 : i32
    %c20_i32 = arith.constant 20 : i32
    %c19_i32 = arith.constant 19 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c0_i32 = arith.constant 0 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg6 : i32 to index
    %1 = arith.index_cast %arg7 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, %1], strides: [%1, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%0, %1], strides: [%1, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0, %1], strides: [%1, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, 1]>, #hivm.address_space<gm>>
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.addi %arg6, %c31_i32 : i32
    %5 = arith.divsi %4, %c32_i32 : i32
    %6 = arith.addi %arg7, %c31_i32 : i32
    %7 = arith.divsi %6, %c32_i32 : i32
    %8 = arith.muli %5, %7 : i32
    %9 = arith.addi %8, %c19_i32 : i32
    %10 = arith.divsi %9, %c20_i32 : i32
    scf.for %arg14 = %c0_i32 to %10 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<32x32xf16, strided<[32, 1]>, #hivm.address_space<ub>>
      %alloc_2 = memref.alloc() : memref<32x32xf16, strided<[32, 1]>, #hivm.address_space<ub>>
      %alloc_3 = memref.alloc() : memref<32x32xf16, strided<[32, 1]>, #hivm.address_space<ub>>
      %11 = arith.addi %arg6, %c31_i32 : i32
      %12 = arith.divsi %11, %c32_i32 : i32
      %13 = arith.addi %arg7, %c31_i32 : i32
      %14 = arith.divsi %13, %c32_i32 : i32
      %15 = arith.muli %12, %14 : i32
      %16 = arith.addi %15, %c19_i32 : i32
      %17 = arith.divsi %16, %c20_i32 : i32
      %18 = arith.muli %3, %17 : i32
      %19 = arith.addi %18, %arg14 : i32
      %20 = arith.cmpi slt, %19, %15 : i32
      scf.if %20 {
        %21 = arith.addi %arg6, %c31_i32 : i32
        %22 = arith.divsi %21, %c32_i32 : i32
        %23 = arith.addi %arg7, %c31_i32 : i32
        %24 = arith.divsi %23, %c32_i32 : i32
        %25 = arith.muli %22, %24 : i32
        %26 = arith.addi %25, %c19_i32 : i32
        %27 = arith.divsi %26, %c20_i32 : i32
        %28 = arith.muli %3, %27 : i32
        %29 = arith.addi %28, %arg14 : i32
        %30 = arith.divsi %29, %24 : i32
        %31 = arith.muli %30, %c32_i32 : i32
        %32 = arith.index_cast %31 : i32 to index
        %33 = arith.subi %arg6, %31 : i32
        %34 = arith.minsi %33, %c32_i32 : i32
        %35 = arith.index_cast %34 : i32 to index
        %36 = arith.remsi %29, %24 : i32
        %37 = arith.muli %36, %c32_i32 : i32
        %38 = arith.index_cast %37 : i32 to index
        %39 = arith.subi %arg7, %37 : i32
        %40 = arith.minsi %39, %c32_i32 : i32
        %41 = arith.index_cast %40 : i32 to index
        %subview = memref.subview %reinterpret_cast[%32, %38] [%35, %41] [1, 1] : memref<?x?xf16, strided<[?, 1]>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %reinterpret_cast_4 = memref.reinterpret_cast %alloc to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<32x32xf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<?x?xf16, strided<[32, 1]>, #hivm.address_space<ub>>
        memref.copy %subview, %reinterpret_cast_4 : memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %subview_5 = memref.subview %reinterpret_cast_1[%32, %38] [%35, %41] [1, 1] : memref<?x?xf16, strided<[?, 1]>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %reinterpret_cast_6 = memref.reinterpret_cast %alloc_2 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<32x32xf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<?x?xf16, strided<[32, 1]>, #hivm.address_space<ub>>
        memref.copy %subview_5, %reinterpret_cast_6 : memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[32, 1]>, #hivm.address_space<ub>>
        hivm.hir.vadd ins(%alloc, %alloc_2 : memref<32x32xf16, strided<[32, 1]>, #hivm.address_space<ub>>, memref<32x32xf16, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_3 : memref<32x32xf16, strided<[32, 1]>, #hivm.address_space<ub>>)
        %subview_7 = memref.subview %alloc_3[0, 0] [%35, %41] [1, 1] : memref<32x32xf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<?x?xf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %subview_8 = memref.subview %reinterpret_cast_0[%32, %38] [%35, %41] [1, 1] : memref<?x?xf16, strided<[?, 1]>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_7, %subview_8 : memref<?x?xf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      }
    }
    return
  }
}