module @softmax attributes {gpu.container_module} {
  memref.global "private" constant @__constant_32x256xf32 : memref<32x256xf32> = dense<5.000000e-01>
  func.func @main() {
    %0 = memref.get_global @__constant_32x256xf32 : memref<32x256xf32>
    %1 = call @test(%0) : (memref<32x256xf32>) -> memref<32x256xf32>
    return
  }
func.func private @printMemrefF32(memref<*xf32>)
  func.func @test(%arg0: memref<32x256xf32>) -> memref<32x256xf32> {
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %memref = gpu.alloc  host_shared () : memref<32x256xf32>
    memref.copy %arg0, %memref : memref<32x256xf32> to memref<32x256xf32>
    %memref_0 = gpu.alloc  host_shared () : memref<32x256xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c32, %c1, %c1) threads in (%c16, %c1, %c1) args(%memref : memref<32x256xf32>, %cst : f32, %memref_0 : memref<32x256xf32>)
    gpu.dealloc  %memref : memref<32x256xf32>
    return %memref_0 : memref<32x256xf32>
  }
  //spirv.module @__spv__test_kernel Physical64 OpenCL attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.module @__spv__test_kernel Physical64 OpenCL requires  #spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, Groups, SubgroupDispatch, SubgroupBufferBlockIOINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_subgroups]> {
    spirv.GlobalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, %arg1: f32, %arg2: !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>) "None" attributes {gpu.known_block_size = array<i32: 16, 1, 1>, gpu.known_grid_size = array<i32: 32, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>, workgroup_attributions = 0 : i64} {
      %cst16_i64 = spirv.Constant 16 : i64
      %cst1024_i64 = spirv.Constant 1024 : i64
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
      %__builtin_var_LocalInvocationId___addr = spirv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spirv.ptr<vector<3xi64>, Input>
      %2 = spirv.Load "Input" %__builtin_var_LocalInvocationId___addr : vector<3xi64>
      %3 = spirv.CompositeExtract %2[0 : i32] : vector<3xi64>
      %4 = spirv.IMul %1, %cst1024_i64 : i64
      %5 = spirv.IAdd %4, %3 : i64
      %6 = spirv.AccessChain %arg0[%5] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      %7 = spirv.IAdd %3, %cst16_i64 : i64
      %8 = spirv.Load "CrossWorkgroup" %6 : f32
      %9 = spirv.IAdd %4, %7 : i64
      %10 = spirv.AccessChain %arg0[%9] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      %11 = spirv.IAdd %7, %cst16_i64 : i64
      %12 = spirv.Load "CrossWorkgroup" %10 : f32
      %13 = spirv.IAdd %4, %11 : i64
      %14 = spirv.AccessChain %arg0[%13] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      %15 = spirv.IAdd %11, %cst16_i64 : i64
      %16 = spirv.Load "CrossWorkgroup" %14 : f32
      %17 = spirv.IAdd %4, %15 : i64
      %18 = spirv.AccessChain %arg0[%17] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      %19 = spirv.IAdd %15, %cst16_i64 : i64
      %20 = spirv.Load "CrossWorkgroup" %18 : f32
      %21 = spirv.IAdd %4, %19 : i64
      %22 = spirv.AccessChain %arg0[%21] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      %23 = spirv.IAdd %19, %cst16_i64 : i64
      %24 = spirv.Load "CrossWorkgroup" %22 : f32
      %25 = spirv.IAdd %4, %23 : i64
      %26 = spirv.AccessChain %arg0[%25] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      %27 = spirv.IAdd %23, %cst16_i64 : i64
      %28 = spirv.Load "CrossWorkgroup" %26 : f32
      %29 = spirv.IAdd %4, %27 : i64
      %30 = spirv.AccessChain %arg0[%29] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      %31 = spirv.IAdd %27, %cst16_i64 : i64
      %32 = spirv.Load "CrossWorkgroup" %30 : f32
      %33 = spirv.IAdd %4, %31 : i64
      %34 = spirv.AccessChain %arg0[%33] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      %35 = spirv.IAdd %31, %cst16_i64 : i64
      %36 = spirv.Load "CrossWorkgroup" %34 : f32
      %37 = spirv.IAdd %4, %35 : i64
      %38 = spirv.AccessChain %arg0[%37] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      %39 = spirv.IAdd %35, %cst16_i64 : i64
      %40 = spirv.Load "CrossWorkgroup" %38 : f32
      %41 = spirv.IAdd %4, %39 : i64
      %42 = spirv.AccessChain %arg0[%41] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      %43 = spirv.IAdd %39, %cst16_i64 : i64
      %44 = spirv.Load "CrossWorkgroup" %42 : f32
      %45 = spirv.IAdd %4, %43 : i64
      %46 = spirv.AccessChain %arg0[%45] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      %47 = spirv.IAdd %43, %cst16_i64 : i64
      %48 = spirv.Load "CrossWorkgroup" %46 : f32
      %49 = spirv.IAdd %4, %47 : i64
      %50 = spirv.AccessChain %arg0[%49] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      %51 = spirv.IAdd %47, %cst16_i64 : i64
      %52 = spirv.Load "CrossWorkgroup" %50 : f32
      %53 = spirv.IAdd %4, %51 : i64
      %54 = spirv.AccessChain %arg0[%53] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      %55 = spirv.IAdd %51, %cst16_i64 : i64
      %56 = spirv.Load "CrossWorkgroup" %54 : f32
      %57 = spirv.IAdd %4, %55 : i64
      %58 = spirv.AccessChain %arg0[%57] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      %59 = spirv.IAdd %55, %cst16_i64 : i64
      %60 = spirv.Load "CrossWorkgroup" %58 : f32
      %61 = spirv.IAdd %4, %59 : i64
      %62 = spirv.AccessChain %arg0[%61] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      %63 = spirv.IAdd %59, %cst16_i64 : i64
      %64 = spirv.Load "CrossWorkgroup" %62 : f32
      %65 = spirv.IAdd %4, %63 : i64
      %66 = spirv.AccessChain %arg0[%65] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      %67 = spirv.Load "CrossWorkgroup" %66 : f32
      %68 = spirv.CompositeConstruct %8, %12, %16, %20, %24, %28, %32, %36, %40, %44, %48, %52, %56, %60, %64, %67 : (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) -> vector<16xf32>
      %69 = spirv.CL.fmax %8, %12 : f32
      %70 = spirv.CL.fmax %69, %16 : f32
      %71 = spirv.CL.fmax %70, %20 : f32
      %72 = spirv.CL.fmax %71, %24 : f32
      %73 = spirv.CL.fmax %72, %28 : f32
      %74 = spirv.CL.fmax %73, %32 : f32
      %75 = spirv.CL.fmax %74, %36 : f32
      %76 = spirv.CL.fmax %75, %40 : f32
      %77 = spirv.CL.fmax %76, %44 : f32
      %78 = spirv.CL.fmax %77, %48 : f32
      %79 = spirv.CL.fmax %78, %52 : f32
      %80 = spirv.CL.fmax %79, %56 : f32
      %81 = spirv.CL.fmax %80, %60 : f32
      %82 = spirv.CL.fmax %81, %64 : f32
      %83 = spirv.CL.fmax %82, %67 : f32
      %84 = spirv.GroupFMax <Subgroup> <Reduce> %83 : f32
      %85 = spirv.CompositeConstruct %84, %84, %84, %84, %84, %84, %84, %84, %84, %84, %84, %84, %84, %84, %84, %84 : (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) -> vector<16xf32>
      %86 = spirv.FSub %68, %85 : vector<16xf32>
      %87 = spirv.CL.exp %86 : vector<16xf32>
      %88 = spirv.CompositeExtract %87[0 : i32] : vector<16xf32>
      %89 = spirv.CompositeExtract %87[1 : i32] : vector<16xf32>
      %90 = spirv.CompositeExtract %87[2 : i32] : vector<16xf32>
      %91 = spirv.CompositeExtract %87[3 : i32] : vector<16xf32>
      %92 = spirv.CompositeExtract %87[4 : i32] : vector<16xf32>
      %93 = spirv.CompositeExtract %87[5 : i32] : vector<16xf32>
      %94 = spirv.CompositeExtract %87[6 : i32] : vector<16xf32>
      %95 = spirv.CompositeExtract %87[7 : i32] : vector<16xf32>
      %96 = spirv.CompositeExtract %87[8 : i32] : vector<16xf32>
      %97 = spirv.CompositeExtract %87[9 : i32] : vector<16xf32>
      %98 = spirv.CompositeExtract %87[10 : i32] : vector<16xf32>
      %99 = spirv.CompositeExtract %87[11 : i32] : vector<16xf32>
      %100 = spirv.CompositeExtract %87[12 : i32] : vector<16xf32>
      %101 = spirv.CompositeExtract %87[13 : i32] : vector<16xf32>
      %102 = spirv.CompositeExtract %87[14 : i32] : vector<16xf32>
      %103 = spirv.CompositeExtract %87[15 : i32] : vector<16xf32>
      %104 = spirv.FAdd %88, %89 : f32
      %105 = spirv.FAdd %104, %90 : f32
      %106 = spirv.FAdd %105, %91 : f32
      %107 = spirv.FAdd %106, %92 : f32
      %108 = spirv.FAdd %107, %93 : f32
      %109 = spirv.FAdd %108, %94 : f32
      %110 = spirv.FAdd %109, %95 : f32
      %111 = spirv.FAdd %110, %96 : f32
      %112 = spirv.FAdd %111, %97 : f32
      %113 = spirv.FAdd %112, %98 : f32
      %114 = spirv.FAdd %113, %99 : f32
      %115 = spirv.FAdd %114, %100 : f32
      %116 = spirv.FAdd %115, %101 : f32
      %117 = spirv.FAdd %116, %102 : f32
      %118 = spirv.FAdd %117, %103 : f32
      %119 = spirv.GroupFAdd <Subgroup> <Reduce> %118 : f32
      //%120 = spirv.CompositeConstruct %119, %119, %119, %119, %119, %119, %119, %119, %119, %119, %119, %119, %119, %119, %119, %119 : (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) -> vector<16xf32>
      //%121 = spirv.FDiv %87, %120 : vector<16xf32>
      %one = spirv.Constant 1.0 : f32
      %div = spirv.FDiv %one, %119 : f32
      %com = spirv.CompositeConstruct %div, %div, %div, %div, %div, %div, %div, %div, %div, %div, %div, %div, %div, %div, %div, %div : (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) -> vector<16xf32>
      %121 = spirv.FMul %87, %com : vector<16xf32>
      %122 = spirv.AccessChain %arg2[%5] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      %123 = spirv.Bitcast %122 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<vector<16xf32>, CrossWorkgroup>
      spirv.Store "CrossWorkgroup" %123, %121 : vector<16xf32>
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__, @__builtin_var_LocalInvocationId__
    spirv.ExecutionMode @test_kernel "SubgroupSize", 16
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<32x256xf32>, %arg1: f32, %arg2: memref<32x256xf32>) kernel attributes {gpu.known_block_size = array<i32: 16, 1, 1>, gpu.known_grid_size = array<i32: 32, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = vector.transfer_read %arg0[%0, %1], %arg1 {in_bounds = [true], mode = "simt"} : memref<32x256xf32>, vector<16xf32>
      %3 = vector.reduction <maxf>, %2 : vector<16xf32> into f32
      //%4 = gpu.subgroup_reduce  max %3 uniform : (f32) -> f32
      %5 = vector.broadcast %3 : f32 to vector<16xf32>
      %6 = arith.subf %2, %5 : vector<16xf32>
      %7 = math.exp %6 : vector<16xf32>
      %8 = vector.reduction <add>, %7 : vector<16xf32> into f32
      //%9 = gpu.subgroup_reduce  add %8 uniform : (f32) -> f32
      %10 = vector.broadcast %8 : f32 to vector<16xf32>
      %11 = arith.divf %7, %10 : vector<16xf32>
      vector.transfer_write %11, %arg2[%0, %1] {in_bounds = [true], mode = "simt"} : vector<16xf32>, memref<32x256xf32>
      gpu.return
    }
  }
}

