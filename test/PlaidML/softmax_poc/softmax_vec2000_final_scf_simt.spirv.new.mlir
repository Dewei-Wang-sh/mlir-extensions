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
      //%cst1024_i64 = spirv.Constant 1024 : i64
      %cst1024_i64 = spirv.Constant 256 : i64
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

      %860  = spirv.FSub  %8, %84 : f32
      %861  = spirv.FSub %12, %84 : f32
      %862  = spirv.FSub %16, %84 : f32
      %863  = spirv.FSub %20, %84 : f32
      %864  = spirv.FSub %24, %84 : f32
      %865  = spirv.FSub %28, %84 : f32
      %866  = spirv.FSub %32, %84 : f32
      %867  = spirv.FSub %36, %84 : f32
      %868  = spirv.FSub %40, %84 : f32
      %869  = spirv.FSub %44, %84 : f32
      %8610 = spirv.FSub %48, %84 : f32
      %8611 = spirv.FSub %52, %84 : f32
      %8612 = spirv.FSub %56, %84 : f32
      %8613 = spirv.FSub %60, %84 : f32
      %8614 = spirv.FSub %64, %84 : f32
      %8615 = spirv.FSub %67, %84 : f32

      %88  = spirv.CL.exp %860 : f32
      %89  = spirv.CL.exp %861 : f32
      %90  = spirv.CL.exp %862 : f32
      %91  = spirv.CL.exp %863 : f32
      %92  = spirv.CL.exp %864 : f32
      %93  = spirv.CL.exp %865 : f32
      %94  = spirv.CL.exp %866 : f32
      %95  = spirv.CL.exp %867 : f32
      %96  = spirv.CL.exp %868 : f32
      %97  = spirv.CL.exp %869 : f32
      %98  = spirv.CL.exp %8610 : f32
      %99  = spirv.CL.exp %8611 : f32
      %100 = spirv.CL.exp %8612 : f32
      %101 = spirv.CL.exp %8613 : f32
      %102 = spirv.CL.exp %8614 : f32
      %103 = spirv.CL.exp %8615 : f32
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
      //%com = spirv.CompositeConstruct %div, %div, %div, %div, %div, %div, %div, %div, %div, %div, %div, %div, %div, %div, %div, %div : (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) -> vector<16xf32>
      //%121 = spirv.FMul %87, %com : vector<16xf32>
      //%122 = spirv.AccessChain %arg2[%5] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      //%123 = spirv.Bitcast %122 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<vector<16xf32>, CrossWorkgroup>
      //spirv.Store "CrossWorkgroup" %123, %121 : vector<16xf32>
      %1210 = spirv.FMul %88, %div : f32
      %1220 = spirv.AccessChain %arg2[%5] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      spirv.Store "CrossWorkgroup" %1220, %1210 : f32

      %1211 = spirv.FMul %89, %div : f32
      %1221 = spirv.AccessChain %arg2[%9] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      spirv.Store "CrossWorkgroup" %1221, %1211 : f32

      %1212 = spirv.FMul %90, %div : f32
      %1222 = spirv.AccessChain %arg2[%13] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      spirv.Store "CrossWorkgroup" %1222, %1212 : f32

      %1213 = spirv.FMul %91, %div : f32
      %1223 = spirv.AccessChain %arg2[%17] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      spirv.Store "CrossWorkgroup" %1223, %1213 : f32

      %1214 = spirv.FMul %92, %div : f32
      %1224 = spirv.AccessChain %arg2[%21] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      spirv.Store "CrossWorkgroup" %1224, %1214 : f32

      %1215 = spirv.FMul %93, %div : f32
      %1225 = spirv.AccessChain %arg2[%25] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      spirv.Store "CrossWorkgroup" %1225, %1215 : f32

      %1216 = spirv.FMul %94, %div : f32
      %1226 = spirv.AccessChain %arg2[%29] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      spirv.Store "CrossWorkgroup" %1226, %1216 : f32

      %1217 = spirv.FMul %95, %div : f32
      %1227 = spirv.AccessChain %arg2[%33] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      spirv.Store "CrossWorkgroup" %1227, %1217 : f32

      %1218 = spirv.FMul %96, %div : f32
      %1228 = spirv.AccessChain %arg2[%37] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      spirv.Store "CrossWorkgroup" %1228, %1218 : f32

      %1219 = spirv.FMul %97, %div : f32
      %1229 = spirv.AccessChain %arg2[%41] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      spirv.Store "CrossWorkgroup" %1229, %1219 : f32

      %12110 = spirv.FMul %98, %div : f32
      %12210 = spirv.AccessChain %arg2[%45] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      spirv.Store "CrossWorkgroup" %12210, %12110 : f32

      %12111 = spirv.FMul %99, %div : f32
      %12211 = spirv.AccessChain %arg2[%49] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      spirv.Store "CrossWorkgroup" %12211, %12111 : f32

      %12112 = spirv.FMul %100, %div : f32
      %12212 = spirv.AccessChain %arg2[%53] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      spirv.Store "CrossWorkgroup" %12212, %12112 : f32

      %12113 = spirv.FMul %101, %div : f32
      %12213 = spirv.AccessChain %arg2[%57] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      spirv.Store "CrossWorkgroup" %12213, %12113 : f32

      %12114 = spirv.FMul %102, %div : f32
      %12214 = spirv.AccessChain %arg2[%61] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      spirv.Store "CrossWorkgroup" %12214, %12114 : f32

      %12115 = spirv.FMul %103, %div : f32
      %12215 = spirv.AccessChain %arg2[%65] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      spirv.Store "CrossWorkgroup" %12215, %12115 : f32

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

