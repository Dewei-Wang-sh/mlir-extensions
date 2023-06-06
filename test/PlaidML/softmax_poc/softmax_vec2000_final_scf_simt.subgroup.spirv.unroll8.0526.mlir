module @softmax attributes {gpu.container_module} {
  memref.global "private" constant @__constant_32x256xf32 : memref<32x256xf32> = dense<5.000000e-01>
  func.func @main() {
    %0 = memref.get_global @__constant_32x256xf32 : memref<32x256xf32>
    %1 = call @test(%0) : (memref<32x256xf32>) -> memref<32x256xf32>
    %cast = memref.cast %1 : memref<32x256xf32> to memref<*xf32>
    //call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }
  func.func @test(%arg0: memref<32x256xf32>) -> memref<32x256xf32> attributes {llvm.emit_c_interface} {
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant 1.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %memref = gpu.alloc  host_shared () : memref<32x256xf32>
    memref.copy %arg0, %memref : memref<32x256xf32> to memref<32x256xf32>
    %memref_1 = gpu.alloc  host_shared () : memref<32x256xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c32, %c1, %c1) threads in (%c16, %c1, %c1) args(%memref : memref<32x256xf32>, %c0 : index, %cst_0 : f32, %c128 : index, %cst : f32, %memref_1 : memref<32x256xf32>, %c64 : index)
    %alloc = memref.alloc() : memref<32x256xf32>
    memref.copy %memref_1, %alloc : memref<32x256xf32> to memref<32x256xf32>
    gpu.dealloc  %memref_1 : memref<32x256xf32>
    gpu.dealloc  %memref : memref<32x256xf32>
    return %alloc : memref<32x256xf32>
  }
  //spirv.module @__spv__test_kernel Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Vector16, Kernel, Groups, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Bfloat16ConversionINTEL, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_INTEL_bfloat16_conversion, SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
  spirv.module @__spv__test_kernel Physical64 OpenCL requires  #spirv.vce<v1.1, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, Groups, SubgroupDispatch, SubgroupBufferBlockIOINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_subgroups, SPV_KHR_no_integer_wrap_decoration]> {
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, %arg1: i64, %arg2: f32, %arg3: i64, %arg4: f32, %arg5: !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, %arg6: i64) "None" attributes {gpu.known_block_size = array<i32: 16, 1, 1>, gpu.known_grid_size = array<i32: 32, 1, 1>, workgroup_attributions = 0 : i64} {
      %cst0_i64 = spirv.Constant 0 : i64
      %cst128_i64 = spirv.Constant 128 : i64
      %cst256_i64 = spirv.Constant 256 : i64
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
      %2 = spirv.IMul %1, %cst256_i64 : i64
      %3 = spirv.IAdd %2, %cst0_i64 : i64
      %4 = spirv.AccessChain %arg0[%3] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      //%5 = spirv.Bitcast %4 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<vector<8xf32>, CrossWorkgroup>
      //%6 = spirv.Load "CrossWorkgroup" %5 : vector<8xf32>
      %411 = spirv.Bitcast %4 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<i32, CrossWorkgroup>
      %611 = spirv.INTEL.SubgroupBlockRead "CrossWorkgroup" %411 : vector<8xi32>
      %6 = spirv.Bitcast %611 : vector<8xi32> to vector<8xf32>
      %7 = spirv.CompositeExtract %6[0 : i32] : vector<8xf32>
      %8 = spirv.CompositeExtract %6[1 : i32] : vector<8xf32>
      %9 = spirv.CompositeExtract %6[2 : i32] : vector<8xf32>
      %10 = spirv.CompositeExtract %6[3 : i32] : vector<8xf32>
      %11 = spirv.CompositeExtract %6[4 : i32] : vector<8xf32>
      %12 = spirv.CompositeExtract %6[5 : i32] : vector<8xf32>
      %13 = spirv.CompositeExtract %6[6 : i32] : vector<8xf32>
      %14 = spirv.CompositeExtract %6[7 : i32] : vector<8xf32>
      %15 = spirv.CL.fmax %7, %8 : f32
      %16 = spirv.CL.fmax %15, %9 : f32
      %17 = spirv.CL.fmax %16, %10 : f32
      %18 = spirv.CL.fmax %17, %11 : f32
      %19 = spirv.CL.fmax %18, %12 : f32
      %20 = spirv.CL.fmax %19, %13 : f32
      %21 = spirv.CL.fmax %20, %14 : f32
      %22 = spirv.IAdd %2, %cst128_i64 : i64
      %23 = spirv.AccessChain %arg0[%22] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      //%24 = spirv.Bitcast %23 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<vector<8xf32>, CrossWorkgroup>
      //%25 = spirv.Load "CrossWorkgroup" %24 : vector<8xf32>
      %231 = spirv.Bitcast %23 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<i32, CrossWorkgroup>
      %251 = spirv.INTEL.SubgroupBlockRead "CrossWorkgroup" %231 : vector<8xi32>
      %25 = spirv.Bitcast %251 : vector<8xi32> to vector<8xf32>
      %26 = spirv.CompositeExtract %25[0 : i32] : vector<8xf32>
      %27 = spirv.CompositeExtract %25[1 : i32] : vector<8xf32>
      %28 = spirv.CompositeExtract %25[2 : i32] : vector<8xf32>
      %29 = spirv.CompositeExtract %25[3 : i32] : vector<8xf32>
      %30 = spirv.CompositeExtract %25[4 : i32] : vector<8xf32>
      %31 = spirv.CompositeExtract %25[5 : i32] : vector<8xf32>
      %32 = spirv.CompositeExtract %25[6 : i32] : vector<8xf32>
      %33 = spirv.CompositeExtract %25[7 : i32] : vector<8xf32>
      %34 = spirv.CL.fmax %26, %27 : f32
      %35 = spirv.CL.fmax %34, %28 : f32
      %36 = spirv.CL.fmax %35, %29 : f32
      %37 = spirv.CL.fmax %36, %30 : f32
      %38 = spirv.CL.fmax %37, %31 : f32
      %39 = spirv.CL.fmax %38, %32 : f32
      %40 = spirv.CL.fmax %39, %33 : f32
      %41 = spirv.CL.fmax %21, %40 : f32
      //%42 = spirv.IsNan %21 : f32
      //%43 = spirv.IsNan %40 : f32
      //%44 = spirv.Select %42, %21, %41 : i1, f32
      //%45 = spirv.Select %43, %40, %44 : i1, f32
      %45 = spirv.CL.fmax %21, %40 : f32
      %46 = spirv.GroupFMax <Subgroup> <Reduce> %45 : f32
      %47 = spirv.CompositeConstruct %46, %46, %46, %46, %46, %46, %46, %46 : (f32, f32, f32, f32, f32, f32, f32, f32) -> vector<8xf32>
      %48 = spirv.FSub %6, %47 : vector<8xf32>
      %49 = spirv.CL.exp %48 : vector<8xf32>
      %50 = spirv.CompositeExtract %49[0 : i32] : vector<8xf32>
      %51 = spirv.CompositeExtract %49[1 : i32] : vector<8xf32>
      %52 = spirv.CompositeExtract %49[2 : i32] : vector<8xf32>
      %53 = spirv.CompositeExtract %49[3 : i32] : vector<8xf32>
      %54 = spirv.CompositeExtract %49[4 : i32] : vector<8xf32>
      %55 = spirv.CompositeExtract %49[5 : i32] : vector<8xf32>
      %56 = spirv.CompositeExtract %49[6 : i32] : vector<8xf32>
      %57 = spirv.CompositeExtract %49[7 : i32] : vector<8xf32>
      %58 = spirv.FAdd %50, %51 : f32
      %59 = spirv.FAdd %58, %52 : f32
      %60 = spirv.FAdd %59, %53 : f32
      %61 = spirv.FAdd %60, %54 : f32
      %62 = spirv.FAdd %61, %55 : f32
      %63 = spirv.FAdd %62, %56 : f32
      %64 = spirv.FAdd %63, %57 : f32
      %65 = spirv.FSub %25, %47 : vector<8xf32>
      %66 = spirv.CL.exp %65 : vector<8xf32>
      %67 = spirv.CompositeExtract %66[0 : i32] : vector<8xf32>
      %68 = spirv.CompositeExtract %66[1 : i32] : vector<8xf32>
      %69 = spirv.CompositeExtract %66[2 : i32] : vector<8xf32>
      %70 = spirv.CompositeExtract %66[3 : i32] : vector<8xf32>
      %71 = spirv.CompositeExtract %66[4 : i32] : vector<8xf32>
      %72 = spirv.CompositeExtract %66[5 : i32] : vector<8xf32>
      %73 = spirv.CompositeExtract %66[6 : i32] : vector<8xf32>
      %74 = spirv.CompositeExtract %66[7 : i32] : vector<8xf32>
      %75 = spirv.FAdd %67, %68 : f32
      %76 = spirv.FAdd %75, %69 : f32
      %77 = spirv.FAdd %76, %70 : f32
      %78 = spirv.FAdd %77, %71 : f32
      %79 = spirv.FAdd %78, %72 : f32
      %80 = spirv.FAdd %79, %73 : f32
      %81 = spirv.FAdd %80, %74 : f32
      %82 = spirv.FAdd %64, %81 : f32
      %83 = spirv.GroupFAdd <Subgroup> <Reduce> %82 : f32
      %cst1 = spirv.Constant 1.0 : f32
      %84 = spirv.FDiv %cst1, %83 : f32
      %85 = spirv.CompositeConstruct %84, %84, %84, %84, %84, %84, %84, %84 : (f32, f32, f32, f32, f32, f32, f32, f32) -> vector<8xf32>
      %86 = spirv.FMul %49, %85 : vector<8xf32>
      %87 = spirv.AccessChain %arg5[%3] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      //%88 = spirv.Bitcast %87 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<vector<8xf32>, CrossWorkgroup>
      //spirv.Store "CrossWorkgroup" %88, %86 : vector<8xf32>
      %871 = spirv.Bitcast %87 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<i32, CrossWorkgroup>
      %861 = spirv.Bitcast %86 : vector<8xf32> to vector<8xi32>
      spirv.INTEL.SubgroupBlockWrite "CrossWorkgroup" %871, %861 : vector<8xi32>
      %89 = spirv.FMul %66, %85 : vector<8xf32>
      %90 = spirv.IAdd %2, %cst128_i64 : i64
      %91 = spirv.AccessChain %arg5[%90] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      //%92 = spirv.Bitcast %91 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<vector<8xf32>, CrossWorkgroup>
      //spirv.Store "CrossWorkgroup" %92, %89 : vector<8xf32>
      %911 = spirv.Bitcast %91 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<i32, CrossWorkgroup>
      %891 = spirv.Bitcast %89 : vector<8xf32> to vector<8xi32>
      spirv.INTEL.SubgroupBlockWrite "CrossWorkgroup" %911, %891 : vector<8xi32>
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
  spirv.ExecutionMode @test_kernel "SubgroupSize", 16
  spirv.ExecutionMode @test_kernel "ContractionOff"
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Bfloat16ConversionINTEL, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_INTEL_bfloat16_conversion, SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<32x256xf32>, %arg1: index, %arg2: f32, %arg3: index, %arg4: f32, %arg5: memref<32x256xf32>, %arg6: index) kernel attributes {gpu.known_block_size = array<i32: 16, 1, 1>, gpu.known_grid_size = array<i32: 32, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = vector.transfer_read %arg0[%0, %arg1], %arg2 {in_bounds = [true], mode = "simd"} : memref<32x256xf32>, vector<8xf32>
      %2 = vector.reduction <maxf>, %1 : vector<8xf32> into f32
      %3 = vector.transfer_read %arg0[%0, %arg3], %arg2 {in_bounds = [true], mode = "simd"} : memref<32x256xf32>, vector<8xf32>
      %4 = vector.reduction <maxf>, %3 : vector<8xf32> into f32
      %5 = arith.maxf %2, %4 : f32
      %6 = gpu.subgroup_reduce  max %5 uniform : (f32) -> f32
      %7 = vector.broadcast %6 : f32 to vector<8xf32>
      %8 = arith.subf %1, %7 : vector<8xf32>
      %9 = math.exp %8 : vector<8xf32>
      %10 = vector.reduction <add>, %9 : vector<8xf32> into f32
      %11 = arith.subf %3, %7 : vector<8xf32>
      %12 = math.exp %11 : vector<8xf32>
      %13 = vector.reduction <add>, %12 : vector<8xf32> into f32
      %14 = arith.addf %10, %13 : f32
      %15 = gpu.subgroup_reduce  add %14 uniform : (f32) -> f32
      %16 = arith.divf %arg4, %15 : f32
      %17 = vector.broadcast %16 : f32 to vector<8xf32>
      %18 = arith.mulf %9, %17 : vector<8xf32>
      vector.transfer_write %18, %arg5[%0, %arg1] {in_bounds = [true], mode = "simd"} : vector<8xf32>, memref<32x256xf32>
      %19 = arith.mulf %12, %17 : vector<8xf32>
      vector.transfer_write %19, %arg5[%0, %arg6] {in_bounds = [true], mode = "simd"} : vector<8xf32>, memref<32x256xf32>
      gpu.return
    }
  }
}

