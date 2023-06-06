module @softmax attributes {gpu.container_module} {
  memref.global "private" constant @__constant_32x256xf32 : memref<32x256xf32> = dense<5.000000e-01>
  func.func @main() {
    %0 = memref.get_global @__constant_32x256xf32 : memref<32x256xf32>
    %1 = call @test(%0) : (memref<32x256xf32>) -> memref<32x256xf32>
    %cast = memref.cast %1 : memref<32x256xf32> to memref<*xf32>
    //call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }
func.func private @printMemrefF32(memref<*xf32>)
  func.func @test(%arg0: memref<32x256xf32>) -> memref<32x256xf32> attributes {llvm.emit_c_interface} {
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant dense<-1.000000e+00> : vector<4xf32>
    %c192 = arith.constant 192 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %memref = gpu.alloc  host_shared () : memref<32x256xf32>
    memref.copy %arg0, %memref : memref<32x256xf32> to memref<32x256xf32>
    %memref_1 = gpu.alloc  host_shared () : memref<32x256xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c32, %c1, %c1) threads in (%c16, %c1, %c1) args(%memref : memref<32x256xf32>, %c0 : index, %cst_0 : f32, %c64 : index, %c128 : index, %c192 : index, %cst : vector<4xf32>, %memref_1 : memref<32x256xf32>)
    %alloc = memref.alloc() : memref<32x256xf32>
    memref.copy %memref_1, %alloc : memref<32x256xf32> to memref<32x256xf32>
    gpu.dealloc  %memref_1 : memref<32x256xf32>
    gpu.dealloc  %memref : memref<32x256xf32>
    return %alloc : memref<32x256xf32>
  }
  //spirv.module @__spv__test_kernel Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Groups, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Bfloat16ConversionINTEL, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_INTEL_bfloat16_conversion, SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
  spirv.module @__spv__test_kernel Physical64 OpenCL requires  #spirv.vce<v1.1, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, Groups, SubgroupDispatch, SubgroupBufferBlockIOINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_subgroups, SPV_KHR_no_integer_wrap_decoration]> {
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, %arg1: i64, %arg2: f32, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: vector<4xf32>, %arg7: !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>) "None" attributes {gpu.known_block_size = array<i32: 16, 1, 1>, gpu.known_grid_size = array<i32: 32, 1, 1>, workgroup_attributions = 0 : i64} {
      %cst256_i64 = spirv.Constant 256 : i64
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
      %2 = spirv.IMul %1, %cst256_i64 : i64
      %3 = spirv.IAdd %2, %arg1 : i64
      %4 = spirv.AccessChain %arg0[%3] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      //%5 = spirv.Bitcast %4 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<vector<4xf32>, CrossWorkgroup>
      //%6 = spirv.Load "CrossWorkgroup" %5 : vector<4xf32>
      %411 = spirv.Bitcast %4 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<i32, CrossWorkgroup>
      %611 = spirv.INTEL.SubgroupBlockRead "CrossWorkgroup" %411 : vector<4xi32>
      %6 = spirv.Bitcast %611 : vector<4xi32> to vector<4xf32>
      %7 = spirv.CompositeExtract %6[0 : i32] : vector<4xf32>
      %8 = spirv.CompositeExtract %6[1 : i32] : vector<4xf32>
      %9 = spirv.CompositeExtract %6[2 : i32] : vector<4xf32>
      %10 = spirv.CompositeExtract %6[3 : i32] : vector<4xf32>
      %11 = spirv.CL.fmax %7, %8 : f32
      %12 = spirv.CL.fmax %11, %9 : f32
      %13 = spirv.CL.fmax %12, %10 : f32
      %14 = spirv.IAdd %2, %arg3 : i64
      %15 = spirv.AccessChain %arg0[%14] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      //%16 = spirv.Bitcast %15 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<vector<4xf32>, CrossWorkgroup>
      //%17 = spirv.Load "CrossWorkgroup" %16 : vector<4xf32>
      %151 = spirv.Bitcast %15 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<i32, CrossWorkgroup>
      %171 = spirv.INTEL.SubgroupBlockRead "CrossWorkgroup" %151 : vector<4xi32>
      %17 = spirv.Bitcast %171 : vector<4xi32> to vector<4xf32>
      %18 = spirv.CompositeExtract %17[0 : i32] : vector<4xf32>
      %19 = spirv.CompositeExtract %17[1 : i32] : vector<4xf32>
      %20 = spirv.CompositeExtract %17[2 : i32] : vector<4xf32>
      %21 = spirv.CompositeExtract %17[3 : i32] : vector<4xf32>
      %22 = spirv.CL.fmax %18, %19 : f32
      %23 = spirv.CL.fmax %22, %20 : f32
      %24 = spirv.CL.fmax %23, %21 : f32
      %25 = spirv.IAdd %2, %arg4 : i64
      %26 = spirv.AccessChain %arg0[%25] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      //%27 = spirv.Bitcast %26 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<vector<4xf32>, CrossWorkgroup>
      //%28 = spirv.Load "CrossWorkgroup" %27 : vector<4xf32>
      %261 = spirv.Bitcast %26 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<i32, CrossWorkgroup>
      %281 = spirv.INTEL.SubgroupBlockRead "CrossWorkgroup" %261 : vector<4xi32>
      %28 = spirv.Bitcast %281 : vector<4xi32> to vector<4xf32>
      %29 = spirv.CompositeExtract %28[0 : i32] : vector<4xf32>
      %30 = spirv.CompositeExtract %28[1 : i32] : vector<4xf32>
      %31 = spirv.CompositeExtract %28[2 : i32] : vector<4xf32>
      %32 = spirv.CompositeExtract %28[3 : i32] : vector<4xf32>
      %33 = spirv.CL.fmax %29, %30 : f32
      %34 = spirv.CL.fmax %33, %31 : f32
      %35 = spirv.CL.fmax %34, %32 : f32
      %36 = spirv.IAdd %2, %arg5 : i64
      %37 = spirv.AccessChain %arg0[%36] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      //%38 = spirv.Bitcast %37 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<vector<4xf32>, CrossWorkgroup>
      //%39 = spirv.Load "CrossWorkgroup" %38 : vector<4xf32>
      %371 = spirv.Bitcast %37 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<i32, CrossWorkgroup>
      %391 = spirv.INTEL.SubgroupBlockRead "CrossWorkgroup" %371 : vector<4xi32>
      %39 = spirv.Bitcast %391 : vector<4xi32> to vector<4xf32>
      %40 = spirv.CompositeExtract %39[0 : i32] : vector<4xf32>
      %41 = spirv.CompositeExtract %39[1 : i32] : vector<4xf32>
      %42 = spirv.CompositeExtract %39[2 : i32] : vector<4xf32>
      %43 = spirv.CompositeExtract %39[3 : i32] : vector<4xf32>
      %44 = spirv.CL.fmax %40, %41 : f32
      %45 = spirv.CL.fmax %44, %42 : f32
      %46 = spirv.CL.fmax %45, %43 : f32
      %47 = spirv.CL.fmax %13, %24 : f32
      %48 = spirv.IsNan %13 : f32
      %49 = spirv.IsNan %24 : f32
      %50 = spirv.Select %48, %13, %47 : i1, f32
      %51 = spirv.Select %49, %24, %50 : i1, f32
      %52 = spirv.CL.fmax %35, %46 : f32
      %53 = spirv.IsNan %35 : f32
      %54 = spirv.IsNan %46 : f32
      %55 = spirv.Select %53, %35, %52 : i1, f32
      %56 = spirv.Select %54, %46, %55 : i1, f32
      %57 = spirv.CL.fmax %51, %56 : f32
      %58 = spirv.IsNan %51 : f32
      %59 = spirv.IsNan %56 : f32
      %60 = spirv.Select %58, %51, %57 : i1, f32
      %61 = spirv.Select %59, %56, %60 : i1, f32
      %62 = spirv.GroupFMax <Subgroup> <Reduce> %61 : f32
      %63 = spirv.CompositeConstruct %62, %62, %62, %62 : (f32, f32, f32, f32) -> vector<4xf32>
      %64 = spirv.FSub %6, %63 : vector<4xf32>
      %65 = spirv.CL.exp %64 : vector<4xf32>
      %66 = spirv.CompositeExtract %65[0 : i32] : vector<4xf32>
      %67 = spirv.CompositeExtract %65[1 : i32] : vector<4xf32>
      %68 = spirv.CompositeExtract %65[2 : i32] : vector<4xf32>
      %69 = spirv.CompositeExtract %65[3 : i32] : vector<4xf32>
      %70 = spirv.FAdd %66, %67 : f32
      %71 = spirv.FAdd %70, %68 : f32
      %72 = spirv.FAdd %71, %69 : f32
      %73 = spirv.FSub %17, %63 : vector<4xf32>
      %74 = spirv.CL.exp %73 : vector<4xf32>
      %75 = spirv.FSub %28, %63 : vector<4xf32>
      %76 = spirv.CL.exp %75 : vector<4xf32>
      %77 = spirv.FSub %39, %63 : vector<4xf32>
      %78 = spirv.CL.exp %77 : vector<4xf32>
      %79 = spirv.GroupFAdd <Subgroup> <Reduce> %72 : f32
      %80 = spirv.CompositeConstruct %79, %79, %79, %79 : (f32, f32, f32, f32) -> vector<4xf32>
      %cst1 = spirv.Constant 1.0 : f32
      %div1 = spirv.FDiv %cst1, %79 : f32
      %81 = spirv.CompositeConstruct %div1, %div1, %div1, %div1 : (f32, f32, f32, f32) -> vector<4xf32>
      //%81 = spirv.FDiv %arg6, %80 : vector<4xf32>
      %82 = spirv.FMul %65, %81 : vector<4xf32>
      %83 = spirv.AccessChain %arg7[%3] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      //%84 = spirv.Bitcast %83 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<vector<4xf32>, CrossWorkgroup>
      //spirv.Store "CrossWorkgroup" %84, %82 : vector<4xf32>
      %831 = spirv.Bitcast %83 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<i32, CrossWorkgroup>
      %821 = spirv.Bitcast %82 : vector<4xf32> to vector<4xi32>
      spirv.INTEL.SubgroupBlockWrite "CrossWorkgroup" %831, %821 : vector<4xi32>
      %85 = spirv.FMul %74, %81 : vector<4xf32>
      %86 = spirv.AccessChain %arg7[%14] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      //%87 = spirv.Bitcast %86 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<vector<4xf32>, CrossWorkgroup>
      //spirv.Store "CrossWorkgroup" %87, %85 : vector<4xf32>
      %861 = spirv.Bitcast %86 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<i32, CrossWorkgroup>
      %851 = spirv.Bitcast %85 : vector<4xf32> to vector<4xi32>
      spirv.INTEL.SubgroupBlockWrite "CrossWorkgroup" %861, %851 : vector<4xi32>
      %88 = spirv.FMul %76, %81 : vector<4xf32>
      %89 = spirv.AccessChain %arg7[%25] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      //%90 = spirv.Bitcast %89 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<vector<4xf32>, CrossWorkgroup>
      //spirv.Store "CrossWorkgroup" %90, %88 : vector<4xf32>
      %891 = spirv.Bitcast %89 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<i32, CrossWorkgroup>
      %881 = spirv.Bitcast %88 : vector<4xf32> to vector<4xi32>
      spirv.INTEL.SubgroupBlockWrite "CrossWorkgroup" %891, %881 : vector<4xi32>
      %91 = spirv.FMul %78, %81 : vector<4xf32>
      %92 = spirv.AccessChain %arg7[%36] : !spirv.ptr<!spirv.array<8192 x f32>, CrossWorkgroup>, i64
      //%93 = spirv.Bitcast %92 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<vector<4xf32>, CrossWorkgroup>
      //spirv.Store "CrossWorkgroup" %93, %91 : vector<4xf32>
      %921 = spirv.Bitcast %92 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<i32, CrossWorkgroup>
      %911 = spirv.Bitcast %91 : vector<4xf32> to vector<4xi32>
      spirv.INTEL.SubgroupBlockWrite "CrossWorkgroup" %921, %911 : vector<4xi32>
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
  spirv.ExecutionMode @test_kernel "SubgroupSize", 16
  spirv.ExecutionMode @test_kernel "ContractionOff"
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Bfloat16ConversionINTEL, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_INTEL_bfloat16_conversion, SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<32x256xf32>, %arg1: index, %arg2: f32, %arg3: index, %arg4: index, %arg5: index, %arg6: vector<4xf32>, %arg7: memref<32x256xf32>) kernel attributes {gpu.known_block_size = array<i32: 16, 1, 1>, gpu.known_grid_size = array<i32: 32, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = vector.transfer_read %arg0[%0, %arg1], %arg2 {in_bounds = [true], mode = "simd"} : memref<32x256xf32>, vector<4xf32>
      %2 = vector.reduction <maxf>, %1 : vector<4xf32> into f32
      %3 = vector.transfer_read %arg0[%0, %arg3], %arg2 {in_bounds = [true], mode = "simd"} : memref<32x256xf32>, vector<4xf32>
      %4 = vector.reduction <maxf>, %3 : vector<4xf32> into f32
      %5 = vector.transfer_read %arg0[%0, %arg4], %arg2 {in_bounds = [true], mode = "simd"} : memref<32x256xf32>, vector<4xf32>
      %6 = vector.reduction <maxf>, %5 : vector<4xf32> into f32
      %7 = vector.transfer_read %arg0[%0, %arg5], %arg2 {in_bounds = [true], mode = "simd"} : memref<32x256xf32>, vector<4xf32>
      %8 = vector.reduction <maxf>, %7 : vector<4xf32> into f32
      %9 = arith.maxf %2, %4 : f32
      %10 = arith.maxf %6, %8 : f32
      %11 = arith.maxf %9, %10 : f32
      %12 = gpu.subgroup_reduce  max %11 uniform : (f32) -> f32
      %13 = vector.broadcast %12 : f32 to vector<4xf32>
      %14 = arith.subf %1, %13 : vector<4xf32>
      %15 = math.exp %14 : vector<4xf32>
      %16 = vector.reduction <add>, %15 : vector<4xf32> into f32
      %17 = arith.subf %3, %13 : vector<4xf32>
      %18 = math.exp %17 : vector<4xf32>
      %19 = arith.subf %5, %13 : vector<4xf32>
      %20 = math.exp %19 : vector<4xf32>
      %21 = arith.subf %7, %13 : vector<4xf32>
      %22 = math.exp %21 : vector<4xf32>
      %23 = gpu.subgroup_reduce  add %16 uniform : (f32) -> f32
      %24 = vector.broadcast %23 : f32 to vector<4xf32>
      %25 = arith.divf %arg6, %24 : vector<4xf32>
      %26 = arith.mulf %15, %25 : vector<4xf32>
      vector.transfer_write %26, %arg7[%0, %arg1] {in_bounds = [true], mode = "simd"} : vector<4xf32>, memref<32x256xf32>
      %27 = arith.mulf %18, %25 : vector<4xf32>
      vector.transfer_write %27, %arg7[%0, %arg3] {in_bounds = [true], mode = "simd"} : vector<4xf32>, memref<32x256xf32>
      %28 = arith.mulf %20, %25 : vector<4xf32>
      vector.transfer_write %28, %arg7[%0, %arg4] {in_bounds = [true], mode = "simd"} : vector<4xf32>, memref<32x256xf32>
      %29 = arith.mulf %22, %25 : vector<4xf32>
      vector.transfer_write %29, %arg7[%0, %arg5] {in_bounds = [true], mode = "simd"} : vector<4xf32>, memref<32x256xf32>
      gpu.return
    }
  }
}

