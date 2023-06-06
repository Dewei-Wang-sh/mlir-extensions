

//mlir-opt -vector-subslice1 softmax_vec2000_final.mlir -lower-affine
module @softmax {
  func.func @test(%arg0: memref<32x256xf32>) -> memref<32x256xf32> {
    %cst = arith.constant dense<0.000000e+00> : vector<256xf32>
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x256xf32>
    %c0_1 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
      %c0_2 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c16 = arith.constant 16 : index
      %c1_3 = arith.constant 1 : index
    scf.parallel (%arg1) = (%c0_1) to (%c32) step (%c1) {
      scf.parallel (%arg2) = (%c0_2) to (%c16) step (%c1_3) {
        %0 = arith.addi %c0, %arg2 : index
        %1 = vector.transfer_read %arg0[%arg1, %0], %cst_0 {in_bounds = [true], mode = "simt"} : memref<32x256xf32>, vector<16xf32>
        %2 = vector.reduction <maxf>, %1 : vector<16xf32> into f32
        %3 = gpu.subgroup_reduce  max %2 uniform : (f32) -> f32
        %4 = vector.broadcast %3 : f32 to vector<16xf32>
        %5 = arith.subf %1, %4 : vector<16xf32>
        %6 = math.exp %5 : vector<16xf32>
        %7 = vector.reduction <add>, %6 : vector<16xf32> into f32
        %8 = gpu.subgroup_reduce  add %7 uniform : (f32) -> f32
        %9 = vector.broadcast %8 : f32 to vector<16xf32>
        %10 = arith.divf %6, %9 : vector<16xf32>
        %11 = arith.addi %c0, %arg2 : index
        vector.transfer_write %10, %alloc[%arg1, %11] {in_bounds = [true], mode = "simt"} : vector<16xf32>, memref<32x256xf32>
        scf.yield
      }
      scf.yield
    }
    return %alloc : memref<32x256xf32>
  }
}

