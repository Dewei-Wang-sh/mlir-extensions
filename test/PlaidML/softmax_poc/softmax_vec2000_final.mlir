//imex-opt --affine-super-vectorize="virtual-vector-size=1024 vectorize-reductions=true" softmax_before_vec.maxf.mlir -affine-loop-normalize=promote-single-iter=true -canonicalize -cse
module @softmax {
  func.func @test(%arg0: memref<32x1024xf32>) -> memref<32x1024xf32> {
    %cst = arith.constant dense<0.000000e+00> : vector<1024xf32>
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x1024xf32>
    affine.parallel (%arg1) = (0) to (32) {
      %0 = vector.transfer_read %arg0[%arg1, %c0], %cst_0 {in_bounds = [true]} : memref<32x1024xf32>, vector<1024xf32>
      %1 = vector.reduction <maxf>, %0 : vector<1024xf32> into f32
      %2 = vector.broadcast %1 : f32 to vector<1024xf32>
      %3 = arith.subf %0, %2 : vector<1024xf32>
      %5 = math.exp %3 : vector<1024xf32>
      %8 = vector.reduction <add>, %5 : vector<1024xf32> into f32
      %9 = vector.broadcast %8 : f32 to vector<1024xf32>
      %10 = arith.divf %5, %9 : vector<1024xf32>
      vector.transfer_write %10, %alloc[%arg1, %c0] {in_bounds = [true]} : vector<1024xf32>, memref<32x1024xf32>
    }
    return %alloc : memref<32x1024xf32>
  }
}


