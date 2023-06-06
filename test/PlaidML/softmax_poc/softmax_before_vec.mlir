module @softmax {
  func.func @test(%arg0: memref<32x2000xf32>) -> memref<32x2000xf32> {
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x2000xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<32x2000xf32>
    affine.for %arg1 = 0 to 32 {
      %0 = affine.for %arg2 = 0 to 2000 iter_args(%arg3 = %cst) -> (f32) {
        %2 = affine.load %arg0[%arg1, %arg2] : memref<32x2000xf32>
        %3 = arith.cmpf ogt, %arg3, %2 : f32
        %4 = arith.select %3, %arg3, %2 : f32
        affine.yield %4 : f32
      }
      affine.for %arg2 = 0 to 2000 {
        %2 = affine.load %arg0[%arg1, %arg2] : memref<32x2000xf32>
        %3 = arith.subf %2, %0 : f32
        affine.store %3, %alloc[%arg1, %arg2] : memref<32x2000xf32>
      }
      affine.for %arg2 = 0 to 2000 {
        %2 = affine.load %alloc[%arg1, %arg2] : memref<32x2000xf32>
        %3 = math.exp %2 : f32
        affine.store %3, %alloc_1[%arg1, %arg2] : memref<32x2000xf32>
      }
      %1 = affine.for %arg2 = 0 to 2000 iter_args(%arg3 = %cst_0) -> (f32) {
        %2 = affine.load %alloc_1[%arg1, %arg2] : memref<32x2000xf32>
        %3 = arith.addf %arg3, %2 : f32
        affine.yield %3 : f32
      }
      affine.for %arg2 = 0 to 2000 {
        %2 = affine.load %alloc_1[%arg1, %arg2] : memref<32x2000xf32>
        %3 = arith.divf %2, %1 : f32
        affine.store %3, %alloc[%arg1, %arg2] : memref<32x2000xf32>
      }
    }
    memref.dealloc %alloc : memref<32x2000xf32>
    memref.dealloc %alloc_1 : memref<32x2000xf32>
    return %alloc : memref<32x2000xf32>
  }
}

//imex-opt rubbish1.mlir -convert-linalg-to-affine-loops -affine-raise-reduction -affine-loop-fusion -affine-loop-invariant-code-motion -affine-scalrep --affine-super-vectorize="virtual-vector-size=2000 vectorize-reductions=true"

