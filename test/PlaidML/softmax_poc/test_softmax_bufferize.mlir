// imex-opt --pass-pipeline="builtin.module(convert-tensor-to-linalg,arith-bufferize,func.func(empty-tensor-to-alloc-tensor,eliminate-empty-tensors,scf-bufferize,shape-bufferize,linalg-bufferize,bufferization-bufferize,tensor-bufferize),func-bufferize,func.func(finalizing-bufferize))" test_softmax.mlir

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>
module @softmax {
  memref.global "private" constant @__constant_32x2000xf32 : memref<32x2000xf32> = dense<5.000000e-01>
  func.func @main() {
    %0 = memref.get_global @__constant_32x2000xf32 : memref<32x2000xf32>
    %1 = call @test(%0) : (memref<32x2000xf32>) -> memref<32x2000xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @test(%arg0: memref<32x2000xf32>) -> memref<32x2000xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x1xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<32x1xf32>
    linalg.fill ins(%cst_0 : f32) outs(%alloc_1 : memref<32x1xf32>)
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<32x1xf32>
    memref.copy %alloc_1, %alloc_2 : memref<32x1xf32> to memref<32x1xf32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : memref<32x2000xf32>) outs(%alloc_2 : memref<32x1xf32>) attrs =  {iterator_ranges = [10, 20], name = "softmax"} {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.cmpf ogt, %out, %in : f32
      %1 = arith.select %0, %out, %in : f32
      linalg.yield %1 : f32
    }
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32x2000xf32>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32x2000xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %alloc_2 : memref<32x2000xf32>, memref<32x1xf32>) outs(%alloc_4 : memref<32x2000xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %0 = arith.subf %in, %in_12 : f32
      linalg.yield %0 : f32
    }
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x2000xf32>
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<32x2000xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_4 : memref<32x2000xf32>) outs(%alloc_6 : memref<32x2000xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = math.exp %in : f32
      linalg.yield %0 : f32
    }
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<32x1xf32>
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<32x1xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_8 : memref<32x1xf32>)
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<32x1xf32>
    memref.copy %alloc_8, %alloc_9 : memref<32x1xf32> to memref<32x1xf32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%alloc_6 : memref<32x2000xf32>) outs(%alloc_9 : memref<32x1xf32>) attrs =  {iterator_ranges = [10, 20], name = "softmax"} {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.addf %out, %in : f32
      linalg.yield %0 : f32
    }
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x2000xf32>
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<32x2000xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_6, %alloc_9 : memref<32x2000xf32>, memref<32x1xf32>) outs(%alloc_11 : memref<32x2000xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %0 = arith.divf %in, %in_12 : f32
      linalg.yield %0 : f32
    }
    return %alloc_11 : memref<32x2000xf32>
  }
}
