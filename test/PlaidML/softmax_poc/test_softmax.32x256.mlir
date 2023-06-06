#map  = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>
module @softmax {
  func.func @test(%arg0: tensor<32x256xf32>) -> tensor<32x256xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<32x1xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<32x256xf32>) outs(%1 : tensor<32x1xf32>)  {
    ^bb0(%in: f32, %out: f32):
      %12 = arith.cmpf ogt, %out, %in : f32
      %13 = arith.select %12, %out, %in : f32
      linalg.yield %13 : f32
    } -> tensor<32x1xf32>
    %3 = tensor.empty() : tensor<32x256xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %2 : tensor<32x256xf32>, tensor<32x1xf32>) outs(%3 : tensor<32x256xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.subf %in, %in_1 : f32
      linalg.yield %12 : f32
    } -> tensor<32x256xf32>
    %5 = tensor.empty() : tensor<32x256xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<32x256xf32>) outs(%5 : tensor<32x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %12 = math.exp %in : f32
      linalg.yield %12 : f32
    } -> tensor<32x256xf32>
    %7 = tensor.empty() : tensor<32x1xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%6 : tensor<32x256xf32>) outs(%8 : tensor<32x1xf32>) } {
    ^bb0(%in: f32, %out: f32):
      %12 = arith.addf %out, %in : f32
      linalg.yield %12 : f32
    } -> tensor<32x1xf32>
    %10 = tensor.empty() : tensor<32x256xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%6, %9 : tensor<32x256xf32>, tensor<32x1xf32>) outs(%10 : tensor<32x256xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.divf %in, %in_1 : f32
      linalg.yield %12 : f32
    } -> tensor<32x256xf32>
    return %11 : tensor<32x256xf32>
  }
}


