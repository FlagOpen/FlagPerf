module {
  func.func @main(%arg0: tensor<1x1024x1024xf32>, %arg1: tensor<1024x1024xf32>) -> tuple<tensor<1x1024x1024xf32>> {
    %0 = "tx8be.Input"(%arg0) {layout_str = "Unknown", mem_scope = #tx8be<memscope_mode DDR>} : (tensor<1x1024x1024xf32>) -> tensor<1x1024x1024xf32>
    %1 = "tx8be.Input"(%arg1) {layout_str = "Unknown", mem_scope = #tx8be<memscope_mode DDR>} : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %2 = "tx8be.None"() : () -> none
    %3 = "tx8be.None"() : () -> none
    %4 = "tx8be.Gemm"(%0, %1, %2, %3) {layout_str = "NTensor", op_mode = #tx8be<GemmMode Normal>, sparse_en = false, transL = false, transR = true} : (tensor<1x1024x1024xf32>, tensor<1024x1024xf32>, none, none) -> tensor<1x1024x1024xf32>
    %5 = "tx8be.Tuple"(%4) {input_nums = 1 : i64, layout_str = "Tuple"} : (tensor<1x1024x1024xf32>) -> tuple<tensor<1x1024x1024xf32>>
    %6 = "tx8be.Output"(%5) {layout_str = "Tuple", mem_scope = #tx8be<memscope_mode DDR>} : (tuple<tensor<1x1024x1024xf32>>) -> tuple<tensor<1x1024x1024xf32>>
    return %6 : tuple<tensor<1x1024x1024xf32>>
  }
}
