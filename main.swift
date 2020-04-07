import TensorFlow
withDevice(.gpu) {
    var hist = Tensor<Int32>(zeros: [10])
    // hist[0] = Tensor(0)
    let x = Tensor<Float>(hist)
    print(x)
}
