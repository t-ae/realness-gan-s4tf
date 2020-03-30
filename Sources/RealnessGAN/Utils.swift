import Foundation
import TensorFlow
import TensorBoardX

@differentiable
func lrelu(_ x: Tensor<Float>) -> Tensor<Float> {
    leakyRelu(x)
}

@differentiable(wrt: (p, q))
func klDivergence(p: Tensor<Float>, q: Tensor<Float>, epsilon: Float = 1e-8) -> Tensor<Float> {
    let tmp = p * log((p + epsilon) / (q + epsilon))
    return tmp.sum(alongAxes: 1).mean()
}

public func sampleNoise(size: Int, latentSize: Int) -> Tensor<Float> {
    Tensor(randomNormal: [size, latentSize])
}

func createAnchor(numberOfOutcomes: Int, center: Float, samples: Int = 1000) -> Tensor<Float> {
    let range: Float = 2
    let noise = Tensor<Float>(randomNormal: [samples]) + center
    var histogram: Tensor<Int32> = _Raw.histogramFixedWidth(
        noise,
        valueRange: Tensor([-range, range]),
        nbins: Tensor(Int32(numberOfOutcomes))
    )
    // Subtract out of range values
    // FIXME: Bug on GPU
    withDevice(.cpu) {
        histogram[0] -= Tensor(noise .< -range).sum()
        histogram[-1] -= Tensor(noise .> range).sum()
    }
    
    let float = Tensor<Float>(histogram)
    return float / float.sum()
}


extension SummaryWriter {
    func plotImages(tag: String,
                    images: Tensor<Float>,
                    colSize: Int = 8,
                    globalStep: Int) {
        var images = images
        images = (images + 1) / 2
        images = images.clipped(min: 0, max: 1)
        addImages(tag: tag,
                  images: images,
                  colSize: colSize,
                  globalStep: globalStep)
    }
}

