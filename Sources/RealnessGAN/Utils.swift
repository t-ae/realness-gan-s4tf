import Foundation
import TensorFlow
import TensorBoardX

@differentiable(wrt: (p, q))
func klDivergence(p: Tensor<Float>, q: Tensor<Float>, epsilon: Float = 1e-8) -> Tensor<Float> {
    let tmp = p * log((p + epsilon) / (q + epsilon))
    return tmp.sum(alongAxes: 1).mean()
}

public func sampleNoise(size: Int, latentSize: Int) -> Tensor<Float> {
    Tensor(randomNormal: [size, latentSize])
}

func createAnchor(numberOfOutcomes: Int, center: Float, samples: Int = 1000) -> Tensor<Float> {
    let histogram: Tensor<Int32> = _Raw.histogramFixedWidth(
        Tensor<Float>(randomNormal: [samples]) + center,
        valueRange: [-2, 2],
        nbins: Tensor(Int32(numberOfOutcomes))
    )
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

