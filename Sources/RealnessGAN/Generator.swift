import Foundation
import TensorFlow
import GANUtils

struct GBlock: Layer {
    var conv: TransposedConv2D<Float>
    
    var bn: BatchNorm<Float>
    
    init(
        inputChannels: Int,
        outputChannels: Int,
        initialBlock: Bool = false
    ) {
        conv = TransposedConv2D(filterShape: (4, 4, outputChannels, inputChannels),
                                padding: initialBlock ? .valid : .same,
                                useBias: false,
                                filterInitializer: heNormal())
        bn = BatchNorm(featureCount: outputChannels)
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = conv(input)
        x = bn(x)
        x = lrelu(x)
        return x
    }
}

struct Generator: Layer {
    struct Config: Codable {
        var latentSize: Int
        var baseChannels: Int = 8
        var maxChannels: Int = 256
    }
    
    var x4Block: GBlock
    var x8Block: GBlock
    var x16Block: GBlock
    var x32Block: GBlock
    var x64Block: GBlock
    var x128Block: GBlock
    var x256Block: GBlock
    
    var toRGB: Conv2D<Float>
    
    var upsample = UpSampling2D<Float>(size: 2)
    
    @noDerivative
    let imageSize: ImageSize
    
    init(config: Config, imageSize: ImageSize) {
        self.imageSize = imageSize
        
        let baseChannels = config.baseChannels
        let maxChannels = config.maxChannels
        
        func ioChannels(for size: ImageSize) -> (i: Int, o: Int) {
            guard size <= imageSize else {
                return (0, 0)
            }
            let d = imageSize.log2 - size.log2
            let o = baseChannels * 1 << d
            let i = o * 2
            return (min(i, maxChannels), min(o, maxChannels))
        }
        
        let io4 = ioChannels(for: .x4)
        x4Block = GBlock(inputChannels: config.latentSize, outputChannels: io4.o, initialBlock: true)
        
        let io8 = ioChannels(for: .x8)
        x8Block = GBlock(inputChannels: io8.i, outputChannels: io8.o)
        
        let io16 = ioChannels(for: .x16)
        x16Block = GBlock(inputChannels: io16.i, outputChannels: io16.o)
        
        let io32 = ioChannels(for: .x32)
        x32Block = GBlock(inputChannels: io32.i, outputChannels: io32.o)
        
        let io64 = ioChannels(for: .x64)
        x64Block = GBlock(inputChannels: io64.i, outputChannels: io64.o)
        
        let io128 = ioChannels(for: .x128)
        x128Block = GBlock(inputChannels: io128.i, outputChannels: io128.o)
        
        let io256 = ioChannels(for: .x256)
        x256Block = GBlock(inputChannels: io256.i, outputChannels: io256.o)
        
        toRGB = Conv2D(filterShape: (3, 3, baseChannels, 3),
                       padding: .same,
                       activation: tanh,
                       useBias: false,
                       filterInitializer: heNormal())
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input.expandingShape(at: 1, 2)
        
        x = x4Block(x)
        if imageSize == .x4 {
            return toRGB(x)
        }
        
        x = upsample(x)
        x = x8Block(x)
        if imageSize == .x8 {
            return toRGB(x)
        }
        
        x = upsample(x)
        x = x16Block(x)
        if imageSize == .x16 {
            return toRGB(x)
        }
        
        x = upsample(x)
        x = x32Block(x)
        if imageSize == .x32 {
            return toRGB(x)
        }
        
        x = upsample(x)
        x = x64Block(x)
        if imageSize == .x64 {
            return toRGB(x)
        }
        
        x = upsample(x)
        x = x128Block(x)
        if imageSize == .x128 {
            return toRGB(x)
        }
        
        x = upsample(x)
        x = x256Block(x)
        return toRGB(x)
    }
}

extension Generator {
    func inferring(from input: Tensor<Float>, batchSize: Int) -> Tensor<Float> {
        let x = input.reshaped(to: [-1, batchSize, input.shape[1]])
        return Tensor(concatenating: (0..<x.shape[0]).map { inferring(from: x[$0]) },
                      alongAxis: 0)
    }
}
