import Foundation
import TensorFlow
import GANUtils

struct DBlock: Layer {
    var conv1: Conv2D<Float>
    var conv2: Conv2D<Float>
    var shortcut: Conv2D<Float>
    
    @noDerivative
    let learnableSC: Bool
    
    var avgPool = AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    
    init(
        inputChannels: Int,
        outputChannels: Int
    ) {
        conv1 = Conv2D(filterShape: (3, 3, inputChannels, outputChannels),
                         padding: .same,
                         filterInitializer: heNormal())
        conv2 = Conv2D(filterShape: (4, 4, outputChannels, outputChannels),
                         strides: (2, 2),
                         padding: .same,
                         filterInitializer: heNormal())
        
        learnableSC = inputChannels != outputChannels
        shortcut = Conv2D(filterShape: (1, 1, inputChannels, learnableSC ? outputChannels : 0),
                            useBias: false,
                            filterInitializer: heNormal())
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        x = conv1(leakyRelu(x))
        x = conv2(leakyRelu(x))
        
        var sc = avgPool(input)
        if learnableSC {
            sc = shortcut(sc)
        }
        
        return 0.1*x + sc
    }
}

struct Discriminator: Layer {
    struct Config: Codable {
        var numberOfOutcomes: Int
        var baseChannels: Int = 8
        var maxChannels: Int = 256
    }
    
    var x256Block: DBlock
    var x128Block: DBlock
    var x64Block: DBlock
    var x32Block: DBlock
    var x16Block: DBlock
    var x8Block: DBlock
//    var x4Block: DBlock
    
    var norm: InstanceNorm<Float>
    
    var meanConv: Conv2D<Float>
    var logVarConv: Conv2D<Float>
    
    var fromRGB: Conv2D<Float>
    
    var avgPool: AvgPool2D<Float> = AvgPool2D(poolSize: (2, 2), strides: (2, 2))
    
    @noDerivative
    private let imageSize: ImageSize
    
    @noDerivative
    var reparametrize: Bool = false
    
    public init(config: Config, imageSize: ImageSize) {
        self.imageSize = imageSize
        
        func ioChannels(for size: ImageSize) -> (i: Int, o: Int) {
            guard size <= imageSize else {
                return (0, 0)
            }
            let d = imageSize.log2 - size.log2
            let i = config.baseChannels * 1 << d
            let o = i * 2
            return (min(i, config.maxChannels), min(o, config.maxChannels))
        }
        
        fromRGB = Conv2D(filterShape: (1, 1, 3, config.baseChannels),
                           filterInitializer: heNormal())
        
        let io256 = ioChannels(for: .x256)
        x256Block = DBlock(inputChannels: io256.i, outputChannels: io256.o)
        
        let io128 = ioChannels(for: .x128)
        x128Block = DBlock(inputChannels: io128.i, outputChannels: io128.o)
        
        let io64 = ioChannels(for: .x64)
        x64Block = DBlock(inputChannels: io64.i, outputChannels: io64.o)
        
        let io32 = ioChannels(for: .x32)
        x32Block = DBlock(inputChannels: io32.i, outputChannels: io32.o)
        
        let io16 = ioChannels(for: .x16)
        x16Block = DBlock(inputChannels: io16.i, outputChannels: io16.o)
        
        let io8 = ioChannels(for: .x8)
        x8Block = DBlock(inputChannels: io8.i, outputChannels: io8.o)
        
        norm = InstanceNorm(featureCount: io8.o)
        
        meanConv = Conv2D(filterShape: (4, 4, io8.0, config.numberOfOutcomes),
                          filterInitializer: heNormal())
        logVarConv = Conv2D(filterShape: (4, 4, io8.0, config.numberOfOutcomes),
                            filterInitializer: heNormal())
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        
        x = fromRGB(x)
        
        if imageSize >= .x256 {
            x = x256Block(x)
        }
        if imageSize >= .x128 {
            x = x128Block(x)
        }
        if imageSize >= .x64 {
            x = x64Block(x)
        }
        if imageSize >= .x32 {
            x = x32Block(x)
        }
        if imageSize >= .x16 {
            x = x16Block(x)
        }
        if imageSize >= .x8 {
            x = x8Block(x)
        }
        
        // The variance of the output of resnet can be large in early steps.
        x = norm(x)
        
        let mean = meanConv(x).squeezingShape(at: 1, 2)
        
        if reparametrize {
            let logVar = logVarConv(x).squeezingShape(at: 1, 2)
            let noise = Tensor<Float>(randomNormal: mean.shape)
            x = noise * exp(0.5 * logVar) + mean
        } else {
            x = mean
        }
        return softmax(x, alongAxis: 1)
    }
}