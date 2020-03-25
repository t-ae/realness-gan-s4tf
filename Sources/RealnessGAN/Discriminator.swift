import Foundation
import TensorFlow
import GANUtils

struct DBlock: Layer {
    var conv1: SNConv2D<Float>
    var conv2: SNConv2D<Float>
    var shortcut: SNConv2D<Float>
    
    @noDerivative
    let learnableSC: Bool
    @noDerivative
    let resnet: Bool
    
    var avgPool = AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    
    init(
        inputChannels: Int,
        outputChannels: Int,
        resnet: Bool
    ) {
        conv1 = SNConv2D(filterShape: (3, 3, inputChannels, outputChannels),
                         padding: .same,
                         filterInitializer: heNormal())
        conv2 = SNConv2D(filterShape: (4, 4, outputChannels, outputChannels),
                         strides: (2, 2),
                         padding: .same,
                         filterInitializer: heNormal())
        
        learnableSC = (inputChannels != outputChannels) && resnet
        shortcut = SNConv2D(filterShape: (1, 1, inputChannels, learnableSC ? outputChannels : 0),
                            useBias: false,
                            filterInitializer: heNormal())
        self.resnet = resnet
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        x = conv1(leakyRelu(x))
        x = conv2(leakyRelu(x))
        
        guard resnet else {
            return x
        }
        
        var sc = avgPool(input)
        if learnableSC {
            sc = shortcut(sc)
        }
        
        return x + sc
    }
}

struct Discriminator: Layer {
    struct Config: Codable {
        var numberOfOutcomes: Int
        var resnet: Bool
        var baseChannels: Int = 8
        var maxChannels: Int = 256
    }
    
    var fromRGB: SNConv2D<Float>
    
    var x256Block: DBlock
    var x128Block: DBlock
    var x64Block: DBlock
    var x32Block: DBlock
    var x16Block: DBlock
    var x8Block: DBlock
    
    var minibatchStdConcat: MinibatchStdConcat<Float>
    var tail: SNConv2D<Float>
    
    var norm: InstanceNorm<Float>
    
    var meanDense: SNDense<Float>
    var logVarDense: SNDense<Float>
    
    var avgPool: AvgPool2D<Float> = AvgPool2D(poolSize: (2, 2), strides: (2, 2))
    
    @noDerivative
    private let imageSize: ImageSize
    
    @noDerivative
    var reparametrize: Bool = false
    
    public init(config: Config, imageSize: ImageSize) {
        self.imageSize = imageSize
        let resnet = config.resnet
        
        func ioChannels(for size: ImageSize) -> (i: Int, o: Int) {
            guard size <= imageSize else {
                return (0, 0)
            }
            let d = imageSize.log2 - size.log2
            let i = config.baseChannels * 1 << d
            let o = i * 2
            return (min(i, config.maxChannels), min(o, config.maxChannels))
        }
        
        fromRGB = SNConv2D(filterShape: (1, 1, 3, config.baseChannels),
                           filterInitializer: heNormal())
        
        let io256 = ioChannels(for: .x256)
        x256Block = DBlock(inputChannels: io256.i, outputChannels: io256.o, resnet: resnet)
        
        let io128 = ioChannels(for: .x128)
        x128Block = DBlock(inputChannels: io128.i, outputChannels: io128.o, resnet: resnet)
        
        let io64 = ioChannels(for: .x64)
        x64Block = DBlock(inputChannels: io64.i, outputChannels: io64.o, resnet: resnet)
        
        let io32 = ioChannels(for: .x32)
        x32Block = DBlock(inputChannels: io32.i, outputChannels: io32.o, resnet: resnet)
        
        let io16 = ioChannels(for: .x16)
        x16Block = DBlock(inputChannels: io16.i, outputChannels: io16.o, resnet: resnet)
        
        let io8 = ioChannels(for: .x8)
        x8Block = DBlock(inputChannels: io8.i, outputChannels: io8.o, resnet: resnet)
        
        minibatchStdConcat = MinibatchStdConcat(groupSize: 4)
        tail = SNConv2D(filterShape: (4, 4, io8.o + 1, io8.o),
                        filterInitializer: heNormal())
        
        norm = InstanceNorm(featureCount: io8.o)
        
        meanDense = SNDense(inputSize: io8.o, outputSize: config.numberOfOutcomes,
                            weightInitializer: heNormal())
        logVarDense = SNDense(inputSize: io8.o, outputSize: config.numberOfOutcomes,
                              weightInitializer: heNormal())
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
        
        x = leakyRelu(x)
        x = minibatchStdConcat(x)
        x = tail(x)
        
        // The variance of the output of resnet can be large in early steps.
        x = norm(x).squeezingShape(at: 1, 2)
        
        let mean = meanDense(x)
        
        if reparametrize {
            let logVar = logVarDense(x)
            let noise = Tensor<Float>(randomNormal: mean.shape)
            x = noise * exp(0.5 * logVar) + mean
        } else {
            x = mean
        }
        return softmax(x)
    }
}
