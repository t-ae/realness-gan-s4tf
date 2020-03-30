import Foundation
import TensorFlow
import GANUtils

struct Discriminator: Layer {
    struct Config: Codable {
        var numberOfOutcomes: Int
        var baseChannels: Int = 8
        var maxChannels: Int = 256
    }
    
    var fromRGB: SNConv2D<Float>
    
    var x256Block: SNConv2D<Float>
    var x128Block: SNConv2D<Float>
    var x64Block: SNConv2D<Float>
    var x32Block: SNConv2D<Float>
    var x16Block: SNConv2D<Float>
    var x8Block: SNConv2D<Float>
    
    var meanConv: SNConv2D<Float>
    var logVarConv: SNConv2D<Float>
    
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
        
        fromRGB = SNConv2D(filterShape: (1, 1, 3, config.baseChannels),
                           activation: lrelu,
                           filterInitializer: heNormal())
        
        let io256 = ioChannels(for: .x256)
        x256Block = SNConv2D<Float>(filterShape: (3, 3, io256.i, io256.o),
                                    strides: (2, 2), padding: .same,
                                    activation: lrelu,
                                    filterInitializer: heNormal())
        
        let io128 = ioChannels(for: .x128)
        x128Block = SNConv2D<Float>(filterShape: (3, 3, io128.i, io128.o),
                                    strides: (2, 2), padding: .same,
                                    activation: lrelu,
                                    filterInitializer: heNormal())
        
        let io64 = ioChannels(for: .x64)
        x64Block = SNConv2D<Float>(filterShape: (3, 3, io64.i, io64.o),
                                   strides: (2, 2), padding: .same,
                                   activation: lrelu,
                                   filterInitializer: heNormal())
        
        let io32 = ioChannels(for: .x32)
        x32Block = SNConv2D<Float>(filterShape: (3, 3, io32.i, io32.o),
                                   strides: (2, 2), padding: .same,
                                   activation: lrelu,
                                   filterInitializer: heNormal())
        
        let io16 = ioChannels(for: .x16)
        x16Block = SNConv2D<Float>(filterShape: (3, 3, io16.i, io16.o),
                                   strides: (2, 2), padding: .same,
                                   activation: lrelu,
                                   filterInitializer: heNormal())
        
        let io8 = ioChannels(for: .x8)
        x8Block = SNConv2D<Float>(filterShape: (3, 3, io8.i, io8.o),
                                  strides: (2, 2), padding: .same,
                                  activation: lrelu,
                                  filterInitializer: heNormal())
        
        meanConv = SNConv2D(filterShape: (4, 4, io8.o, config.numberOfOutcomes),
                            filterInitializer: heNormal())
        logVarConv = SNConv2D(filterShape: (4, 4, io8.o, config.numberOfOutcomes),
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
        
        let mean = meanConv(x)
        
        if reparametrize {
            let logVar = logVarConv(x)
            let noise = Tensor<Float>(randomNormal: mean.shape)
            x = noise * exp(0.5 * logVar) + mean
        } else {
            x = mean
        }
        return softmax(x.squeezingShape(at: 1, 2))
    }
}
