import Foundation
import GANUtils

struct Config: Codable {
    var batchSize: Int
    var learningRates: GDPair<Float>
    var reparameterizeInGTraining: Bool
    
    var imageSize: ImageSize
    var G: Generator.Config
    var D: Discriminator.Config
}

struct GDPair<T: Codable>: Codable {
    var G: T
    var D: T
    
    init(G: T, D: T) {
        self.G = G
        self.D = D
    }
}
