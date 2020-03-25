import Foundation
import TensorFlow
import GANUtils
import ImageLoader
import TensorBoardX

Context.local.randomSeed = (42, 42)
let rng = XorshiftRandomNumberGenerator()

let imageSize: ImageSize = .x256
let latentSize = 256
let batchSize = 16

let config = Config(
    batchSize: batchSize,
    learningRates: GDPair(G: 1e-4, D: 4e-4),
    reparameterizeInGTraining: false,
    imageSize: imageSize,
    G: Generator.Config(
        latentSize: latentSize,
        resizeMethod: .bilinear,
        enableBatchNorm: true
    ),
    D: Discriminator.Config(
        numberOfOutcomes: 16
    )
)

var generator = Generator(config: config.G, imageSize: imageSize)
let avgG = ModelAveraging(average: generator, beta: 0.99)
var discriminator = Discriminator(config: config.D, imageSize: imageSize)

let optG = Adam(for: generator, learningRate: config.learningRates.D, beta1: 0.5, beta2: 0.999)
let optD = Adam(for: discriminator, learningRate: config.learningRates.D, beta1: 0.5, beta2: 0.999)

let realAnchor = createAnchor(numberOfOutcomes: config.D.numberOfOutcomes, center: 1)
let fakeAnchor = createAnchor(numberOfOutcomes: config.D.numberOfOutcomes, center: -1)

// MARK: - Dataset
let args = ProcessInfo.processInfo.arguments
guard args.count == 2 else {
    print("Image directory is not specified.")
    exit(1)
}
print("Search images...")
let imageDir = URL(fileURLWithPath: args[1])
let entries = [Entry](directory: imageDir)
print("\(entries.count) images found")
let loader = ImageLoader(
    entries: entries,
    transforms: [
        Transforms.paddingToSquare(with: 1),
        Transforms.resize(.area, width: imageSize.rawValue, height: imageSize.rawValue),
        Transforms.randomFlipHorizontally()
    ],
    rng: rng
)

// MARK: - Plot
let logdir = URL(fileURLWithPath: "./logdir")
let writer = SummaryWriter(logdir: logdir)
try writer.addJSONText(tag: "config", encodable: config)

// MARK: - Training
func train() {
    Context.local.learningPhase = .training
    
    let inferStep = 10000
    
    var step = 0
    
    for epoch in 0..<1_000_000 {
        loader.shuffle()
        
        for batch in loader.iterator(batchSize: config.batchSize) {
            if step % 1 == 0 {
                print("epoch: \(epoch), step:\(step)")
            }
            
            let reals = 2 * batch.images - 1
            
            trainSingleStep(reals: reals, step: step)
            
            if step % inferStep == 0 {
                infer(step: step)
            }
            
            step += 1
        }
    }
    
    // last inference
    infer(step: step)
}

func trainSingleStep(reals: Tensor<Float>, step: Int) {
    let noise = sampleNoise(size: batchSize, latentSize: latentSize)
    
    let fakePlotPeriod = 1000
    
    // Update discriminator
    discriminator.reparametrize = true
    let 𝛁discriminator = gradient(at: discriminator) { discriminator -> Tensor<Float> in
        let fakes = generator(noise)
        let realScores = discriminator(reals)
        let fakeScores = discriminator(fakes)
        
        let loss = klDivergence(p: realAnchor, q: realScores) + klDivergence(p: fakeAnchor, q: fakeScores)
        
        writer.addScalar(tag: "loss/D", scalar: loss.scalarized(), globalStep: step)
        
        if step % fakePlotPeriod == 0 {
            writer.plotImages(tag: "reals", images: reals, globalStep: step)
            writer.plotImages(tag: "fakes", images: fakes, globalStep: step)
            writer.flush()
        }
        return loss
    }
    optD.update(&discriminator, along: 𝛁discriminator)
    
    // Update Generator
    discriminator.reparametrize = config.reparameterizeInGTraining
    let 𝛁generator = gradient(at: generator) { generator ->Tensor<Float> in
        let fakes = generator(noise)
        let realScores = discriminator(reals)
        let fakeScores = discriminator(fakes)
        
        let loss = klDivergence(p: realScores, q: fakeScores) - klDivergence(p: fakeAnchor, q: fakeScores)
        
        writer.addScalar(tag: "loss/G", scalar: loss.scalarized(), globalStep: step)
        
        return loss
    }
    optG.update(&generator, along: 𝛁generator)
    
    avgG.update(model: generator)
}

let truncationFactor: Float = 0.7
let testNoises = (0..<8).map { _ in sampleNoise(size: 64, latentSize: latentSize) * truncationFactor }
let testGridNoises = (0..<8).map { _ in
    makeGrid(corners: sampleNoise(size: 4, latentSize: latentSize) * truncationFactor,
             gridSize: 8, flatten: true)
}

func infer(step: Int) {
    print("infer...")
    
    for (i, noise) in testNoises.enumerated() {
        let reals = generator.inferring(from: noise, batchSize: batchSize)
        writer.plotImages(tag: "test_random/\(i)", images: reals, colSize: 8, globalStep: step)
    }
    for (i, noise) in testGridNoises.enumerated() {
        let reals = generator.inferring(from: noise, batchSize: batchSize)
        writer.plotImages(tag: "test_intpl/\(i)", images: reals, colSize: 8, globalStep: step)
    }
    
    let generator = avgG.average
    for (i, noise) in testNoises.enumerated() {
        let reals = generator.inferring(from: noise, batchSize: batchSize)
        writer.plotImages(tag: "test_avg_random/\(i)", images: reals, colSize: 8, globalStep: step)
    }
    for (i, noise) in testGridNoises.enumerated() {
        let reals = generator.inferring(from: noise, batchSize: batchSize)
        writer.plotImages(tag: "test_avg_intpl/\(i)", images: reals, colSize: 8, globalStep: step)
    }
    
    writer.flush()
}

train()
writer.close()

