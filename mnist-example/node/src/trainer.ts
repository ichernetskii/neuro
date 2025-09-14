import fs from "fs";
import { type LayerConfig, Network } from "neuro-lib/src/network.ts";
import { ActivationFunctionCollection, type ActivationFunctionName } from "neuro-lib/src/functions/activation.ts";
import { randomInt } from "neuro-lib/src/functions/random.ts";
import { ImageProcessor } from "./imageProcessor.ts";
import { type Options } from "./cli.ts";

export class NetworkTrainer {
	/**
	 * Creates a new neural network based on layer configuration
	 */
	static createNetwork(layersConfig: string, activationsConfig?: string): Network {
		const neuronsPerLayer = layersConfig.split(",").map(neuronsNumber => parseInt(neuronsNumber.trim(), 10));

		// Determine activation functions
		let activationFunctionNames: ActivationFunctionName[];
		if (activationsConfig) {
			activationFunctionNames = activationsConfig
				.split(",")
				.map(fnName => fnName.trim()) as ActivationFunctionName[];
		} else {
			// Default: ReLU for hidden layers, Softmax for output
			activationFunctionNames = neuronsPerLayer.map((_, index) =>
				index < neuronsPerLayer.length - 1 ? "ReLU" : "Softmax",
			);
		}

		// Create layer configurations
		const layerConfigs: LayerConfig[] = neuronsPerLayer.map((neurons, index) => {
			const activationFunction = ActivationFunctionCollection.get(activationFunctionNames[index]);

			return {
				neurons,
				activationFunction,
			};
		});

		// Loss function is automatically selected based on the last layer
		return new Network(layerConfigs);
	}

	/**
	 * Loads an existing model from file
	 */
	static loadModel(modelPath: string): Network {
		const modelJson = fs.readFileSync(modelPath, "utf-8");
		const modelData = JSON.parse(modelJson);
		return Network.fromJSON(modelData);
	}

	/**
	 * Saves model to file
	 */
	static saveModel(network: Network, fileName: string): void {
		const modelJson = JSON.stringify(network.toJSON(), null, 0);
		fs.writeFileSync(fileName, modelJson);
		console.log(`💾 Model saved to ${fileName}`);
	}

	/**
	 * Shuffles training data array
	 */
	static shuffleData<T>(data: T[]): T[] {
		const shuffled = [...data];
		for (let i = shuffled.length - 1; i > 0; i--) {
			const randomIndex = randomInt(0, i);
			[shuffled[i], shuffled[randomIndex]] = [shuffled[randomIndex], shuffled[i]];
		}
		return shuffled;
	}

	/**
	 * Trains the neural network
	 */
	static async train(options: Options): Promise<void> {
		console.log("🛠️ Training mode selected");

		let network: Network;

		// Load or create model
		if (!options.layers) {
			network = this.loadModel(options.model);
			console.log(`💾 Existing model loaded from ${options.model}`);
		} else {
			network = this.createNetwork(options.layers, options.activations);
			const activationsInfo = options.activations ? ` with activations: [${options.activations}]` : "";
			console.log(`🆕 New network created with layers: [${options.layers}]${activationsInfo}`);
		}

		// Prepare training data
		console.log(`📂 Reading images from folder: ${options.folder}`);
		const trainingData = await ImageProcessor.createPathSet(options.folder);
		console.log(`📂 Found ${trainingData.length} images.`);

		// Shuffle data
		const shuffledData = this.shuffleData(trainingData);
		console.log(`🔀 Shuffled the training data.`);

		// Train network
		const epochs = options.epochs;
		const learningRate = options.speed;

		for (let epoch = 0; epoch < epochs; epoch++) {
			console.log(`🚀 Starting epoch ${epoch + 1}/${epochs} at ${new Date().toLocaleString()}`);

			for (let i = 0; i < shuffledData.length; i++) {
				const [result, imageData] = shuffledData[i];
				const input = ImageProcessor.flattenImage(imageData);
				const expectedOutput = ImageProcessor.createExpectedOutput(result);

				// Forward and backward pass
				network.setInputSignals(input).forward().backward(expectedOutput, learningRate);

				if ((i + 1) % 100 === 0 || i === shuffledData.length - 1) {
					console.log(`   Processed ${i + 1}/${shuffledData.length} images`);
				}
			}

			// Save model after each epoch
			this.saveModel(network, options.model);
		}

		console.log("✅ Training completed");
	}

	/**
	 * Prints model information
	 */
	static printModelInfo(network: Network): void {
		console.log("📊 Model Information:");
		console.log(`   Layers: ${network.layers.length}`);
		console.log(`   Loss Function: ${network.lossFunction.functionName}`);

		network.layers.forEach((layer, index) => {
			console.log(
				`   Layer ${index + 1}: ${layer.neurons.length} neurons, ${layer.activationFunction.functionName}`,
			);
		});
	}
}
