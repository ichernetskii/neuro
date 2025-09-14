import { Layer } from "./layer.ts";
import { Signal } from "./neuron.ts";
import {
	type ActivationFunction as ActivationFunctionType,
	ActivationFunctionCollection,
	type ActivationFunctionName,
} from "./functions/activation.ts";
import { type LossFunction, LossFunctionCollection, type LossFunctionName } from "./functions/loss.ts";
import { randomNormalHe, randomUniform, randomUniformXavier } from "./functions/random.ts";

export interface LayerConfig {
	neurons: number;
	activationFunction: ActivationFunctionType;
}

export interface NetworkJSON {
	layers: {
		neurons: {
			bias: number;
			weights: number[];
		}[];
		activationFunction: ActivationFunctionName;
	}[];
	lossFunction: LossFunctionName;
}

export class Network {
	layers: Layer[];
	inputSignals: Signal[] = [];
	lossFunction: LossFunction;

	constructor(layerConfigs: LayerConfig[], lossFunction?: LossFunction) {
		if (layerConfigs.length === 0) {
			throw new Error("Network must contain at least one layer");
		}

		// Create layers based on configuration
		this.layers = layerConfigs.map(config => {
			return new Layer(config.neurons, config.activationFunction);
		});

		// Automatically select loss function based on the last layer
		if (lossFunction) {
			this.lossFunction = lossFunction;
		} else {
			this.lossFunction =
				this.lastLayer.activationFunction === ActivationFunctionCollection.Softmax
					? LossFunctionCollection.CrossEntropy
					: LossFunctionCollection.MSE;
		}

		this.initInputs().initWeightAndBiases();
	}

	private initWeightAndBiases() {
		this.layers.forEach((layer, layerIndex) => {
			const fanIn = layerIndex === 0 ? this.inputSignals.length : this.layers[layerIndex - 1].neurons.length;
			const fanOut =
				layerIndex === this.layers.length - 1
					? layer.neurons.length
					: this.layers[layerIndex + 1].neurons.length;

			layer.neurons.forEach(neuron => {
				neuron.bias = randomUniform(0, 0.05); // Small random bias
				neuron.inputs.forEach(input => {
					switch (layer.activationFunction) {
						case ActivationFunctionCollection.Sigmoid:
							input.weight = randomUniformXavier(fanIn, fanOut); // Xavier uniform initialization
							break;
						default:
							input.weight = randomNormalHe(fanIn); // Kaiming He normal initialization
							break;
					}
				});
			});
		});
		return this;
	}

	// Initialize input signals and connect layers
	private initInputs() {
		if (this.layers.length > 0) {
			this.inputSignals = Array.from({ length: this.layers[0].neurons.length }, () => new Signal());
			// Add inputs for each neuron in the first layer
			this.layers[0].neurons.forEach(neuron => {
				neuron.addInputs(this.inputSignals);
			});

			// Connect each layer with the previous one
			if (this.layers.length > 1) {
				for (let i = 1; i < this.layers.length; i++) {
					this.layers[i].connect(this.layers[i - 1]);
				}
			}
		}
		return this;
	}

	setInputSignals(values: number[]) {
		if (values.length !== this.inputSignals.length) {
			throw new Error("Input array length does not match the network's input layer size.");
		}
		this.inputSignals.forEach((signal, index) => {
			signal.value = values[index];
		});
		return this;
	}

	get outputValues(): number[] {
		return this.lastLayer.outputValues;
	}

	get lastLayer(): Layer {
		if (this.layers.length === 0) {
			throw new Error("Network has no layers");
		}
		return this.layers[this.layers.length - 1];
	}

	forward() {
		this.layers.forEach(layer => layer.forward());
		return this;
	}

	backward(expectedOutput: number[], learningRate: number = 0.05) {
		// neuronInput = bias + Σ(input_i * weight_i)
		// Neuron's error: delta = dE/d(output) * F'(neuronInput)
		// weights update: weight_i -= learningRate * delta * input_i
		// bias update: bias -= learningRate * delta

		if (expectedOutput.length !== this.lastLayer.neurons.length) {
			throw new Error("Expected output length does not match the network's output layer size.");
		}

		const deltaMatrix: number[][] = this.layers.map(layer => {
			return Array.from({ length: layer.neurons.length }, () => 0);
		});

		const outputLayerIndex = this.layers.length - 1;
		// Calculate deltas for the output layer
		const lossDerivatives = this.lossFunction.derivative(this.outputValues, expectedOutput);

		// Calculate deltas for output layer
		const outputPreActivations = this.lastLayer.neurons.map(neuron => neuron.preActivation);
		const outputActivationDerivatives = this.lastLayer.activationFunction.derivative(outputPreActivations);

		this.lastLayer.neurons.forEach((_, neuronIndex) => {
			const lossFunctionDerivative = lossDerivatives[neuronIndex];
			const activationDerivative = outputActivationDerivatives[neuronIndex];

			// For Softmax, the derivative is already incorporated in the loss function derivative
			// For other activation functions, multiply by the activation derivative
			const delta =
				this.lastLayer.activationFunction === ActivationFunctionCollection.Softmax
					? lossFunctionDerivative
					: lossFunctionDerivative * activationDerivative;

			deltaMatrix[outputLayerIndex][neuronIndex] = delta;
		});

		// Calculate deltas for hidden layers
		for (let layerIndex = outputLayerIndex - 1; layerIndex >= 0; layerIndex--) {
			const layer = this.layers[layerIndex];
			const preActivations = layer.neurons.map(neuron => neuron.preActivation);
			const activationDerivatives = layer.activationFunction.derivative(preActivations);

			layer.neurons.forEach((_, neuronIndex) => {
				let sum = 0;
				this.layers[layerIndex + 1].neurons.forEach((nextNeuron, nextNeuronIndex) => {
					const weight = nextNeuron.inputs[neuronIndex].weight;
					const nextDelta = deltaMatrix[layerIndex + 1][nextNeuronIndex];
					sum += weight * nextDelta;
				});
				const delta = sum * activationDerivatives[neuronIndex];
				deltaMatrix[layerIndex][neuronIndex] = delta;
			});
		}

		// Update weights and biases
		this.layers.forEach((layer, layerIndex) => {
			layer.neurons.forEach((neuron, neuronIndex) => {
				const delta = deltaMatrix[layerIndex][neuronIndex];
				neuron.inputs.forEach(input => {
					const gradient = delta * input.signal.value;
					input.weight = input.weight - learningRate * gradient;
				});
				neuron.bias = neuron.bias - learningRate * delta;
			});
		});

		return this;
	}

	toJSON(fractionDigits: number = 6): NetworkJSON {
		return {
			layers: this.layers.map(layer => ({
				neurons: layer.neurons.map(neuron => ({
					bias: Math.round(10 ** fractionDigits * neuron.bias),
					weights: neuron.inputs.map(input => Math.round(10 ** fractionDigits * input.weight)),
				})),
				activationFunction: layer.activationFunction.functionName,
			})),
			lossFunction: this.lossFunction.functionName,
		};
	}

	static fromJSON(json: NetworkJSON, fractionDigits: number = 6) {
		// Create layer configurations from JSON
		const layerConfigs: LayerConfig[] = json.layers.map(layer => ({
			neurons: layer.neurons.length,
			activationFunction: ActivationFunctionCollection.get(layer.activationFunction),
		}));

		// Get loss function
		const lossFunction = LossFunctionCollection.get(json.lossFunction);

		const network = new Network(layerConfigs, lossFunction);

		// Load weights and biases
		network.layers.forEach((layer, layerIndex) => {
			layer.neurons.forEach((neuron, neuronIndex) => {
				const storedNeuron = json.layers[layerIndex].neurons[neuronIndex];
				neuron.bias = storedNeuron.bias / 10 ** fractionDigits;
				neuron.inputs.forEach((input, inputIndex) => {
					input.weight = storedNeuron.weights[inputIndex] / 10 ** fractionDigits;
				});
			});
		});

		return network;
	}
}
