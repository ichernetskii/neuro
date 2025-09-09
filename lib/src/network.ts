import { Layer } from "./layer.ts";
import { Signal } from "./neuron.ts";
import { type ActivationFunction, getActivationFunction, ReLU, Sigmoid } from "./functions/activation.ts";
import { randomNormalHe, randomUniform, randomUniformXavier } from "./functions/random.ts";
import { round } from "./functions/round.ts";

export interface NetworkJSON {
	layers: {
		neurons: {
			bias: number;
			inputs: number[];
		}[];
	}[];
	activationFunction: string;
}

export class Network {
	layers: Layer[];
	inputSignals: Signal[] = [];
	activationFunction: ActivationFunction;

	constructor(neuronsNumberPerLayer: number[], activationFunction: ActivationFunction = ReLU) {
		this.layers = neuronsNumberPerLayer.map(count => new Layer(count, activationFunction));
		this.activationFunction = activationFunction;
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
					switch (this.activationFunction) {
						case Sigmoid:
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

	getOutputSignals(): Readonly<Signal>[] {
		return this.layers[this.layers.length - 1].neurons.map(neuron => neuron.output);
	}

	forward() {
		this.layers.forEach(layer => layer.forward());
		return this;
	}

	backward(expectedOutput: number[], learningRate: number = 0.05) {
		// Loss function: Mean Squared Error (MSE): E = 1/2 * (output - expectedOutput)^2
		// Loss function derivative: dE/d(output) = output - expectedOutput
		// neuronInput = bias + Σ(input_i * weight_i)
		// Neuron's error: delta = dE/d(output) * F'(neuronInput)
		// weights update: weight_i -= learningRate * delta * input_i
		// bias update: bias -= learningRate * delta

		if (expectedOutput.length !== this.layers[this.layers.length - 1].neurons.length) {
			throw new Error("Expected output length does not match the network's output layer size.");
		}

		const deltaMatrix: number[][] = this.layers.map(layer => {
			return Array.from({ length: layer.neurons.length }, () => 0);
		});

		// Calculate deltas for the output layer
		const outputLayerIndex = this.layers.length - 1;
		this.layers[outputLayerIndex].neurons.forEach((neuron, neuronIndex) => {
			const outputValue = neuron.output.value;
			const lossFunctionDerivative = outputValue - expectedOutput[neuronIndex];
			const delta = lossFunctionDerivative * neuron.activationFunction.derivative(neuron.preActivation || 0);
			deltaMatrix[outputLayerIndex][neuronIndex] = delta;
		});

		// Calculate deltas for hidden layers
		for (let layerIndex = outputLayerIndex - 1; layerIndex >= 0; layerIndex--) {
			this.layers[layerIndex].neurons.forEach((neuron, neuronIndex) => {
				let sum = 0;
				this.layers[layerIndex + 1].neurons.forEach((nextNeuron, nextNeuronIndex) => {
					const weight = nextNeuron.inputs[neuronIndex].weight;
					const nextDelta = deltaMatrix[layerIndex + 1][nextNeuronIndex];
					sum += weight * nextDelta;
				});
				const delta = sum * neuron.activationFunction.derivative(neuron.preActivation || 0);
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
					bias: round(neuron.bias, fractionDigits),
					inputs: neuron.inputs.map(input => round(input.weight, fractionDigits)),
				})),
			})),
			activationFunction: this.activationFunction.functionName,
		};
	}

	static fromJSON(json: NetworkJSON) {
		const activationFunction = getActivationFunction(json.activationFunction);
		const neuronsNumberPerLayer = json.layers.map(layer => layer.neurons.length);
		const network = new Network(neuronsNumberPerLayer, activationFunction);

		// Load weights and biases
		network.layers.forEach((layer, layerIndex) => {
			layer.neurons.forEach((neuron, neuronIndex) => {
				const storedNeuron = json.layers[layerIndex].neurons[neuronIndex];
				neuron.bias = storedNeuron.bias;
				neuron.inputs.forEach((input, inputIndex) => {
					input.weight = storedNeuron.inputs[inputIndex];
				});
			});
		});

		return network;
	}
}
