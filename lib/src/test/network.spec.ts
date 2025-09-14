import { ActivationFunctionCollection } from "../functions/activation.ts";
import { LossFunctionCollection } from "../functions/loss.ts";
import { LayerConfig, Network } from "../network.ts";

describe("Network", () => {
	it("Initialization", () => {
		const layerConfigs: LayerConfig[] = [
			{ neurons: 3, activationFunction: ActivationFunctionCollection.ReLU },
			{ neurons: 2, activationFunction: ActivationFunctionCollection.ReLU },
			{ neurons: 1, activationFunction: ActivationFunctionCollection.ReLU },
		];
		const network = new Network(layerConfigs);
		expect(network.layers.length).toBe(3);
		expect(network.layers[0].neurons[0].inputs.length).toBe(3);
		expect(network.layers[0].neurons[0].inputs[0].signal).toBe(network.inputSignals[0]);
		expect(network.layers[0].neurons.length).toBe(3);
		expect(network.layers[1].neurons.length).toBe(2);
	});

	it("Forward Propagation", () => {
		const layerConfigs: LayerConfig[] = [
			{ neurons: 2, activationFunction: ActivationFunctionCollection.ReLU },
			{ neurons: 2, activationFunction: ActivationFunctionCollection.ReLU },
			{ neurons: 1, activationFunction: ActivationFunctionCollection.ReLU },
		];
		const network = new Network(layerConfigs, LossFunctionCollection.MSE).setInputSignals([1, 1]);

		network.layers[0].neurons.forEach(neuron => {
			neuron.inputs.forEach(input => {
				input.weight = 1;
			});
		});

		network.layers.forEach(layer => {
			layer.neurons.forEach(neuron => {
				neuron.bias = 0;
				neuron.inputs.forEach(input => {
					input.weight = 1;
				});
			});
		});

		network.forward();

		const outputLayer = network.layers[2];
		expect(outputLayer.neurons[0].output.value).toEqual(8);
	});

	it.each([
		ActivationFunctionCollection.ReLU,
		ActivationFunctionCollection.LeakyReLU,
		ActivationFunctionCollection.Sigmoid,
	])("Backward Propagation with %p", activationFunction => {
		const learningRate = 0.1;
		const inputSignals = [0.1, 0.2, 0.3, 0.4];
		const expectedOutput = [0.5, 0.8];
		const trainingIterations = 500;

		const layerConfigs: LayerConfig[] = [
			{ neurons: 4, activationFunction },
			{ neurons: 6, activationFunction },
			{ neurons: 2, activationFunction },
		];
		const network = new Network(layerConfigs, LossFunctionCollection.MSE).setInputSignals(inputSignals);

		// Initialize weights and biases to avoid randomness in the test
		network.layers.forEach(layer => {
			layer.neurons.forEach(neuron => {
				neuron.bias = 0.01;
				neuron.inputs.forEach(input => {
					input.weight = 0.1;
				});
			});
		});

		for (let i = 0; i < trainingIterations; i++) {
			network.forward().backward(expectedOutput, learningRate);
		}

		network.forward().layers[network.layers.length - 1].neurons.forEach((neuron, neuronIndex) => {
			expect(neuron.output.value).toBeCloseTo(expectedOutput[neuronIndex]);
		});
	});

	it("Softmax activation function", () => {
		const layerConfigs: LayerConfig[] = [
			{ neurons: 2, activationFunction: ActivationFunctionCollection.ReLU },
			{ neurons: 3, activationFunction: ActivationFunctionCollection.Softmax },
		];
		const network = new Network(layerConfigs);
		network.setInputSignals([1, 2]);

		// Set some weights to get predictable outputs
		network.layers[0].neurons.forEach(neuron => {
			neuron.bias = 0;
			neuron.inputs.forEach(input => {
				input.weight = 1;
			});
		});

		network.layers[1].neurons.forEach(neuron => {
			neuron.bias = 0;
			neuron.inputs.forEach(input => {
				input.weight = 1;
			});
		});

		network.forward();

		const outputLayer = network.layers[1];
		const outputs = outputLayer.neurons.map(neuron => neuron.output.value);

		// Check that outputs sum to 1 (softmax property)
		const sum = outputs.reduce((acc, val) => acc + val, 0);
		expect(sum).toBeCloseTo(1, 5);

		// Check that all outputs are positive
		outputs.forEach(output => {
			expect(output).toBeGreaterThan(0);
		});
	});

	it("Cross Entropy with Softmax backward propagation", () => {
		const learningRate = 0.1;
		const inputSignals = [0.1, 0.2, 0.3, 0.4];
		const expectedOutput = [0.1, 0.8, 0.1]; // One-hot like encoding
		const trainingIterations = 100;

		const layerConfigs: LayerConfig[] = [
			{ neurons: 4, activationFunction: ActivationFunctionCollection.ReLU },
			{ neurons: 6, activationFunction: ActivationFunctionCollection.ReLU },
			{ neurons: 3, activationFunction: ActivationFunctionCollection.Softmax },
		];
		const network = new Network(layerConfigs, LossFunctionCollection.CrossEntropy).setInputSignals(inputSignals);

		// Initialize weights and biases to avoid randomness in the test
		network.layers.forEach(layer => {
			layer.neurons.forEach(neuron => {
				neuron.bias = 0.01;
				neuron.inputs.forEach(input => {
					input.weight = 0.1;
				});
			});
		});

		// Train the network
		for (let i = 0; i < trainingIterations; i++) {
			network.forward().backward(expectedOutput, learningRate);
		}

		// Check that the network learned to approximate the expected output
		network.forward();
		const outputs = network.layers[network.layers.length - 1].neurons.map(neuron => neuron.output.value);

		// The highest output should correspond to the highest expected value
		const maxExpectedIndex = expectedOutput.indexOf(Math.max(...expectedOutput));
		const maxOutputIndex = outputs.indexOf(Math.max(...outputs));

		expect(maxOutputIndex).toBe(maxExpectedIndex);
	});

	it("Network with different activation functions per layer", () => {
		const layerConfigs: LayerConfig[] = [
			{ neurons: 3, activationFunction: ActivationFunctionCollection.ReLU },
			{ neurons: 4, activationFunction: ActivationFunctionCollection.Sigmoid },
			{ neurons: 2, activationFunction: ActivationFunctionCollection.Softmax },
		];
		const network = new Network(layerConfigs);

		// Check that each layer has the correct activation function
		expect(network.layers[0].activationFunction).toBe(ActivationFunctionCollection.ReLU);
		expect(network.layers[1].activationFunction).toBe(ActivationFunctionCollection.Sigmoid);
		expect(network.layers[2].activationFunction).toBe(ActivationFunctionCollection.Softmax);
	});

	it("Network constructor with empty layer configs - error handling", () => {
		expect(() => {
			new Network([]);
		}).toThrow("Сеть должна содержать хотя бы один слой");
	});

	it("Network with mixed activation functions - forward and backward propagation", () => {
		const layerConfigs: LayerConfig[] = [
			{ neurons: 2, activationFunction: ActivationFunctionCollection.ReLU },
			{ neurons: 3, activationFunction: ActivationFunctionCollection.LeakyReLU },
			{ neurons: 2, activationFunction: ActivationFunctionCollection.Softmax },
		];
		const network = new Network(layerConfigs);

		// Set input data
		network.setInputSignals([0.5, 0.3]);

		// Initialize weights for predictable results
		network.layers.forEach(layer => {
			layer.neurons.forEach(neuron => {
				neuron.bias = 0.1;
				neuron.inputs.forEach(input => {
					input.weight = 0.2;
				});
			});
		});

		// Forward pass
		network.forward();
		const outputs = network.outputValues;

		// Check that outputs sum to 1 (softmax property)
		const sum = outputs.reduce((acc, val) => acc + val, 0);
		expect(sum).toBeCloseTo(1, 5);
	});

	it("JSON serialization and deserialization with different activation functions", () => {
		const layerConfigs: LayerConfig[] = [
			{ neurons: 2, activationFunction: ActivationFunctionCollection.ReLU },
			{ neurons: 3, activationFunction: ActivationFunctionCollection.Sigmoid },
			{ neurons: 2, activationFunction: ActivationFunctionCollection.Softmax },
		];
		const originalNetwork = new Network(layerConfigs, LossFunctionCollection.CrossEntropy);

		// Set some weights for testing
		originalNetwork.layers.forEach(layer => {
			layer.neurons.forEach(neuron => {
				neuron.bias = 0.1;
				neuron.inputs.forEach(input => {
					input.weight = 0.2;
				});
			});
		});

		// Serialize to JSON
		const json = originalNetwork.toJSON();

		// Check JSON structure
		expect(json.layers).toHaveLength(3);
		expect(json.layers[0].activationFunction).toBe("ReLU");
		expect(json.layers[1].activationFunction).toBe("Sigmoid");
		expect(json.layers[2].activationFunction).toBe("Softmax");
		expect(json.lossFunction).toBe("CrossEntropy");

		// Deserialize from JSON
		const restoredNetwork = Network.fromJSON(json);

		// Check that activation functions are restored correctly
		expect(restoredNetwork.layers[0].activationFunction).toBe(ActivationFunctionCollection.ReLU);
		expect(restoredNetwork.layers[1].activationFunction).toBe(ActivationFunctionCollection.Sigmoid);
		expect(restoredNetwork.layers[2].activationFunction).toBe(ActivationFunctionCollection.Softmax);
		expect(restoredNetwork.lossFunction).toBe(LossFunctionCollection.CrossEntropy);

		// Check that weights and biases are restored correctly
		originalNetwork.layers.forEach((originalLayer, layerIndex) => {
			const restoredLayer = restoredNetwork.layers[layerIndex];
			originalLayer.neurons.forEach((originalNeuron, neuronIndex) => {
				const restoredNeuron = restoredLayer.neurons[neuronIndex];
				expect(restoredNeuron.bias).toBe(originalNeuron.bias);
				originalNeuron.inputs.forEach((originalInput, inputIndex) => {
					expect(restoredNeuron.inputs[inputIndex].weight).toBe(originalInput.weight);
				});
			});
		});
	});

	it("Loss function interface", () => {
		// Test MSE loss function
		const mseLayerConfigs: LayerConfig[] = [
			{ neurons: 2, activationFunction: ActivationFunctionCollection.ReLU },
			{ neurons: 3, activationFunction: ActivationFunctionCollection.ReLU },
			{ neurons: 2, activationFunction: ActivationFunctionCollection.ReLU },
		];
		const mseNetwork = new Network(mseLayerConfigs, LossFunctionCollection.MSE);
		expect(mseNetwork.lossFunction).toBe(LossFunctionCollection.MSE);
		expect(mseNetwork.lossFunction.functionName).toBe("MSE");

		// Test CrossEntropy loss function
		const ceLayerConfigs: LayerConfig[] = [
			{ neurons: 2, activationFunction: ActivationFunctionCollection.ReLU },
			{ neurons: 3, activationFunction: ActivationFunctionCollection.ReLU },
			{ neurons: 2, activationFunction: ActivationFunctionCollection.Softmax },
		];
		const ceNetwork = new Network(ceLayerConfigs, LossFunctionCollection.CrossEntropy);
		expect(ceNetwork.lossFunction).toBe(LossFunctionCollection.CrossEntropy);
		expect(ceNetwork.lossFunction.functionName).toBe("CrossEntropy");
	});
});
