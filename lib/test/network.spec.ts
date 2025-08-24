import { Network } from "../network.ts";
import { LeakyReLU, ReLU, Sigmoid } from "../functions/activation.ts";

describe("Network", () => {
	it("Initialization", () => {
		const network = new Network([3, 2, 1]);
		expect(network.layers.length).toBe(3);
		expect(network.layers[0].neurons[0].inputs.length).toBe(3);
		expect(network.layers[0].neurons[0].inputs[0].signal).toBe(network.inputSignals[0]);
		expect(network.layers[0].neurons.length).toBe(3);
		expect(network.layers[1].neurons.length).toBe(2);
	});

	it("Forward Propagation", () => {
		const network = new Network([2, 2, 1]).setInputSignals([1, 1]);

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

	it.each([ReLU, LeakyReLU, Sigmoid])("Backward Propagation with %p", activationFunction => {
		const learningRate = 0.1;
		const neuronsPerLayer = [4, 6, 2];
		const inputSignals = [0.1, 0.2, 0.3, 0.4];
		const expectedOutput = [0.5, 0.8];
		const trainingIterations = 500;

		const network = new Network(neuronsPerLayer, activationFunction).setInputSignals(inputSignals);

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
});
