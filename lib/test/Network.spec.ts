import { Network } from "../Network.ts";

describe("Network", () => {
	test("Initialization", () => {
		const network = new Network([3, 2, 1]);
		expect(network.layers.length).toBe(3);
		expect(network.layers[0].neurons.length).toBe(3);
		expect(network.layers[1].neurons.length).toBe(2);
	});

	test("Add Layer", () => {
		const network = new Network([2]);
		network.addLayer(3);
		expect(network.layers.length).toBe(2);
		expect(network.layers[1].neurons.length).toBe(3);
		expect(network.layers[0].neurons[0].output).toBe(network.layers[1].neurons[0].inputs[0].signal);
		expect(network.layers[1].neurons[0].inputs.length).toBe(2);
	});

	test("Process", () => {
		const network = new Network([2, 2, 1]);
		const inputLayer = network.layers[0];

		inputLayer.neurons[0].inputs[0] = { signal: { value: 1 }, weight: 1 };
		inputLayer.neurons[1].inputs[0] = { signal: { value: 1 }, weight: 1 };

		network.layers.forEach(layer => {
			layer.neurons.forEach(neuron => {
				neuron.bias = 0;
				neuron.inputs.forEach(input => {
					input.weight = 1;
				});
			});
		});

		network.process();

		const outputLayer = network.layers[2];
		expect(outputLayer.neurons[0].output.value).toEqual(4);
	});
});
