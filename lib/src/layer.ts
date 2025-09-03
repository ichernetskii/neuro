import { type ActivationFunction, ReLU } from "./functions/activation.ts";
import { Neuron, Input } from "./neuron.ts";

export class Layer {
	neurons: Neuron[];

	constructor(neuronsNumber: number, activationFunction: ActivationFunction = ReLU) {
		this.neurons = Array.from({ length: neuronsNumber }, () => new Neuron(activationFunction));
	}

	forward() {
		this.neurons.forEach(neuron => neuron.forward());
		return this;
	}

	connect(previousLayer: Layer) {
		this.neurons.forEach(neuron => {
			// Connect each neuron in the current layer to all neurons in the previous layer
			neuron.inputs = [];
			previousLayer.neurons.forEach(neuronFromPreviousLayer => {
				neuron.inputs.push(new Input(neuronFromPreviousLayer.output));
			});
		});
		return this;
	}
}
