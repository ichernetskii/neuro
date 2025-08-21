import { ActivationFunction, ActivationFunctionCollection } from "./ActivationFunction.ts";
import { Input, Neuron } from "./Neuron.ts";

export class Layer {
	neurons: Neuron[];

	constructor(
		neuronsNumber: number,
		activationFunction: ActivationFunction = ActivationFunctionCollection.ReLU.function,
	) {
		this.neurons = Array.from({ length: neuronsNumber }, () => new Neuron(activationFunction));
	}

	addNeuron(activationFunction: ActivationFunction = ActivationFunctionCollection.ReLU.function) {
		this.neurons.push(new Neuron(activationFunction));
		return this;
	}

	process() {
		this.neurons.forEach(neuron => neuron.process());
		return this;
	}

	addInputToAllNeurons() {
		this.neurons.forEach(neuron => neuron.addInput());
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
