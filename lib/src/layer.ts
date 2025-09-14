import { type ActivationFunction, ActivationFunctionCollection } from "./functions/activation.ts";
import { Input, Neuron } from "./neuron.ts";

export class Layer {
	neurons: Neuron[];
	activationFunction: ActivationFunction;

	constructor(neuronsNumber: number, activationFunction: ActivationFunction = ActivationFunctionCollection.ReLU) {
		this.activationFunction = activationFunction;
		this.neurons = Array.from({ length: neuronsNumber }, () => new Neuron());
	}

	forward() {
		// First, compute pre-activations for all neurons
		this.neurons.forEach(neuron => neuron.computePreActivation());

		// Then apply activation function to all pre-activations
		const preActivations = this.neurons.map(neuron => neuron.preActivation || 0);
		const activations = this.activationFunction(preActivations);

		// Set the output values
		this.neurons.forEach((neuron, index) => {
			neuron.setOutputValue(activations[index]);
		});

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
