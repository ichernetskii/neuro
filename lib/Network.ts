import { Layer } from "./Layer.ts";
import { ActivationFunction, ActivationFunctionCollection } from "./ActivationFunction.ts";

export class Network {
	layers: Layer[];

	constructor(
		neuronsNumberPerLayer: number[],
		activationFunction: ActivationFunction = ActivationFunctionCollection.ReLU.function,
	) {
		this.layers = neuronsNumberPerLayer.map(count => new Layer(count, activationFunction));

		// Add connections between each layer
		if (this.layers.length > 0) {
			// Add inputs for each neuron in the first layer
			this.layers[0].addInputToAllNeurons();

			// Connect each layer to the previous one
			if (this.layers.length > 1) {
				for (let i = 1; i < this.layers.length; i++) {
					this.layers[i].connect(this.layers[i - 1]);
				}
			}
		}
	}

	addLayer(
		neuronsNumber: number,
		activationFunction: ActivationFunction = ActivationFunctionCollection.ReLU.function,
	) {
		this.layers.push(new Layer(neuronsNumber, activationFunction));

		// Connect with the previous layer
		if (this.layers.length > 1) {
			this.layers[this.layers.length - 1].connect(this.layers[this.layers.length - 2]);
		}

		return this;
	}

	process() {
		this.layers.forEach(layer => layer.process());
	}
}
