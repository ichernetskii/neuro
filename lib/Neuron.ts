import { ActivationFunction, ActivationFunctionCollection } from "./ActivationFunction.ts";

interface Signal {
	value: number | null;
}

export class Input {
	signal: Signal;
	weight: number;

	constructor(
		signal: Signal = { value: null },
		weight: number = Math.random() * 0.1 - 0.05, // Random weight in range [-0.05, +0.05)
	) {
		this.signal = signal;
		this.weight = weight;
	}
}

export class Neuron {
	inputs: Input[];
	output: Signal;
	bias: number;
	private readonly activationFunction: ActivationFunction;

	constructor(activationFunction: ActivationFunction = ActivationFunctionCollection.ReLU.function) {
		this.inputs = [];
		this.output = { value: null };
		this.bias = 0.02 * (Math.random() - 0.5); // [-0.01; +0.01)
		this.activationFunction = activationFunction;
	}

	addInput() {
		this.inputs.push(new Input());
		return this;
	}

	process() {
		this.output.value = this.activationFunction(
			this.bias + this.inputs.reduce((acc, val) => acc + val.weight * (val.signal.value ?? 0), 0),
		);
		return this;
	}
}
