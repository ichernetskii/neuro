import { type ActivationFunction, ReLU } from "./functions/activation.ts";

export class Signal {
	value: number;

	constructor(value: number = 0) {
		this.value = value;
	}
}

export class Input {
	signal: Signal;
	weight: number;

	constructor(signal: Signal) {
		this.signal = signal;
		this.weight = 0;
	}
}

export class Neuron {
	inputs: Input[];
	bias: number;
	private _preActivation: number | null = null;
	get preActivation() {
		return this._preActivation;
	}
	readonly activationFunction: ActivationFunction;
	private readonly _output: Signal;
	get output(): Readonly<Signal> {
		return this._output;
	}

	constructor(activationFunction: ActivationFunction = ReLU) {
		this.inputs = [];
		this._output = new Signal();
		this.activationFunction = activationFunction;
		this.bias = 0;
	}

	addInputs(signals: Signal[]) {
		this.inputs.push(...signals.map(signal => new Input(signal)));
		return this;
	}

	forward() {
		// output = F(bias + Σ(input_i * weight_i))
		this._preActivation =
			this.bias +
			this.inputs.reduce((acc, input) => {
				return acc + input.weight * input.signal.value;
			}, 0);
		this._output.value = this.activationFunction(this._preActivation);
		return this;
	}
}
