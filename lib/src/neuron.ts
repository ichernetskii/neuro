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
	preActivation: number; // calculates in Neuron class
	readonly output: Signal; // calculates in Layer class

	constructor() {
		this.inputs = [];
		this.bias = 0;
		this.preActivation = 0;
		this.output = new Signal();
	}

	addInputs(signals: Signal[]) {
		this.inputs.push(...signals.map(signal => new Input(signal)));
		return this;
	}

	computePreActivation() {
		// Compute pre-activation: bias + Σ(input_i * weight_i)
		this.preActivation =
			this.bias +
			this.inputs.reduce((acc, input) => {
				return acc + input.weight * input.signal.value;
			}, 0);
		return this;
	}
}
