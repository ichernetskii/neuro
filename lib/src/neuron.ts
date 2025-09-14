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
	private readonly _output: Signal;
	get output(): Readonly<Signal> {
		return this._output;
	}

	constructor() {
		this.inputs = [];
		this._output = new Signal();
		this.bias = 0;
	}

	addInputs(signals: Signal[]) {
		this.inputs.push(...signals.map(signal => new Input(signal)));
		return this;
	}

	computePreActivation() {
		// Compute pre-activation: bias + Σ(input_i * weight_i)
		this._preActivation =
			this.bias +
			this.inputs.reduce((acc, input) => {
				return acc + input.weight * input.signal.value;
			}, 0);
		return this;
	}

	setOutputValue(value: number) {
		this._output.value = value;
		return this;
	}
}
