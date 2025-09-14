export type ActivationFunctionName = "ReLU" | "LeakyReLU" | "Sigmoid" | "Softmax";

export interface ActivationFunction {
	(values: number[]): number[];
	derivative(values: number[]): number[];
	functionName: ActivationFunctionName;
}

const ReLU: ActivationFunction = values => {
	return values.map(x => Math.max(0, x));
};
ReLU.derivative = values => {
	return values.map(x => Number(x >= 0));
};
ReLU.functionName = "ReLU";

const LeakyReLU: ActivationFunction = values => {
	return values.map(x => (x > 0 ? x : 0.01 * x));
};
LeakyReLU.derivative = values => {
	return values.map(x => (x > 0 ? 1 : 0.01));
};
LeakyReLU.functionName = "LeakyReLU";

const Sigmoid: ActivationFunction = values => {
	return values.map(x => 1 / (1 + Math.exp(-x)));
};
Sigmoid.derivative = values => {
	return values.map(x => {
		const sigmoid = 1 / (1 + Math.exp(-x));
		return sigmoid * (1 - sigmoid);
	});
};
Sigmoid.functionName = "Sigmoid";

const Softmax: ActivationFunction = values => {
	// Calculate the maximum value for numerical stability
	const maxValue = Math.max(...values);

	// Calculate exponentials and sum
	const exponentials = values.map(x => Math.exp(x - maxValue));
	const sum = exponentials.reduce((acc, val) => acc + val, 0);

	// Return normalized values
	return exponentials.map(exp => exp / sum);
};
Softmax.derivative = values => {
	// For softmax, the derivative is more complex and depends on the loss function
	// This will be handled in the backward pass
	return values.map(() => 1);
};
Softmax.functionName = "Softmax";

export class ActivationFunctionCollection {
	static ReLU: ActivationFunction = ReLU;
	static LeakyReLU: ActivationFunction = LeakyReLU;
	static Sigmoid: ActivationFunction = Sigmoid;
	static Softmax: ActivationFunction = Softmax;

	static get(name: ActivationFunctionName): ActivationFunction {
		switch (name) {
			case "ReLU":
				return ReLU;
			case "LeakyReLU":
				return LeakyReLU;
			case "Sigmoid":
				return Sigmoid;
			case "Softmax":
				return Softmax;
			default:
				throw new Error(`Unknown activation function: ${name}`);
		}
	}
}
