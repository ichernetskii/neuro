export interface ActivationFunction {
	(x: number): number;
	derivative(x: number): number;
	functionName: string;
}

export const ReLU: ActivationFunction = (x: number) => Math.max(0, x);
ReLU.derivative = x => Number(x >= 0);
ReLU.functionName = "ReLU";

export const LeakyReLU: ActivationFunction = x => (x > 0 ? x : 0.01 * x);
LeakyReLU.derivative = x => (x > 0 ? 1 : 0.01);
LeakyReLU.functionName = "LeakyReLU";

export const Sigmoid: ActivationFunction = (x: number) => 1 / (1 + Math.exp(-x));
Sigmoid.derivative = x => Sigmoid(x) * (1 - Sigmoid(x));
Sigmoid.functionName = "Sigmoid";

export const getActivationFunction = (name: string): ActivationFunction => {
	switch (name) {
		case "ReLU":
			return ReLU;
		case "LeakyReLU":
			return LeakyReLU;
		case "Sigmoid":
			return Sigmoid;
		default:
			throw new Error(`Unknown activation function: ${name}`);
	}
};
