export interface ActivationFunction {
	(x: number): number;
	derivative(x: number): number;
}

export const ReLU: ActivationFunction = (x: number) => Math.max(0, x);
ReLU.derivative = x => Number(x >= 0);

export const LeakyReLU: ActivationFunction = x => (x > 0 ? x : 0.01 * x);
LeakyReLU.derivative = x => (x > 0 ? 1 : 0.01);

export const Sigmoid: ActivationFunction = (x: number) => 1 / (1 + Math.exp(-x));
Sigmoid.derivative = x => Sigmoid(x) * (1 - Sigmoid(x));
