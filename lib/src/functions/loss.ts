export type LossFunctionName = "MSE" | "CrossEntropy";

export interface LossFunction {
	(predicted: number[], expected: number[]): number;
	derivative(predicted: number[], expected: number[]): number[];
	functionName: LossFunctionName;
}

// Create loss functions
const MSE: LossFunction = (predicted: number[], expected: number[]): number => {
	let loss = 0;
	for (let i = 0; i < predicted.length; i++) {
		const error = predicted[i] - expected[i];
		loss += 0.5 * error * error;
	}
	return loss;
};
MSE.derivative = (predicted: number[], expected: number[]): number[] => {
	return predicted.map((pred, index) => pred - expected[index]);
};
MSE.functionName = "MSE";

const CrossEntropy: LossFunction = (predicted: number[], expected: number[]): number => {
	let loss = 0;
	for (let i = 0; i < predicted.length; i++) {
		// Add small epsilon to avoid log(0)
		loss -= expected[i] * Math.log(predicted[i] + 1e-15);
	}
	return loss;
};
CrossEntropy.derivative = (predicted: number[], expected: number[]): number[] => {
	return predicted.map((pred, index) => pred - expected[index]);
};
CrossEntropy.functionName = "CrossEntropy";

export abstract class LossFunctionCollection {
	static MSE: LossFunction = MSE;
	static CrossEntropy: LossFunction = CrossEntropy;

	static get(name: LossFunctionName): LossFunction {
		switch (name) {
			case "MSE":
				return MSE;
			case "CrossEntropy":
				return CrossEntropy;
			default:
				throw new Error(`Unknown loss function: ${name}`);
		}
	}

	static all(): LossFunction[] {
		return [MSE, CrossEntropy];
	}

	static names(): LossFunctionName[] {
		return ["MSE", "CrossEntropy"];
	}
}
