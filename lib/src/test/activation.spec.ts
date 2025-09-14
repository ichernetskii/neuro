import { ActivationFunctionCollection } from "../functions/activation.ts";

describe("Activation Functions", () => {
	describe("ReLU", () => {
		it("should apply ReLU correctly", () => {
			const inputs = [-2, -1, 0, 1, 2];
			const outputs = ActivationFunctionCollection.ReLU(inputs);
			expect(outputs).toEqual([0, 0, 0, 1, 2]);
		});

		it("should calculate ReLU derivative correctly", () => {
			const inputs = [-2, -1, 0, 1, 2];
			const derivatives = ActivationFunctionCollection.ReLU.derivative(inputs);
			expect(derivatives).toEqual([0, 0, 1, 1, 1]);
		});

		it("should have correct function name", () => {
			expect(ActivationFunctionCollection.ReLU.functionName).toBe("ReLU");
		});
	});

	describe("LeakyReLU", () => {
		it("should apply LeakyReLU correctly", () => {
			const inputs = [-2, -1, 0, 1, 2];
			const outputs = ActivationFunctionCollection.LeakyReLU(inputs);
			expect(outputs).toEqual([-0.02, -0.01, 0, 1, 2]);
		});

		it("should calculate LeakyReLU derivative correctly", () => {
			const inputs = [-2, -1, 0, 1, 2];
			const derivatives = ActivationFunctionCollection.LeakyReLU.derivative(inputs);
			expect(derivatives).toEqual([0.01, 0.01, 0.01, 1, 1]);
		});

		it("should have correct function name", () => {
			expect(ActivationFunctionCollection.LeakyReLU.functionName).toBe("LeakyReLU");
		});
	});

	describe("Sigmoid", () => {
		it("should apply Sigmoid correctly", () => {
			const inputs = [-2, 0, 2];
			const outputs = ActivationFunctionCollection.Sigmoid(inputs);

			// Sigmoid(-2) ≈ 0.119, Sigmoid(0) = 0.5, Sigmoid(2) ≈ 0.881
			expect(outputs[0]).toBeCloseTo(0.119, 2);
			expect(outputs[1]).toBeCloseTo(0.5, 2);
			expect(outputs[2]).toBeCloseTo(0.881, 2);
		});

		it("should calculate Sigmoid derivative correctly", () => {
			const inputs = [0];
			const derivatives = ActivationFunctionCollection.Sigmoid.derivative(inputs);

			// Sigmoid'(0) = Sigmoid(0) * (1 - Sigmoid(0)) = 0.5 * 0.5 = 0.25
			expect(derivatives[0]).toBeCloseTo(0.25, 2);
		});

		it("should have correct function name", () => {
			expect(ActivationFunctionCollection.Sigmoid.functionName).toBe("Sigmoid");
		});
	});

	describe("Softmax", () => {
		it("should apply Softmax correctly", () => {
			const inputs = [1, 2, 3];
			const outputs = ActivationFunctionCollection.Softmax(inputs);

			// All outputs should be positive
			outputs.forEach(output => {
				expect(output).toBeGreaterThan(0);
			});

			// Sum should be 1
			const sum = outputs.reduce((acc, val) => acc + val, 0);
			expect(sum).toBeCloseTo(1, 5);
		});

		it("should handle negative inputs", () => {
			const inputs = [-1, 0, 1];
			const outputs = ActivationFunctionCollection.Softmax(inputs);

			const sum = outputs.reduce((acc, val) => acc + val, 0);
			expect(sum).toBeCloseTo(1, 5);
		});

		it("should handle large inputs (numerical stability)", () => {
			const inputs = [100, 101, 102];
			const outputs = ActivationFunctionCollection.Softmax(inputs);

			const sum = outputs.reduce((acc, val) => acc + val, 0);
			expect(sum).toBeCloseTo(1, 5);
		});

		it("should calculate Softmax derivative correctly", () => {
			const inputs = [1, 2, 3];
			const derivatives = ActivationFunctionCollection.Softmax.derivative(inputs);

			// For now, derivative returns array of 1s (simplified implementation)
			expect(derivatives).toEqual([1, 1, 1]);
		});

		it("should have correct function name", () => {
			expect(ActivationFunctionCollection.Softmax.functionName).toBe("Softmax");
		});
	});

	describe("ActivationFunction class", () => {
		it("should return ReLU for 'ReLU'", () => {
			const func = ActivationFunctionCollection.get("ReLU");
			expect(func).toBe(ActivationFunctionCollection.ReLU);
		});

		it("should return LeakyReLU for 'LeakyReLU'", () => {
			const func = ActivationFunctionCollection.get("LeakyReLU");
			expect(func).toBe(ActivationFunctionCollection.LeakyReLU);
		});

		it("should return Sigmoid for 'Sigmoid'", () => {
			const func = ActivationFunctionCollection.get("Sigmoid");
			expect(func).toBe(ActivationFunctionCollection.Sigmoid);
		});

		it("should return Softmax for 'Softmax'", () => {
			const func = ActivationFunctionCollection.get("Softmax");
			expect(func).toBe(ActivationFunctionCollection.Softmax);
		});
	});
});
