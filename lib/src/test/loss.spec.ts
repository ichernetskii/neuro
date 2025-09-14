import { LossFunctionCollection } from "../functions/loss.ts";

describe("Loss Functions", () => {
	describe("MSE", () => {
		it("should calculate MSE loss correctly", () => {
			const predicted = [1, 2, 3];
			const expected = [1, 1, 1];
			const loss = LossFunctionCollection.MSE(predicted, expected);

			// MSE = 0.5 * ((1-1)² + (2-1)² + (3-1)²) = 0.5 * (0 + 1 + 4) = 2.5
			expect(loss).toBe(2.5);
		});

		it("should calculate MSE derivative correctly", () => {
			const predicted = [1, 2, 3];
			const expected = [1, 1, 1];
			const derivatives = LossFunctionCollection.MSE.derivative(predicted, expected);

			// Derivative = predicted - expected = [1-1, 2-1, 3-1] = [0, 1, 2]
			expect(derivatives).toEqual([0, 1, 2]);
		});

		it("should have correct function name", () => {
			expect(LossFunctionCollection.MSE.functionName).toBe("MSE");
		});
	});

	describe("CrossEntropy", () => {
		it("should calculate CrossEntropy loss correctly", () => {
			const predicted = [0.1, 0.8, 0.1];
			const expected = [0, 1, 0];
			const loss = LossFunctionCollection.CrossEntropy(predicted, expected);

			// CrossEntropy = -sum(expected * log(predicted))
			// = -(0*log(0.1) + 1*log(0.8) + 0*log(0.1))
			// = -log(0.8) ≈ 0.223
			expect(loss).toBeCloseTo(0.223, 2);
		});

		it("should handle zero predictions with epsilon", () => {
			const predicted = [0, 1, 0];
			const expected = [0, 1, 0];
			const loss = LossFunctionCollection.CrossEntropy(predicted, expected);

			// Should not be NaN or Infinity
			expect(isFinite(loss)).toBe(true);
		});

		it("should calculate CrossEntropy derivative correctly", () => {
			const predicted = [0.1, 0.8, 0.1];
			const expected = [0, 1, 0];
			const derivatives = LossFunctionCollection.CrossEntropy.derivative(predicted, expected);

			// Derivative = predicted - expected = [0.1-0, 0.8-1, 0.1-0] = [0.1, -0.2, 0.1]
			expect(derivatives[0]).toBeCloseTo(0.1, 5);
			expect(derivatives[1]).toBeCloseTo(-0.2, 5);
			expect(derivatives[2]).toBeCloseTo(0.1, 5);
		});

		it("should have correct function name", () => {
			expect(LossFunctionCollection.CrossEntropy.functionName).toBe("CrossEntropy");
		});
	});

	describe("LossFunctionCollection class", () => {
		it("should return MSE for 'MSE'", () => {
			const func = LossFunctionCollection.get("MSE");
			expect(func).toBe(LossFunctionCollection.MSE);
		});

		it("should return CrossEntropy for 'CrossEntropy'", () => {
			const func = LossFunctionCollection.get("CrossEntropy");
			expect(func).toBe(LossFunctionCollection.CrossEntropy);
		});

		it("should return all loss functions", () => {
			const allFunctions = LossFunctionCollection.all();
			expect(allFunctions).toHaveLength(2);
			expect(allFunctions).toContain(LossFunctionCollection.MSE);
			expect(allFunctions).toContain(LossFunctionCollection.CrossEntropy);
		});

		it("should return all loss function names", () => {
			const names = LossFunctionCollection.names();
			expect(names).toEqual(["MSE", "CrossEntropy"]);
		});
	});
});
