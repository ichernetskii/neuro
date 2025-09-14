import {
	randomNormal,
	randomUniform,
	randomNormalHe,
	randomNormalXavier,
	randomUniformHe,
	randomUniformXavier,
	randomInt,
} from "../functions/random.ts";

describe("Random Functions", () => {
	describe("randomNormal", () => {
		it("should generate numbers with correct mean and std", () => {
			const values = Array.from({ length: 1000 }, () => randomNormal(5, 2));
			const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
			const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
			const std = Math.sqrt(variance);

			expect(mean).toBeCloseTo(5, 0);
			expect(std).toBeCloseTo(2, 0);
		});

		it("should use default parameters", () => {
			const value = randomNormal();
			expect(typeof value).toBe("number");
			expect(isFinite(value)).toBe(true);
		});
	});

	describe("randomUniform", () => {
		it("should generate numbers within range", () => {
			const values = Array.from({ length: 100 }, () => randomUniform(1, 5));

			values.forEach(value => {
				expect(value).toBeGreaterThanOrEqual(1);
				expect(value).toBeLessThan(5);
			});
		});

		it("should use default parameters", () => {
			const value = randomUniform();
			expect(value).toBeGreaterThanOrEqual(0);
			expect(value).toBeLessThan(1);
		});
	});

	describe("randomNormalHe", () => {
		it("should generate numbers with He initialization", () => {
			const fanIn = 10;
			const values = Array.from({ length: 100 }, () => randomNormalHe(fanIn));
			const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
			const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
			const std = Math.sqrt(variance);

			// He initialization: std = sqrt(2/fanIn)
			const expectedStd = Math.sqrt(2 / fanIn);
			expect(std).toBeCloseTo(expectedStd, 0);
		});
	});

	describe("randomNormalXavier", () => {
		it("should generate numbers with Xavier initialization", () => {
			const fanIn = 10;
			const fanOut = 5;
			const values = Array.from({ length: 100 }, () => randomNormalXavier(fanIn, fanOut));
			const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
			const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
			const std = Math.sqrt(variance);

			// Xavier initialization: std = sqrt(2/(fanIn + fanOut))
			const expectedStd = Math.sqrt(2 / (fanIn + fanOut));
			expect(std).toBeCloseTo(expectedStd, 0);
		});
	});

	describe("randomUniformHe", () => {
		it("should generate numbers within He uniform range", () => {
			const fanIn = 10;
			const values = Array.from({ length: 100 }, () => randomUniformHe(fanIn));
			const limit = Math.sqrt(6 / fanIn);

			values.forEach(value => {
				expect(value).toBeGreaterThanOrEqual(-limit);
				expect(value).toBeLessThanOrEqual(limit);
			});
		});
	});

	describe("randomUniformXavier", () => {
		it("should generate numbers within Xavier uniform range", () => {
			const fanIn = 10;
			const fanOut = 5;
			const values = Array.from({ length: 100 }, () => randomUniformXavier(fanIn, fanOut));
			const limit = Math.sqrt(6 / (fanIn + fanOut));

			values.forEach(value => {
				expect(value).toBeGreaterThanOrEqual(-limit);
				expect(value).toBeLessThanOrEqual(limit);
			});
		});
	});

	describe("randomInt", () => {
		it("should generate integers within range", () => {
			const values = Array.from({ length: 100 }, () => randomInt(1, 5));

			values.forEach(value => {
				expect(Number.isInteger(value)).toBe(true);
				expect(value).toBeGreaterThanOrEqual(1);
				expect(value).toBeLessThanOrEqual(5);
			});
		});

		it("should handle single value range", () => {
			const value = randomInt(5, 5);
			expect(value).toBe(5);
		});
	});
});
