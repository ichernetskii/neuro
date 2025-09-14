import { round } from "../functions/round.ts";

describe("Round Function", () => {
	it("should round to default 2 decimal places", () => {
		expect(round(1.23456)).toBe(1.23);
		expect(round(1.236)).toBe(1.24);
		expect(round(1.2)).toBe(1.2);
	});

	it("should round to specified decimal places", () => {
		expect(round(1.23456, 3)).toBe(1.235);
		expect(round(1.23456, 1)).toBe(1.2);
		expect(round(1.23456, 0)).toBe(1);
	});

	it("should handle negative numbers", () => {
		expect(round(-1.23456, 2)).toBe(-1.23);
		expect(round(-1.236, 2)).toBe(-1.24);
	});

	it("should handle zero", () => {
		expect(round(0, 2)).toBe(0);
		expect(round(0.001, 2)).toBe(0);
	});

	it("should handle large numbers", () => {
		expect(round(1234.5678, 2)).toBe(1234.57);
		expect(round(1234.5678, 0)).toBe(1235);
	});

	it("should handle edge cases", () => {
		expect(round(0.5, 0)).toBe(1);
		expect(round(-0.5, 0)).toBe(-0);
		expect(round(0.25, 1)).toBe(0.3);
	});
});
