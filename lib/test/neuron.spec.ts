import { Neuron } from "../neuron.ts";

describe("Neuron", () => {
	it("Forward", () => {
		const neuron = new Neuron();
		const [bias, s1, w1, s2, w2] = [0.42, 0.123, 0.321, 0.789, 0.987];
		neuron.bias = bias;
		neuron.inputs.push(
			{
				signal: { value: s1 },
				weight: w1,
			},
			{
				signal: { value: s2 },
				weight: w2,
			},
		);
		neuron.forward();
		expect(neuron.output.value).toBe(bias + s1 * w1 + s2 * w2);
	});
});
