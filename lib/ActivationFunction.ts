export type ActivationFunction = (x: number) => number;

export const ActivationFunctionCollection = {
	ReLU: {
		function: x => Math.max(0, x),
		derivative: x => Number(x >= 0),
	},
} as const satisfies Record<string, { function: ActivationFunction; derivative: ActivationFunction }>;
