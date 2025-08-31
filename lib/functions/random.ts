export function randomNormal(mean = 0, stdDev = 1): number {
	let u = 0,
		v = 0;
	while (u === 0) u = Math.random();
	while (v === 0) v = Math.random();
	const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
	return z * stdDev + mean;
}

export function randomUniform(min = 0, max = 1): number {
	return Math.random() * (max - min) + min;
}

export function randomNormalHe(fanIn: number): number {
	const stdDev = Math.sqrt(2 / fanIn);
	return randomNormal(0, stdDev);
}

export function randomNormalXavier(fanIn: number, fanOut: number): number {
	const stdDev = Math.sqrt(2 / (fanIn + fanOut));
	return randomNormal(0, stdDev);
}

export function randomUniformHe(fanIn: number): number {
	const limit = Math.sqrt(6 / fanIn);
	return randomUniform(-limit, limit);
}

export function randomUniformXavier(fanIn: number, fanOut: number): number {
	const limit = Math.sqrt(6 / (fanIn + fanOut));
	return randomUniform(-limit, limit);
}

export function randomInt(min: number, max: number): number {
	return Math.floor(Math.random() * (max - min + 1)) + min;
}
