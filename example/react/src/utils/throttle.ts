// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const throttle = <T extends (...args: any[]) => any>(
	func: T,
	limit: number,
): ((...args: Parameters<T>) => void) => {
	let inThrottle: boolean;
	return function (...args: Parameters<T>): void {
		if (!inThrottle) {
			func(...args);
			inThrottle = true;
			setTimeout(() => (inThrottle = false), limit);
		}
	};
};
