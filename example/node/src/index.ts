#!/usr/bin/env node

import { parseArguments, setupSignalHandlers, showHelp, validateOptions } from "./cli.ts";
import { NetworkTrainer } from "./trainer.ts";
import { ImageRecognizer } from "./recognizer.ts";

export async function main(): Promise<void> {
	try {
		setupSignalHandlers();

		const options = parseArguments();

		if (options.help) {
			showHelp();
			return;
		}

		validateOptions(options);

		if (options.mode === "train") {
			await NetworkTrainer.train(options);
		} else if (options.mode === "recognize") {
			const result = await ImageRecognizer.recognizeBatch(options);
			ImageRecognizer.printDetailedStats(result);
		}

		console.log("\n✅ Application completed successfully!");
	} catch (error: any) {
		console.error("❌ Error:", error.message);
		process.exit(1);
	}
}

// Run application if file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
	main();
}
