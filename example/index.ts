#!/usr/bin/env node

import { fileURLToPath } from "url";
import { dirname } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

interface Options {
	file: string;
	help: boolean;
}

export function parseArguments() {
	const args = process.argv.slice(2);
	const options: Options = {
		file: "",
		help: false,
	};

	for (let i = 0; i < args.length; i++) {
		const arg = args[i];

		switch (arg) {
			case "-f":
			case "--file":
				options.file = args[++i] || "";
				break;
			case "-h":
			case "--help":
				options.help = true;
				break;
			default:
				if (!arg.startsWith("-")) {
					options.file = arg;
				}
				break;
		}
	}

	return options;
}

export function showHelp() {
	console.log(`
== Image recognizer application ==

Usage: node index.js [options] [name]

Options:
  -h, --help                Show this help message
  -f, --file <fileName>     Specify file to recognize

Examples:
  node index.js -f "~/image.jpg"
  node index.js -h
    `);
}

function processFile(options: Options) {
	console.log(`File processing: "${options.file}"`);
}

export function handleError(error: Error) {
	console.error("❌ Error:", error.message);
	process.exit(1);
}

export function setupSignalHandlers() {
	process.on("SIGINT", () => {
		console.log("\n\n🔄 Gracefully shutting down...");
		console.log("👋 Goodbye!");
		process.exit(0);
	});

	process.on("SIGTERM", () => {
		console.log("\n📋 Received SIGTERM, shutting down...");
		process.exit(0);
	});
}

export function validateOptions(options: Options) {
	if (options.file.trim() === "") {
		throw new Error("File name cannot be empty");
	}
}

export async function main() {
	try {
		setupSignalHandlers();

		const options = parseArguments();

		if (options.help) {
			showHelp();
			return;
		}

		validateOptions(options);
		processFile(options);

		console.log("\n✅ Application completed successfully!");
	} catch (error: any) {
		handleError(error);
	}
}

if (import.meta.url === `file://${process.argv[1]}`) {
	main();
}
