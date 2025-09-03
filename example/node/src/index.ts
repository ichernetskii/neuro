#!/usr/bin/env node

import { fileURLToPath } from "url";
import { dirname } from "path";
import fs from "fs";
import { PNG } from "pngjs";
import { getDirectories, getFiles } from "./utils.ts";
import { randomInt } from "neuro-lib/src/functions/random.ts";
import { Network } from "neuro-lib/src/network.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

interface Options {
	mode: "train" | "recognize";
	file: string;
	folder: string;
	input: string;
	output: string;
	help: boolean;
}

export function parseArguments() {
	const args = process.argv.slice(2);
	const options: Options = {
		mode: "recognize",
		file: "",
		folder: "",
		input: "",
		output: "",
		help: false,
	};

	for (let i = 0; i < args.length; i++) {
		const arg = args[i];

		switch (arg) {
			case "-m":
			case "--mode":
				const mode = args[++i];
				if (mode === "train" || mode === "recognize") {
					options.mode = mode;
				} else {
					throw new Error(`Invalid mode: ${mode}. Use "train" or "recognize".`);
				}
				break;
			case "-f":
			case "--file":
				options.file = args[++i] || "";
				break;
			case "-F":
			case "--folder":
				options.folder = args[++i] || "";
				break;
			case "-i":
			case "--input":
				options.input = args[++i] || "";
				break;
			case "-o":
			case "--output":
				options.output = args[++i] || "";
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
  -m, --mode <mode>         Set mode: "train" or "recognize" (default: "recognize")
  -f, --file <fileName>     Specify file to recognize
  -F, --folder <folderName> Specify folder with images to teach the model
  -i, --input <fileName>    Specify input file to load the model
  -o, --output <fileName>   Specify output file to save the model
  -h, --help                Show this help message

Examples:
  node index.js -m train -F ./images -o model.json
  node index.js -m recognize -f image.png -i model.json
  node index.js -h
    `);
}

async function train(options: Options) {
	console.log("🛠️ Training mode selected");
	const network = new Network([28 * 28, 128, 64, 10]);

	// prepare training data from options.folder
	const trainingPathSet: [result: number, imageData: number[][]][] = [];
	const folders = await getDirectories(options.folder);
	for (const folder of folders) {
		console.log(`📂 Reading images from folder: ${folder}`);
		const files = await getFiles(`${options.folder}/${folder}`);
		for (const file of files) {
			const filePath = `${options.folder}/${folder}/${file}`;

			const src = fs.createReadStream(filePath);
			const pngImageData = await new Promise<PNG>((resolve, reject) => {
				src.pipe(new PNG())
					.on("parsed", function () {
						resolve(this);
					})
					.on("error", function (err) {
						reject(err);
					});
			});

			const imageData: number[][] = [];
			for (let y = 0; y < pngImageData.height; y++) {
				imageData.push([]);
				for (let x = 0; x < pngImageData.width; x++) {
					const idx = (pngImageData.width * y + x) << 2;
					imageData[y].push(pngImageData.data[idx] / 255); // normalize to [0, 1]
				}
			}

			trainingPathSet.push([Number(folder), imageData]);
		}
	}
	console.log(`📂 Found ${trainingPathSet.length} images in ${folders.length} categories.`);

	// shuffle the training data
	for (let i = trainingPathSet.length - 1; i > 0; i--) {
		const randomIndex = randomInt(0, i);
		[trainingPathSet[i], trainingPathSet[randomIndex]] = [trainingPathSet[randomIndex], trainingPathSet[i]];
	}
	console.log(`🔀 Shuffled the training data.`);

	// train the network
	const epochs = 10;
	const learningRate = 0.001;
	for (let epoch = 0; epoch < epochs; epoch++) {
		console.log(`🚀 Starting epoch ${epoch + 1}/${epochs}`);
		for (let i = 0; i < trainingPathSet.length; i++) {
			const [result, imageData] = trainingPathSet[i];
			const input = imageData.flat();
			const expectedOutput = Array(10).fill(0);
			expectedOutput[result] = 1;

			network.setInputSignals(input).forward().backward(expectedOutput, learningRate);

			if ((i + 1) % 100 === 0 || i === trainingPathSet.length - 1) {
				console.log(`   Processed ${i + 1}/${trainingPathSet.length} images`);
			}
		}
	}

	// save the model to options.output JSON file
	const modelJson = JSON.stringify(network.toJSON(), null, 2);
	fs.writeFileSync(options.output, modelJson);
	console.log(`💾 Model saved to ${options.output}`);

	// save the model to options.output JSON file

	console.log("✅ Training completed");
}

async function recognize(options: Options) {
	console.log("🔍 Recognition mode selected");
	// load the model from options.input JSON file
	if (!fs.existsSync(options.input)) {
		throw new Error(`Input file not found: ${options.input}`);
	}
	const modelJson = fs.readFileSync(options.input, "utf-8");
	const modelData = JSON.parse(modelJson);
	const network = Network.fromJSON(modelData);
	console.log(`💾 Model loaded from ${options.input}`);

	// load and preprocess the image from options.file
	if (!fs.existsSync(options.file)) {
		throw new Error(`Image file not found: ${options.file}`);
	}
	const src = fs.createReadStream(options.file);
	const pngImageData = await new Promise<PNG>((resolve, reject) => {
		src.pipe(new PNG())
			.on("parsed", function () {
				resolve(this);
			})
			.on("error", function (err) {
				reject(err);
			});
	});

	const imageData: number[][] = [];
	for (let y = 0; y < pngImageData.height; y++) {
		imageData.push([]);
		for (let x = 0; x < pngImageData.width; x++) {
			const idx = (pngImageData.width * y + x) << 2;
			imageData[y].push(pngImageData.data[idx] / 255); // normalize to [0, 1]
		}
	}
	const input = imageData.flat();

	// recognize the image
	const outputSignals = network.setInputSignals(input).forward().getOutputSignals();
	const outputValues = outputSignals.map(signal => signal.value ?? 0);
	const recognizedIndex = outputValues.indexOf(Math.max(...outputValues));

	console.log(`🖼️ Recognized image: ${options.file}`);
	console.log(`   Output values: [${outputValues.map(v => v.toFixed(4)).join(", ")}]`);
	console.log(`   Recognized as category: ${recognizedIndex}`);
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
	if (options.mode === "train") {
		if (!options.folder) {
			throw new Error("Training mode requires a folder with images. Use -F or --folder to specify it.");
		}
		if (!options.output) {
			throw new Error(
				"Training mode requires an output file to save the model. Use -o or --output to specify it.",
			);
		}
	}
	if (options.mode === "recognize") {
		if (!options.file) {
			throw new Error("Recognition mode requires a file to recognize. Use -f or --file to specify it.");
		}
		if (!options.input) {
			throw new Error(
				"Recognition mode requires an input file to load the model. Use -i or --input to specify it.",
			);
		}
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

		if (options.mode === "train") {
			// train mode
			await train(options);
		}
		if (options.mode === "recognize") {
			await recognize(options);
		}

		console.log("\n✅ Application completed successfully!");
	} catch (error: any) {
		handleError(error);
	}
}

if (import.meta.url === `file://${process.argv[1]}`) {
	main();
}
