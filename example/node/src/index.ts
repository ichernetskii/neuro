#!/usr/bin/env node

import fs from "fs";
import { PNG } from "pngjs";
import { getDirectories, getFiles } from "./utils.ts";
import { randomInt } from "neuro-lib/src/functions/random.ts";
import { Network } from "neuro-lib/src/network.ts";
import { ReLU } from "neuro-lib/src/functions/activation.ts";
import { round } from "neuro-lib/src/functions/round.ts";

interface Options {
	mode: "train" | "recognize";
	help: boolean;
	model: string;
	folder: string;
	layers?: string;
	epochs: number;
	speed: number;
}

export function parseArguments() {
	const args = process.argv.slice(2);
	const options: Options = {
		mode: "recognize",
		model: "",
		folder: "",
		help: false,
		epochs: 100,
		speed: 0.001,
	};

	for (let i = 0; i < args.length; i++) {
		const arg = args[i];

		switch (arg) {
			case "-t":
			case "--train":
				options.mode = "train";
				break;
			case "-r":
			case "--recognize":
				options.mode = "recognize";
				break;
			case "-h":
			case "--help":
				options.help = true;
				break;
			case "-f":
			case "--folder":
				options.folder = args[++i] || "";
				break;
			case "-m":
			case "--model":
				options.model = args[++i] || "";
				break;
			case "-l":
			case "--layers":
				options.layers = args[++i] || "";
				break;
			case "-e":
			case "--epochs":
				options.epochs = parseInt(args[++i] || "100", 10);
				if (isNaN(options.epochs) || options.epochs <= 0) {
					throw new Error(`Invalid number of epochs: ${args[i]}. Please provide a positive integer.`);
				}
				break;
			case "-s":
			case "--speed":
				options.speed = parseFloat(args[++i] || "0.001");
				if (isNaN(options.speed) || options.speed <= 0) {
					throw new Error(`Invalid learning speed: ${args[i]}. Please provide a positive number.`);
				}
				break;
			default:
				if (!arg.startsWith("-")) {
					options.folder = arg;
				}
				break;
		}
	}

	return options;
}

export function showHelp() {
	console.log(`
== Image recognizer application ==

Usage: node index.ts [options] [name]

Options:
  -t, --train               Enable training mode
  -r, --recognize           Enable recognition mode
  -m, --model  <fileName>   Specify model filename
  -h, --help                Show this help message
  
  Training mode options:
  -f, --folder <folderName> Specify folder with images to train the model
  -l, --layers <layers>    	Specify network layers to create a new model if needed (784,512,256,96,10)
  -e, --epochs <number>     Specify number of training epochs (default: 100)
  -s, --speed   <number>    Specify learning speed (default: 0.001)
  
  Recognition mode options:
  -f, --folder <folderName> Specify folder with images to recognize

Examples:
  node index.ts -t -m model.json -f ./images -e 100 -r 0.001
  node index.ts -t -m model.json -f ./images -l 784,512,256,96,10 -e 100 -r 0.001
  node index.ts -r -m model.json -f ./images
  node index.ts -h
    `);
}

function saveModel(network: Network, fileName: string) {
	// save the model to options.output JSON file
	const modelJson = JSON.stringify(network.toJSON(), null, 0);
	fs.writeFileSync(fileName, modelJson);
	console.log(`💾 Model saved to ${fileName}`);
}

async function train(options: Options) {
	console.log("🛠️ Training mode selected");

	let network: Network;
	if (!options.layers) {
		// load existing model to continue training
		const modelJson = fs.readFileSync(options.model, "utf-8");
		const modelData = JSON.parse(modelJson);
		network = Network.fromJSON(modelData);
		console.log(`💾 Existing model loaded from ${options.model}`);
	} else {
		// create a new network
		console.log("🆕 Creating a new neural network");
		const neuronsNumberPerLayer = options.layers
			.split(",")
			.map(neuronsNumber => parseInt(neuronsNumber.trim(), 10));
		network = new Network(neuronsNumberPerLayer, ReLU);
		console.log(`🆕 New network created with layers: [${neuronsNumberPerLayer.join(", ")}]`);
	}

	// prepare training data from options.folder
	console.log(`📂 Reading images from folder: ${options.folder}`);
	const trainingPathSet = await createPathSet(options);
	console.log(`📂 Found ${trainingPathSet.length} images.`);

	// shuffle the training data
	for (let i = trainingPathSet.length - 1; i > 0; i--) {
		const randomIndex = randomInt(0, i);
		[trainingPathSet[i], trainingPathSet[randomIndex]] = [trainingPathSet[randomIndex], trainingPathSet[i]];
	}
	console.log(`🔀 Shuffled the training data.`);

	// train the network
	const epochs = options.epochs;
	const learningRate = options.speed;
	for (let epoch = 0; epoch < epochs; epoch++) {
		console.log(`🚀 Starting epoch ${epoch + 1}/${epochs} at ${new Date().toLocaleString()}`);
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
		saveModel(network, options.model);
	}

	console.log("✅ Training completed");
}

async function recognize(options: Options) {
	console.log("🔍 Recognition mode selected");

	const modelJson = fs.readFileSync(options.model, "utf-8");
	const modelData = JSON.parse(modelJson);
	const network = Network.fromJSON(modelData);
	console.log(`💾 Model loaded from ${options.model}`);

	// load and preprocess the images from options.folder
	console.log(`📂 Reading images from folder: ${options.folder}`);
	const recognizingPathSet = await createPathSet(options);
	console.log(`📂 Found ${recognizingPathSet.length} images.`);

	let correctAnswers = 0;
	let i = 0;
	const setLength = recognizingPathSet.length;
	for (const [result, imageData] of recognizingPathSet) {
		const input = imageData.flat();

		// recognize the image
		const outputSignals = network.setInputSignals(input).forward().getOutputSignals();
		const outputValues = outputSignals.map(signal => signal.value);
		const recognizedIndex = outputValues.indexOf(Math.max(...outputValues));
		if (recognizedIndex === result) {
			correctAnswers++;
		}
		if ((i + 1) % 1000 === 0) {
			console.log(`Recognized ${i + 1} images of ${setLength}`);
		}
		i++;
	}
	const rate = correctAnswers / setLength;
	console.log(`Recognized ${correctAnswers} images of ${setLength}. Successful rate: ${round(rate * 100, 2)}%`);
}

async function createPathSet(options: Options) {
	const folders = await getDirectories(options.folder);
	const trainingPathSet: [result: number, imageData: number[][]][] = [];
	for (const folder of folders) {
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
	return trainingPathSet;
}

export function validateOptions(options: Options) {
	if (options.help) {
		return;
	}
	if (!options.folder) {
		throw new Error("Specify folder name with -f or --folder.");
	}
	if (!options.model) {
		throw new Error("Specify model filename with -m or --model.");
	}
	if (options.mode === "train") {
		if (options.epochs <= 0) {
			throw new Error("Invalid number of epochs. Please provide a positive integer.");
		}
		if (options.speed <= 0) {
			throw new Error("Invalid learning rate. Please provide a positive number.");
		}
		if (!fs.existsSync(options.model) && !options.layers) {
			throw new Error(`Input file not found: ${options.model}. You can create a new network with -l or --layers`);
		}
		if (options.layers) {
			const neuronsNumberPerLayer = options.layers
				.split(",")
				.map(neuronsNumber => parseInt(neuronsNumber.trim(), 10));
			if (neuronsNumberPerLayer.length < 2) {
				throw new Error("Network must have at least input and output layers.");
			}
			if (neuronsNumberPerLayer.some(n => isNaN(n) || n <= 0)) {
				throw new Error("Invalid layer sizes. Please provide positive integers separated by commas.");
			}
		}
	}

	return options;
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
			await train(options);
		} else if (options.mode === "recognize") {
			await recognize(options);
		}

		console.log("\n✅ Application completed successfully!");
	} catch (error: any) {
		console.error("❌ Error:", error.message);
		process.exit(1);
	}
}

if (import.meta.url === `file://${process.argv[1]}`) {
	main();
}
