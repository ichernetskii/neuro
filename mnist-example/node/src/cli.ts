import fs from "fs";

export interface Options {
	mode: "train" | "recognize";
	help: boolean;
	model: string;
	folder: string;
	layers?: string;
	activations?: string;
	epochs: number;
	speed: number;
}

export function parseArguments(): Options {
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
			case "-a":
			case "--activations":
				options.activations = args[++i] || "";
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

export function showHelp(): void {
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
  -a, --activations <activations> Specify activation functions for each layer (ReLU,LeakyReLU,Sigmoid,Softmax)
  -e, --epochs <number>     Specify number of training epochs (default: 100)
  -s, --speed   <number>    Specify learning speed (default: 0.001)
  
  Recognition mode options:
  -f, --folder <folderName> Specify folder with images to recognize

Examples:
  node index.ts -t -m model.json -f ./images -e 100 -s 0.001
  node index.ts -t -m model.json -f ./images -l 784,512,256,96,10 -e 100 -s 0.001
  node index.ts -t -m model.json -f ./images -l 784,128,10 -a ReLU,ReLU,Softmax -e 50 -s 0.001
  node index.ts -r -m model.json -f ./images
  node index.ts -h
    `);
}

export function validateOptions(options: Options): Options {
	if (options.help) {
		return options;
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

			// Validate activation functions
			if (options.activations) {
				const activationFunctions = options.activations.split(",").map(act => act.trim());
				const validActivations = ["ReLU", "LeakyReLU", "Sigmoid", "Softmax"];

				if (activationFunctions.length !== neuronsNumberPerLayer.length) {
					throw new Error(
						`Number of activation functions (${activationFunctions.length}) must match number of layers (${neuronsNumberPerLayer.length}).`,
					);
				}

				if (activationFunctions.some(act => !validActivations.includes(act))) {
					throw new Error(`Invalid activation function. Valid options: ${validActivations.join(", ")}`);
				}
			}
		}
	}

	return options;
}

export function setupSignalHandlers(): void {
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
