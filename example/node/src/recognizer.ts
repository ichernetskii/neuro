import fs from "fs";
import { Network } from "neuro-lib/src/network.ts";
import { round } from "neuro-lib/src/functions/round.ts";
import { ImageProcessor } from "./imageProcessor.ts";
import { type Options } from "./cli.ts";

export interface RecognitionResult {
	correctAnswers: number;
	totalImages: number;
	successRate: number;
	details: Array<{
		expected: number;
		predicted: number;
		confidence: number;
		correct: boolean;
	}>;
}

export class ImageRecognizer {
	/**
	 * Loads model from file
	 */
	static loadModel(modelPath: string): Network {
		const modelJson = fs.readFileSync(modelPath, "utf-8");
		const modelData = JSON.parse(modelJson);
		return Network.fromJSON(modelData);
	}

	/**
	 * Recognizes an image and returns the class index with highest probability
	 */
	static recognizeImage(
		network: Network,
		imageData: number[][],
	): { index: number; confidence: number; probabilities: number[] } {
		const input = ImageProcessor.flattenImage(imageData);

		// Forward pass through network
		const outputSignals = network.setInputSignals(input).forward().getOutputSignals();
		const probabilities = outputSignals.map((signal: any) => signal.value);

		// Find class with maximum probability
		const maxIndex = probabilities.indexOf(Math.max(...probabilities));
		const confidence = probabilities[maxIndex];

		return {
			index: maxIndex,
			confidence,
			probabilities,
		};
	}

	/**
	 * Recognizes a batch of images and returns statistics
	 */
	static async recognizeBatch(options: Options): Promise<RecognitionResult> {
		console.log("🔍 Recognition mode selected");

		// Load model
		const network = this.loadModel(options.model);
		console.log(`💾 Model loaded from ${options.model}`);

		// Load images
		console.log(`📂 Reading images from folder: ${options.folder}`);
		const testData = await ImageProcessor.createPathSet(options.folder);
		console.log(`📂 Found ${testData.length} images.`);

		// Print model information
		this.printModelInfo(network);

		let correctAnswers = 0;
		const details: RecognitionResult["details"] = [];

		console.log("🔍 Starting recognition...");

		for (let i = 0; i < testData.length; i++) {
			const [expectedResult, imageData] = testData[i];

			// Recognize image
			const result = this.recognizeImage(network, imageData);
			const isCorrect = result.index === expectedResult;

			if (isCorrect) {
				correctAnswers++;
			}

			// Save details for analysis
			details.push({
				expected: expectedResult,
				predicted: result.index,
				confidence: result.confidence,
				correct: isCorrect,
			});

			// Progress
			if ((i + 1) % 1000 === 0) {
				console.log(`   Processed ${i + 1}/${testData.length} images`);
			}
		}

		const successRate = correctAnswers / testData.length;

		console.log(`✅ Recognition completed`);
		console.log(`📊 Results: ${correctAnswers}/${testData.length} correct (${round(successRate * 100, 2)}%)`);

		return {
			correctAnswers,
			totalImages: testData.length,
			successRate,
			details,
		};
	}

	/**
	 * Prints detailed recognition statistics
	 */
	static printDetailedStats(result: RecognitionResult): void {
		console.log("\n📈 Detailed Statistics:");

		// Statistics by class
		const classStats = new Map<number, { correct: number; total: number; avgConfidence: number }>();

		result.details.forEach(detail => {
			const expected = detail.expected;
			if (!classStats.has(expected)) {
				classStats.set(expected, { correct: 0, total: 0, avgConfidence: 0 });
			}

			const stats = classStats.get(expected)!;
			stats.total++;
			if (detail.correct) {
				stats.correct++;
			}
			stats.avgConfidence += detail.confidence;
		});

		// Normalize average values
		classStats.forEach(stats => {
			stats.avgConfidence /= stats.total;
		});

		console.log("📊 Per-class accuracy:");
		classStats.forEach((stats, className) => {
			const accuracy = (stats.correct / stats.total) * 100;
			console.log(
				`   Class ${className}: ${stats.correct}/${stats.total} (${round(accuracy, 1)}%) - Avg confidence: ${round(stats.avgConfidence * 100, 1)}%`,
			);
		});

		// Overall statistics
		console.log(`\n📊 Overall accuracy: ${round(result.successRate * 100, 2)}%`);

		// Average confidence
		const avgConfidence =
			result.details.reduce((sum, detail) => sum + detail.confidence, 0) / result.details.length;
		console.log(`📊 Average confidence: ${round(avgConfidence * 100, 1)}%`);
	}

	/**
	 * Prints model information
	 */
	private static printModelInfo(network: Network): void {
		console.log("📊 Model Information:");
		console.log(`   Layers: ${network.layers.length}`);
		console.log(`   Loss Function: ${network.lossFunction.functionName}`);

		network.layers.forEach((layer: any, index: number) => {
			console.log(
				`   Layer ${index + 1}: ${layer.neurons.length} neurons, ${layer.activationFunction.functionName}`,
			);
		});
		console.log();
	}
}
