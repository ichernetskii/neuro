import fs from "fs";
import { PNG } from "pngjs";
import { getDirectories, getFiles } from "./utils.ts";

export type ImageData = number[][];
export type TrainingSample = [result: number, imageData: ImageData];

export class ImageProcessor {
	/**
	 * Creates a dataset for training/recognition from a folder with images
	 */
	static async createPathSet(folderPath: string): Promise<TrainingSample[]> {
		const folders = await getDirectories(folderPath);
		const trainingPathSet: TrainingSample[] = [];

		for (const folder of folders) {
			const files = await getFiles(`${folderPath}/${folder}`);
			for (const file of files) {
				const filePath = `${folderPath}/${folder}/${file}`;
				const imageData = await this.loadImage(filePath);
				trainingPathSet.push([Number(folder), imageData]);
			}
		}

		return trainingPathSet;
	}

	/**
	 * Loads an image and converts it to a normalized pixel array
	 */
	static async loadImage(filePath: string): Promise<ImageData> {
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

		const imageData: ImageData = [];
		for (let y = 0; y < pngImageData.height; y++) {
			imageData.push([]);
			for (let x = 0; x < pngImageData.width; x++) {
				const idx = (pngImageData.width * y + x) << 2;
				// Normalize to range [0, 1] - take only red channel
				imageData[y].push(pngImageData.data[idx] / 255);
			}
		}

		return imageData;
	}

	/**
	 * Converts 2D image array to 1D array for neural network input
	 */
	static flattenImage(imageData: ImageData): number[] {
		return imageData.reduce((acc, row) => acc.concat(row), []);
	}

	/**
	 * Creates one-hot encoding for expected result
	 */
	static createExpectedOutput(result: number, numClasses: number = 10): number[] {
		const expectedOutput = Array(numClasses).fill(0);
		expectedOutput[result] = 1;
		return expectedOutput;
	}
}
