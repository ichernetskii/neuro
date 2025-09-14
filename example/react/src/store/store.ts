import { action, makeAutoObservable } from "mobx";
import { PixelModel } from "../types/types.ts";

export class Store {
	private _pixels: PixelModel[][] = [];

	constructor(columns: number, rows: number) {
		makeAutoObservable(this);
		this._pixels = Array.from({ length: rows }, () => {
			return Array.from({ length: columns }, () => new PixelModel());
		});
	}

	get pixels() {
		return this._pixels;
	}

	get flatPixels() {
		return this._pixels.map(pixelRow => pixelRow.map(pixel => (pixel.isSelected ? 1.0 : 0.0))).flat();
	}

	clear = action(() => {
		this._pixels.forEach(pixelRow => pixelRow.forEach(pixel => pixel.setSelected(false)));
	});

	get isEmpty() {
		return this._pixels.every(pixelRow => pixelRow.every(pixel => !pixel.isSelected));
	}
}
