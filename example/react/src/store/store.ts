import { makeAutoObservable } from "mobx";
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
		return this._pixels.map(pixelRow => pixelRow.map(pixel => (pixel.isSelected ? 255 : 0))).flat();
	}
}
