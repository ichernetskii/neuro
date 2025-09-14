import { action, makeAutoObservable } from "mobx";

export class PixelModel {
	private _isSelected: boolean;

	constructor() {
		makeAutoObservable(this);
		this._isSelected = false;
	}

	get isSelected() {
		return this._isSelected;
	}

	setSelected = action((isSelected: boolean) => {
		this._isSelected = isSelected;
	});
}
