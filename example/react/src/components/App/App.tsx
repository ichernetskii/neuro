import { useStore } from "../../store/Context.tsx";
import { Pixel } from "../Pixel/Pixel.tsx";
import styles from "./App.module.scss";
import { observer } from "mobx-react";
import { type FC, type MouseEventHandler, useEffect, useState, useTransition } from "react";
import { Network } from "../../../../../lib/src/network.ts";
import { throttle } from "../../utils/throttle.ts";
import { IMAGE_HEIGHT, IMAGE_WIDTH } from "../../index.tsx";

const PIXEL_SIZE = 20;

export const App = observer<FC>(() => {
	const { pixels, flatPixels } = useStore();
	const [recognizedValue, setRecognizedValue] = useState<number>();
	const [isPending, startTransition] = useTransition();
	const [network, setNetwork] = useState<Network>();

	useEffect(() => {
		const importModel = async () => {
			const model = await import("../../../../dataset/model.json");
			const network = Network.fromJSON(model.default);
			setNetwork(network);
		};

		importModel();
	}, []);

	useEffect(() => {
		const outputSignals = network?.setInputSignals(flatPixels).forward().getOutputSignals();
		const outputValues = outputSignals?.map(signal => signal.value ?? 0);
		startTransition(() => {
			setRecognizedValue(outputValues?.indexOf(Math.max(...outputValues)));
		});
	}, [flatPixels, network]);

	if (!network) {
		return <div>Loading ...</div>;
	}

	const mouseHandler = throttle<MouseEventHandler>(e => {
		e.preventDefault();
		const { x, y } = e.currentTarget.getBoundingClientRect();
		const column = Math.floor((e.clientX - x) / PIXEL_SIZE);
		const row = Math.floor((e.clientY - y) / PIXEL_SIZE);
		if (column < IMAGE_WIDTH && row < IMAGE_HEIGHT) {
			if (e.buttons === 1) {
				pixels[row][column].setSelected(true);
				pixels[row + 1][column].setSelected(true);
				pixels[row][column + 1].setSelected(true);
				pixels[row + 1][column + 1].setSelected(true);
			}
			if (e.buttons === 2) {
				pixels[row][column].setSelected(false);
				pixels[row + 1][column].setSelected(false);
				pixels[row][column + 1].setSelected(false);
				pixels[row + 1][column + 1].setSelected(false);
			}
		}
	}, 50);

	return (
		<div onMouseDown={mouseHandler} onMouseMove={mouseHandler} onContextMenu={e => e.preventDefault()}>
			{pixels.map((pixelRow, row) => {
				return (
					<div key={row} className={styles.row}>
						{pixelRow.map((pixel, column) => (
							<Pixel
								key={`${column}:${row}`}
								isSelected={pixel.isSelected}
								// setSelected={pixel.setSelected}
							/>
						))}
					</div>
				);
			})}
			<span>Recognized value: {isPending ? "?" : recognizedValue}</span>
		</div>
	);
});

App.displayName = "App";
