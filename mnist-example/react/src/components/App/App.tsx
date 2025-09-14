import { useStore } from "neuro-react/src/store/Context.tsx";
import { Pixel } from "neuro-react/src/components/Pixel/Pixel.tsx";
import styles from "./App.module.scss";
import { observer } from "mobx-react";
import { type FC, type MouseEventHandler, useEffect, useState, useTransition } from "react";
import { Network, type NetworkJSON } from "neuro-lib/src/network.ts";
import { IMAGE_HEIGHT, IMAGE_WIDTH } from "neuro-react/src/index.tsx";
import { Statistics, type StatisticsData } from "neuro-react/src/components/Statistics/Statistics.tsx";
import { ActivationFunctionCollection } from "neuro-lib/src/functions/activation.ts";

const PIXEL_SIZE = 20;

export const App = observer<FC>(() => {
	const { pixels, flatPixels, clear, isEmpty } = useStore();
	const [recognizedValue, setRecognizedValue] = useState<number>();
	const [probabilities, setProbabilities] = useState<StatisticsData>();
	const [isPending, startTransition] = useTransition();
	const [network, setNetwork] = useState<Network>();

	useEffect(() => {
		const importModel = async () => {
			const model = (await import("../../../../dataset/model.json")) as { default: NetworkJSON };
			const network = Network.fromJSON(model.default);
			setNetwork(network);
		};

		importModel();

		const onKeyDown = (event: KeyboardEvent) => {
			if (event.key === "Escape") {
				clear();
			}
		};

		document.addEventListener("keydown", onKeyDown);

		return () => {
			document.removeEventListener("keydown", onKeyDown);
		};
	}, [clear]);

	useEffect(() => {
		if (!network) {
			return;
		}
		if (isEmpty) {
			setRecognizedValue(undefined);
			setProbabilities(undefined);
			return;
		}

		const outputValues = network.setInputSignals(flatPixels).forward().outputValues;
		const maxIndex = outputValues.indexOf(Math.max(...outputValues));
		const sum = outputValues.reduce((acc, value) => acc + value, 0);
		const probabilities =
			network.lastLayer.activationFunction === ActivationFunctionCollection.Softmax
				? outputValues // If the model already has Softmax in the output layer, so use the values directly
				: outputValues.map(value => value / sum); // Otherwise, we need to normalize the values
		startTransition(() => {
			setRecognizedValue(maxIndex);
			setProbabilities(Object.fromEntries(probabilities.map((value, index) => [index.toString(), value])));
		});
	}, [flatPixels, isEmpty, network]);

	if (!network) {
		return <div>Loading ...</div>;
	}

	const mouseHandler: MouseEventHandler = e => {
		e.preventDefault();
		const { x, y } = e.currentTarget.getBoundingClientRect();
		const column = Math.floor((e.clientX - x) / PIXEL_SIZE);
		const row = Math.floor((e.clientY - y) / PIXEL_SIZE);
		for (let i = -1; i <= 1; i++) {
			for (let j = -1; j <= 1; j++) {
				const newRow = row + i;
				const newCol = column + j;
				if (newRow >= 0 && newRow < IMAGE_HEIGHT && newCol >= 0 && newCol < IMAGE_WIDTH) {
					if (e.buttons === 1) {
						pixels[newRow][newCol].setSelected(true);
					} else if (e.buttons === 2) {
						pixels[newRow][newCol].setSelected(false);
					}
				}
			}
		}
	};

	return (
		<div className={styles.app}>
			<div
				className={styles.pixels}
				onMouseDown={mouseHandler}
				onMouseMove={mouseHandler}
				onContextMenu={e => e.preventDefault()}
			>
				{pixels.map((pixelRow, row) => {
					return (
						<div key={row} className={styles.row}>
							{pixelRow.map((pixel, column) => (
								<Pixel key={`${column}:${row}`} isSelected={pixel.isSelected} />
							))}
						</div>
					);
				})}
			</div>
			<div className={styles.toolbox}>
				<button className={styles.btn} onClick={clear}>
					Clear
				</button>
				<div className={styles.result}>
					<div>
						Recognized value:{" "}
						<span className={styles.value}>
							{isPending || recognizedValue === undefined ? "?" : recognizedValue}
						</span>
					</div>
					<div className={styles.statistics}>
						{!isPending && !!probabilities && <Statistics statisticsData={probabilities} />}
					</div>
				</div>
			</div>
		</div>
	);
});

App.displayName = "App";
