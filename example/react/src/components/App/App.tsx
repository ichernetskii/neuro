import { useStore } from "../../store/Context.tsx";
import { Pixel } from "../Pixel/Pixel.tsx";
import styles from "./App.module.scss";
import { observer } from "mobx-react";
import { type FC, type MouseEventHandler, useEffect, useState, useTransition } from "react";
import { Network, type NetworkJSON } from "neuro-lib/src/network.ts";
import { throttle } from "../../utils/throttle.ts";
import { IMAGE_HEIGHT, IMAGE_WIDTH } from "../../index.tsx";
import { Statistics, type StatisticsData } from "../Statistics/Statistics.tsx";
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
		const outputSignals = network.setInputSignals(flatPixels).forward().getOutputSignals();
		const outputValues = outputSignals.map(signal => signal.value);
		startTransition(() => {
			const index = outputValues.indexOf(Math.max(...outputValues));
			setRecognizedValue(index);
			const softmaxArray = ActivationFunctionCollection.Softmax(outputValues);
			setProbabilities(Object.fromEntries(softmaxArray.map((value, index) => [index.toString(), value])));
		});
	}, [flatPixels, isEmpty, network]);

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
				for (let i = -1; i < 2; i++) {
					for (let j = -1; j < 2; j++) {
						pixels[row + i][column + j].setSelected(true);
					}
				}
			}
			if (e.buttons === 2) {
				for (let i = -1; i < 2; i++) {
					for (let j = -1; j < 2; j++) {
						pixels[row + i][column + j].setSelected(false);
					}
				}
			}
		}
	}, 50);

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
