import { useStore } from "../../store/Context.tsx";
import { Pixel } from "../Pixel/Pixel.tsx";
import styles from "./App.module.scss";
import { observer } from "mobx-react";
import { type FC, useEffect, useState, useTransition } from "react";
import { Network } from "../../../../../lib/src/network.ts";

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

	return (
		<div>
			{pixels.map((pixelRow, row) => {
				return (
					<div key={row} className={styles.row}>
						{pixelRow.map((pixel, column) => (
							<Pixel
								key={`${column}:${row}`}
								isSelected={pixel.isSelected}
								setSelected={pixel.setSelected}
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
