import { type FC, memo, type MouseEventHandler } from "react";
import styles from "./Pixel.module.scss";
import cn from "classnames";

interface PixelProps {
	isSelected: boolean;
	setSelected: (isSelected: boolean) => void;
}

export const Pixel: FC<PixelProps> = memo(({ isSelected, setSelected }) => {
	const mouseHandler: MouseEventHandler = e => {
		e.preventDefault();
		if (e.buttons === 1) {
			setSelected(true);
		}
		if (e.buttons === 2) {
			setSelected(false);
		}
	};
	return (
		<div
			className={cn(styles.pixel, isSelected && styles.selected)}
			onMouseDown={mouseHandler}
			onMouseEnter={mouseHandler}
			onContextMenu={e => e.preventDefault()}
		/>
	);
});

Pixel.displayName = "Pixel";
