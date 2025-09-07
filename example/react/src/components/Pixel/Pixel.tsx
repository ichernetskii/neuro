import { type FC, memo } from "react";
import styles from "./Pixel.module.scss";
import cn from "classnames";

interface PixelProps {
	isSelected: boolean;
}

export const Pixel: FC<PixelProps> = memo(({ isSelected }) => (
	<div className={cn(styles.pixel, isSelected && styles.selected)} />
));

Pixel.displayName = "Pixel";
