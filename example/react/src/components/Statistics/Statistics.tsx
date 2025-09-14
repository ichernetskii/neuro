import type { FC } from "react";
import { round } from "neuro-lib/src/functions/round.ts";
import styles from "./Statistics.module.scss";

export type StatisticsData = Record<string, number>;

interface StatisticsProps {
	statisticsData: StatisticsData;
}

export const Statistics: FC<StatisticsProps> = ({ statisticsData }) => {
	const sum = Object.values(statisticsData).reduce((acc, value) => acc + value, 0);
	const dataLength = Object.keys(statisticsData).length;
	return (
		<div
			className={styles.statistics}
			style={{
				gridTemplateColumns: `repeat(${dataLength}, 1fr)`,
				gridTemplateRows: `1fr auto`,
			}}
		>
			{Object.entries(statisticsData).map(([key, value]) => (
				<div className={styles.barWrapper}>
					<div key={key} style={{ height: `${(100 * value) / sum}%` }} className={styles.bar} />
					<div className={styles.valueText}>{`${round(100 * value, 1)}%`}</div>
				</div>
			))}
			{Object.keys(statisticsData).map(key => (
				<div className={styles.title} key={key}>
					{key}
				</div>
			))}
		</div>
	);
};
