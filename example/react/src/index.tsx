import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./components/App/App.tsx";
import { Context } from "./store/Context.tsx";
import { Store } from "./store/store.ts";

export const IMAGE_WIDTH = 28;
export const IMAGE_HEIGHT = 28;

createRoot(document.getElementById("root")!).render(
	<StrictMode>
		<Context.Provider value={new Store(IMAGE_WIDTH, IMAGE_HEIGHT)}>
			<App />
		</Context.Provider>
	</StrictMode>,
);
