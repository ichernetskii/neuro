import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./components/App/App.tsx";
import { Context } from "./store/Context.tsx";
import { Store } from "./store/store.ts";

createRoot(document.getElementById("root")!).render(
	<StrictMode>
		<Context.Provider value={new Store(28, 28)}>
			<App />
		</Context.Provider>
	</StrictMode>,
);
