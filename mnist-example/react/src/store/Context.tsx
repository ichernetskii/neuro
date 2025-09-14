import { createContext, useContext } from "react";
import type { Store } from "./store.ts";

export const Context = createContext<Store>(null!);

export const useStore = () => {
	return useContext(Context);
};
