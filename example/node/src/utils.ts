import { readdir } from "fs/promises";
import { type PathLike } from "fs";

export const getDirectories = async (source: PathLike) =>
	// prettier-ignore
	(await readdir(source, { withFileTypes: true }))
        .filter(dirent => dirent.isDirectory())
        .map(dirent => dirent.name);

export const getFiles = async (source: PathLike) =>
	// prettier-ignore
	(await readdir(source, { withFileTypes: true }))
        .filter(dirent => dirent.isFile() && dirent.name.endsWith(".png"))
        .map(dirent => dirent.name);
