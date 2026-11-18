"""The results of the stage pages as files, so a saved session brings them back: one .npz holding a text header (JSON)
and arrays of numbers. Nothing is pickled, so a session folder from someone else cannot run code when it is opened."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

HEADER = "partaker_stage_file"
VERSION = 1


def plain(value):
    """numpy numbers as the plain numbers JSON knows."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} cannot be written to a stage file")


def write(path, kind: str, header: dict, arrays: Dict[str, np.ndarray]) -> str:
    text = json.dumps({"kind": kind, "version": VERSION, **header}, default=plain)
    np.savez_compressed(str(path), **{HEADER: np.array(text)}, **arrays)
    return str(path)


def read(path, kind: str) -> Tuple[dict, Dict[str, np.ndarray]]:
    """(header, arrays) of a file written by write(); ValueError when the file is not one of ours or of another kind."""
    name = Path(path).name
    try:
        with np.load(str(path), allow_pickle=False) as f:
            if HEADER not in f.files:
                raise ValueError(f"{name} is not a file of this tool")
            header = json.loads(str(f[HEADER])); arrays = {k: f[k] for k in f.files if k != HEADER}
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} could not be read ({type(exc).__name__})") from exc
    if header.get("kind") != kind or int(header.get("version", 0)) > VERSION:
        raise ValueError(f"{name} is not a {kind} file this version of the tool can read")
    return header, arrays


def int_keys(d: dict) -> dict:
    return {int(k): v for k, v in d.items()}
