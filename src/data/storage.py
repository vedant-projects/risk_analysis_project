"""storage.py - Parquet cache IO for raw series and processed artifacts."""
from __future__ import annotations

import json
import pathlib

import pandas as pd

RAW_DIR = pathlib.Path("data/raw")
PROC_DIR = pathlib.Path("data/processed")


def cache_path(source: str, name: str, cache_dir=RAW_DIR) -> pathlib.Path:
    return pathlib.Path(cache_dir) / f"{source}_{name}.parquet"


def read_cache(source: str, name: str, cache_dir=RAW_DIR) -> pd.Series | None:
    p = cache_path(source, name, cache_dir)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    return df.iloc[:, 0].sort_index()


def write_cache(s: pd.Series, source: str, name: str, cache_dir=RAW_DIR) -> pathlib.Path:
    p = cache_path(source, name, cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame(name=s.name or name).to_parquet(p)
    return p


def merge_incremental(old: pd.Series | None, new: pd.Series) -> pd.Series:
    """Concatenate cache + fresh pull; on overlap the fresh pull wins."""
    out = new if old is None or old.empty else pd.concat([old, new])
    return out[~out.index.duplicated(keep="last")].sort_index()


def write_manifest(info: dict, path=RAW_DIR / "_manifest.json") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(path.read_text()) if path.exists() else {}
    prev.update(info)
    path.write_text(json.dumps(prev, indent=2, default=str))