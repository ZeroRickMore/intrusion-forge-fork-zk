import pickle
from pathlib import Path

import pandas as pd

_LOADERS = {
    ".parquet": pd.read_parquet,
    ".csv": pd.read_csv,
}

_SAVERS = {
    ".parquet": lambda df, p, **kw: df.to_parquet(p, **kw),
    ".csv": lambda df, p, **kw: df.to_csv(p, **kw),
}

_PICKLE_EXTS = {".pkl", ".pickle"}
_ALL_EXTS = sorted({*_LOADERS, *_PICKLE_EXTS})


def load_df(file_path: str | Path, **kwargs) -> pd.DataFrame:
    """Load a DataFrame from a file based on its extension."""
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext in _PICKLE_EXTS:
        with open(file_path, "rb") as f:
            return pickle.load(f)

    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file extension: {ext!r}. Supported: {_ALL_EXTS}")
    return loader(file_path, **kwargs)


def save_df(
    df: pd.DataFrame,
    file_path: str | Path,
    *,
    index: bool = False,
    **kwargs,
) -> None:
    """Save a DataFrame to a file based on its extension."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    ext = file_path.suffix.lower()

    if ext in _PICKLE_EXTS:
        with open(file_path, "wb") as f:
            pickle.dump(df, f)
        return

    saver = _SAVERS.get(ext)
    if saver is None:
        raise ValueError(f"Unsupported file extension: {ext!r}. Supported: {_ALL_EXTS}")
    saver(df, file_path, index=index, **kwargs)


def load_listed_dfs(base_dir: str | Path, file_names: list[str]) -> list[pd.DataFrame]:
    """Load each `file_names` entry from `base_dir`, preserving order."""
    base_dir = Path(base_dir)
    return [load_df(base_dir / name) for name in file_names]
