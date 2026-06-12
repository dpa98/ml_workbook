import numpy as np
import pandas as pd

from .config import Config


def load_iris(cfg: Config) -> pd.DataFrame:
    """Load the Iris dataset from a URL or a local path.

    Raises
    ------
    ValueError
        If the dataset cannot be downloaded or read.
    """
    source = cfg.local_path or cfg.data_url
    try:
        return pd.read_csv(source)
    except Exception as exc:
        raise ValueError(f"Could not load Iris dataset from {source!r}: {exc}") from exc


def prepare_dataset(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Load the first two Iris classes and return (X, y) as numpy arrays.

    y values follow ``cfg.class_map``; only samples whose species is in the
    mapping are kept, so the class order of the original dataset is preserved.
    """
    df = load_iris(cfg)
    df = df[df["species"].isin(cfg.class_map)]
    y = df["species"].map(cfg.class_map).to_numpy(dtype=np.int64)
    X = df[list(cfg.feature_cols)].to_numpy(dtype=np.float64)
    return X, y


def standardize(X: np.ndarray) -> np.ndarray:
    """Standardize features to zero mean and unit variance."""
    X_std = np.empty_like(X)
    for i in range(X.shape[1]):
        X_std[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()
    return X_std
