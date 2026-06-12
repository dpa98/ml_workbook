from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    """Centralized configuration for the training pipeline."""

    data_url: str = (
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    )
    local_path: str | None = None

    target_col: str = "species"
    feature_cols: tuple[str, ...] = ("sepal_length", "petal_length")
    class_map: dict[str, int] = field(default_factory=lambda: {"setosa": 0, "versicolor": 1})

    eta: float = 0.5
    n_iter: int = 50
    random_state: int = 1
