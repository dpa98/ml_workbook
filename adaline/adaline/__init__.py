"""Adaline: ADAptive LInear NEuron trained with gradient descent."""

from .config import Config
from .data import load_iris, prepare_dataset, standardize
from .models import AdalineGD
from .plotting import plot_decision_regions, plot_loss_curve

__all__ = [
    "AdalineGD",
    "Config",
    "load_iris",
    "prepare_dataset",
    "standardize",
    "plot_decision_regions",
    "plot_loss_curve",
]

__version__ = "0.1.0"