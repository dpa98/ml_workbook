"""AdalineSGD: ADAptive LInear NEuron trained with stochastic gradient descent."""

from .config import Config
from .data import load_iris, prepare_dataset, standardize
from .models import AdalineSGD
from .plotting import plot_decision_regions, plot_loss_curve, plot_weight_evolution

__all__ = [
    "AdalineSGD",
    "Config",
    "load_iris",
    "prepare_dataset",
    "standardize",
    "plot_decision_regions",
    "plot_loss_curve",
    "plot_weight_evolution",
]

__version__ = "0.1.0"
