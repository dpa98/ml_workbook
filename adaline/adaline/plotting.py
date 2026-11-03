import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, ax=None, resolution=0.02):
    """Plot the decision surface and the class examples."""
    ax = ax or plt.gca()

    markers = ("o", "s", "^", "v", "<")
    colors = ("#fb4934", "#83a598", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)

    ax.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                   alpha=0.8, c=colors[idx],
                   marker=markers[idx], label=f"Class {cl}",
                   edgecolor="black")


def plot_loss_curve(losses, ax=None):
    """Plot the mean squared error loss per epoch."""
    ax = ax or plt.gca()
    ax.plot(range(1, len(losses) + 1), losses, marker="o", color="#504945")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Mean squared error loss")
