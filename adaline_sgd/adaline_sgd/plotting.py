import numpy as np
import matplotlib.pyplot as plt
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
    """Plot the average loss per epoch."""
    ax = ax or plt.gca()
    ax.plot(range(1, len(losses) + 1), losses, marker="o", color="#504945")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Average loss")


def plot_weight_evolution(w_history, n_iter, epoch_size, axes=None):
    """Plot the evolution of weights and bias over sample updates.

    ``w_history`` is the list of (weights, bias) tuples recorded after each
    sample update. Dashed lines mark epoch boundaries. If ``axes`` is None a
    new 2-row figure is created and returned along with the axes.
    """
    w_history = np.array([w for w, b in w_history])
    b_history = np.array([b for w, b in w_history])
    n_updates = len(w_history)

    if axes is None:
        _, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True)
    ax_w, ax_b = axes
    updates = range(1, n_updates + 1)

    ax_w.plot(updates, w_history[:, 0], label="$w_1$ (sepal length)", color="#83a598")
    ax_w.plot(updates, w_history[:, 1], label="$w_2$ (petal length)", color="#d3869b")
    ax_w.set_ylabel("Weights")
    ax_w.legend(loc="upper right")
    ax_w.grid(alpha=0.3)

    ax_b.plot(updates, b_history, label="$b$ (bias)", color="#fb4934")
    ax_b.set_ylabel("Bias")
    ax_b.set_xlabel("Sample updates")
    ax_b.legend(loc="upper right")
    ax_b.grid(alpha=0.3)

    for ax in (ax_w, ax_b):
        for epoch in range(1, n_iter):
            ax.axvline(epoch * epoch_size, color="gray", linestyle="--", alpha=0.5)
        ax.annotate("epoch 1", xy=(epoch_size / 2, 1.02), xycoords=("data", "axes fraction"),
                    ha="center", fontsize=9, color="gray")
        ax.annotate(f"epoch {n_iter}", xy=((n_iter - 0.5) * epoch_size, 1.02),
                    xycoords=("data", "axes fraction"), ha="center", fontsize=9, color="gray")

    return ax_w, ax_b
