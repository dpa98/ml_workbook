"""Show how the weights and bias evolve during AdalineGD training.

Usage:
    python scripts/weights.py [--save DIR]
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaline import Config, prepare_dataset, standardize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        metavar="DIR",
        help="save the figures as PNGs into DIR instead of showing them",
    )
    args = parser.parse_args()

    cfg = Config()
    X, y = prepare_dataset(cfg)
    X_std = standardize(X)

    rgen = np.random.RandomState(cfg.random_state)
    w = rgen.normal(loc=0.0, scale=0.01, size=X_std.shape[1])
    b = 0.0
    epochs = list(range(1, cfg.n_iter + 1))
    w1, w2, bias = [], [], []

    for _ in range(cfg.n_iter):
        output = X_std.dot(w) + b
        errors = y - output
        w += cfg.eta * 2.0 * X_std.T.dot(errors) / X_std.shape[0]
        b += cfg.eta * 2.0 * errors.mean()
        w1.append(w[0])
        w2.append(w[1])
        bias.append(b)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(epochs, w1, marker="o", color="#fb4934", label="w1 (sepal length)")
    ax.plot(epochs, w2, marker="s", color="#83a598", label="w2 (petal length)")
    ax.plot(epochs, bias, marker="^", color="#504945", label="b (bias)")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Weight value")
    ax.set_title("Weight evolution during training")
    ax.legend(loc="upper right")

    fig.tight_layout()

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        fig.savefig(os.path.join(args.save, "weights.png"), dpi=150)
        print(f"Figures saved to {args.save}/")
    else:
        plt.show()


if __name__ == "__main__":
    main()
