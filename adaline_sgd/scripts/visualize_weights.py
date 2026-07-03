"""Visualize weight evolution during AdalineSGD training.

Usage:
    python scripts/visualize_weights.py [--save DIR]
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaline_sgd import (
    AdalineSGD,
    Config,
    prepare_dataset,
    standardize,
    plot_weight_evolution,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        metavar="DIR",
        help="save the figure as PNG into DIR instead of showing it",
    )
    args = parser.parse_args()

    cfg = Config()
    print("Loading Iris dataset...")
    X, y = prepare_dataset(cfg)

    X_std = standardize(X)
    print("Training AdalineSGD...")
    ada_sgd = AdalineSGD(eta=cfg.eta, n_iter=cfg.n_iter, random_state=cfg.random_state)
    ada_sgd.fit(X_std, y)

    ax_w, ax_b = plot_weight_evolution(
        ada_sgd.w_history_, cfg.n_iter, epoch_size=X_std.shape[0]
    )
    ax_w.get_figure().tight_layout()

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        ax_w.get_figure().savefig(os.path.join(args.save, "weight_evolution.png"), dpi=150)
        print(f"Figure saved to {args.save}/weight_evolution.png")
    else:
        plt.show()


if __name__ == "__main__":
    main()
