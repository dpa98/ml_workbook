"""Train AdalineSGD on the Iris dataset and visualize the results.

Usage:
    python scripts/train.py [--save DIR]
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
    plot_decision_regions,
    plot_loss_curve,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        metavar="DIR",
        help="save the figures as PNGs into DIR instead of showing them",
    )
    args = parser.parse_args()

    cfg = Config()
    print("Loading Iris dataset...")
    X, y = prepare_dataset(cfg)

    X_std = standardize(X)
    print("Training AdalineSGD...")
    ada_sgd = AdalineSGD(eta=cfg.eta, n_iter=cfg.n_iter, random_state=cfg.random_state)
    ada_sgd.fit(X_std, y)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    plot_decision_regions(X_std, y, classifier=ada_sgd, ax=axes[0])
    axes[0].set_title("Adaline - Stochastic Gradient Descent")
    axes[0].set_xlabel("Sepal length [standardized]")
    axes[0].set_ylabel("Petal length [standardized]")
    axes[0].legend(loc="upper left")

    plot_loss_curve(ada_sgd.losses_, ax=axes[1])
    fig.tight_layout()

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        fig.savefig(os.path.join(args.save, "training.png"), dpi=150)
        print(f"Figure saved to {args.save}/training.png")
    else:
        plt.show()


if __name__ == "__main__":
    main()
