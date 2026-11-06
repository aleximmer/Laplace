import os
from pathlib import Path

import matplotlib.pyplot as plt

EXAMPLES_DIR = Path(__file__).resolve().parents[1]

# Set `LAPLACE_EXAMPLES_ASSETS_DIR` to write the plots elsewhere, e.g. in tests.
ASSETS_DIR = Path(
    os.environ.get(
        "LAPLACE_EXAMPLES_ASSETS_DIR", EXAMPLES_DIR.parent / "docs" / "assets"
    )
)


def plot_regression(
    X_train, y_train, X_test, f_test, y_std, plot=True, file_name="regression_example"
):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(4.5, 2.8))
    ax1.set_title("MAP")
    ax1.scatter(X_train.flatten(), y_train.flatten(), alpha=0.3, color="tab:orange")
    ax1.plot(X_test, f_test, color="black", label=r"$f_{MAP}$")
    ax1.legend()

    ax2.set_title("LA")
    ax2.scatter(X_train.flatten(), y_train.flatten(), alpha=0.3, color="tab:orange")
    ax2.plot(X_test, f_test, label=r"$\mathbb{E}[f]$")
    ax2.fill_between(
        X_test,
        f_test - y_std * 2,
        f_test + y_std * 2,
        alpha=0.3,
        color="tab:blue",
        label=r"$2\sqrt{\mathbb{V}\,[y]}$",
    )
    ax2.legend()
    ax1.set_ylim([-4, 6])
    ax1.set_xlim([X_test.min(), X_test.max()])
    ax2.set_xlim([X_test.min(), X_test.max()])
    ax1.set_ylabel("$y$")
    ax1.set_xlabel("$x$")
    ax2.set_xlabel("$x$")
    plt.tight_layout()
    if plot:
        plt.show()
    else:
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(ASSETS_DIR / f"{file_name}.png")
