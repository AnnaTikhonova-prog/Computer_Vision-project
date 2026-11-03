"""Generate the report figures from committed evaluation and training CSVs."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"


def metric(condition: str, augmentation: str, name: str) -> tuple[float, float]:
    path = TABLES / f"{condition}_{augmentation}_test_summary.csv"
    data = pd.read_csv(path).set_index("metric")
    return float(data.loc[name, "mean"]), float(data.loc[name, "std"])


def plot_test_comparison() -> None:
    names = [
        "C0 · aug on", "C1 · aug on", "C2 · aug on", "C3 · aug on",
        "C0 · aug off", "C3 · aug off",
    ]
    keys = [
        ("C0", "augment_on"), ("C1", "augment_on"),
        ("C2", "augment_on"), ("C3", "augment_on"),
        ("C0", "augment_off"), ("C3", "augment_off"),
    ]
    accuracy = [metric(*key, "accuracy") for key in keys]
    macro_f1 = [metric(*key, "macro_f1") for key in keys]
    x = list(range(len(names)))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    for ax, values, title, marker in [
        (axes[0], accuracy, "Accuracy", "o"),
        (axes[1], macro_f1, "Macro-F1", "s"),
    ]:
        ax.errorbar(x, [v[0] for v in values], yerr=[v[1] for v in values],
                    marker=marker, capsize=4, linewidth=1.8, color="#4C78A8")
        ax.set_ylim(0.92, 0.99)
        ax.set_ylabel(f"{title}\n(mean ± sample SD)")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title("Final test comparison · 3 checkpoints per configuration")
    axes[1].set_xticks(x, names, rotation=12, ha="right")
    fig.savefig(FIGURES / "final_test_comparison.png", dpi=180)
    plt.close(fig)


def plot_training_curves() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    colors = {"C0": "#4C78A8", "C1": "#F58518", "C2": "#54A24B", "C3": "#E45756"}
    for condition, lr in [("C0", "1e-3"), ("C1", "1e-4"),
                          ("C2", "1e-5"), ("C3", "1e-5")]:
        per_seed = []
        for seed in (42, 43, 44):
            history = pd.read_csv(
                ROOT / "deliverables" / condition.lower() / "runs"
                / f"lr_{lr}" / f"seed_{seed}" / "history.csv"
            ).set_index("epoch")["val_macro_f1"]
            per_seed.append(history.rename(seed))
        aligned = pd.concat(per_seed, axis=1)
        mean = aligned.mean(axis=1)
        std = aligned.std(axis=1, ddof=1)
        color = colors[condition]
        ax.plot(mean.index, mean, label=condition, color=color, linewidth=1.8)
        ax.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Validation learning curves · selected LR, mean ± SD across seeds")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(FIGURES / "selected_training_curves.png", dpi=180)
    plt.close(fig)


def plot_ablation_validation() -> None:
    # Values are the committed three-seed aggregates in reports/ablation.md.
    means = {"C0": (0.9700663, 0.9597596), "C3": (0.9766957, 0.9677224)}
    stds = {"C0": (0.0034078, 0.0057848), "C3": (0.0066162, 0.0052049)}
    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    x = [0, 1]
    for condition, color in [("C0", "#4C78A8"), ("C3", "#E45756")]:
        values, errors = means[condition], stds[condition]
        ax.errorbar(x, values, yerr=errors, marker="o", capsize=4,
                    linewidth=1.8, color=color, label=condition)
    ax.set_xticks(x, ["Augmentation on", "Augmentation off"])
    ax.set_ylabel("Validation macro-F1 (mean ± sample SD)")
    ax.set_ylim(0.94, 0.99)
    ax.set_title("Validation augmentation ablation · 3 seeds")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(FIGURES / "validation_augmentation_ablation.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_test_comparison()
    plot_training_curves()
    plot_ablation_validation()
