"""
compare.py
----------
Trains ID3, C4.5 and CART (our from-scratch implementations) plus
scikit-learn's DecisionTreeClassifier as a reference baseline, on several
datasets, and produces:
  - results/metrics.csv           (raw numbers)
  - results/comparison_<ds>.png   (per-dataset bar charts)
  - results/summary.png           (overall accuracy heatmap)
  - printed tables in the console

Datasets used
-------------
1. play_tennis   - classic fully-categorical toy dataset (ID3's home turf)
2. iris          - continuous features, 3 classes
3. breast_cancer - continuous features, binary, higher dimensional
4. car_evaluation- categorical, UCI-style, generated locally (no internet
                   dependency) to also test multi-way categorical splits
                   at a larger scale

Run:
    python compare.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent))
from src.id3 import ID3Classifier
from src.c45 import C45Classifier
from src.cart import CARTClassifier
from src.utils import evaluate, timed_fit

warnings.filterwarnings("ignore")
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------- #
# Dataset loaders
# --------------------------------------------------------------------- #
def load_play_tennis():
    """The canonical fully-categorical ID3 textbook dataset."""
    data = {
        "outlook":     ["sunny", "sunny", "overcast", "rain", "rain", "rain", "overcast",
                         "sunny", "sunny", "rain", "sunny", "overcast", "overcast", "rain"],
        "temperature": ["hot", "hot", "hot", "mild", "cool", "cool", "cool",
                         "mild", "cool", "mild", "mild", "mild", "hot", "mild"],
        "humidity":    ["high", "high", "high", "high", "normal", "normal", "normal",
                         "high", "normal", "normal", "normal", "high", "normal", "high"],
        "wind":        ["weak", "strong", "weak", "weak", "weak", "strong", "strong",
                         "weak", "weak", "weak", "strong", "strong", "weak", "strong"],
        "play":        ["no", "no", "yes", "yes", "yes", "no", "yes",
                         "no", "yes", "yes", "yes", "yes", "yes", "no"],
    }
    df = pd.DataFrame(data)
    X, y = df.drop(columns=["play"]), df["play"]
    return X, y, []  # no continuous columns


def load_car_evaluation(n=1200, seed=42):
    """
    Locally generated dataset in the style of the UCI Car Evaluation
    dataset (fully categorical, multi-way attributes, 4 classes), so the
    comparison has a larger categorical benchmark without a network
    dependency.
    """
    rng = np.random.RandomState(seed)
    buying = rng.choice(["vhigh", "high", "med", "low"], n)
    maint = rng.choice(["vhigh", "high", "med", "low"], n)
    doors = rng.choice(["2", "3", "4", "5more"], n)
    persons = rng.choice(["2", "4", "more"], n)
    lug_boot = rng.choice(["small", "med", "big"], n)
    safety = rng.choice(["low", "med", "high"], n)

    score_map = {"vhigh": 0, "high": 1, "med": 2, "low": 3}
    persons_map = {"2": 0, "4": 1, "more": 2}
    safety_map = {"low": 0, "med": 1, "high": 2}
    lug_map = {"small": 0, "med": 1, "big": 2}

    quality = (np.vectorize(score_map.get)(buying) * -1
               + np.vectorize(score_map.get)(maint) * -1
               + np.vectorize(persons_map.get)(persons) * 2
               + np.vectorize(safety_map.get)(safety) * 2
               + np.vectorize(lug_map.get)(lug_boot))

    bins = np.quantile(quality, [0.25, 0.5, 0.75])
    labels = np.digitize(quality, bins)
    class_names = np.array(["unacc", "acc", "good", "vgood"])
    y = pd.Series(class_names[labels], name="class")

    X = pd.DataFrame({
        "buying": buying, "maint": maint, "doors": doors,
        "persons": persons, "lug_boot": lug_boot, "safety": safety,
    })
    return X, y, []


def load_iris_df():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target_names[data.target], name="species")
    return X, y, list(X.columns)


def load_breast_cancer_df():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target_names[data.target], name="diagnosis")
    return X, y, list(X.columns)


DATASETS = {
    "play_tennis": load_play_tennis,
    "car_evaluation": load_car_evaluation,
    "iris": load_iris_df,
    "breast_cancer": load_breast_cancer_df,
}


# --------------------------------------------------------------------- #
# ID3 requires categorical features -> bucket continuous ones for it only
# --------------------------------------------------------------------- #
def discretize_for_id3(X: pd.DataFrame, continuous_cols: list, bins: int = 4) -> pd.DataFrame:
    X = X.copy()
    for c in continuous_cols:
        X[c] = pd.cut(X[c], bins=bins, labels=[f"b{i}" for i in range(bins)])
    return X


# --------------------------------------------------------------------- #
def run_dataset(name: str, loader, test_size=0.3, seed=42) -> list:
    print(f"\n{'=' * 70}\nDataset: {name}\n{'=' * 70}")
    X, y, continuous_cols = loader()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    results = []

    # ---------------- ID3 (categorical only -> discretize if needed) ----
    X_train_id3 = discretize_for_id3(X_train, continuous_cols)
    X_test_id3 = discretize_for_id3(X_test, continuous_cols)
    model, fit_time = timed_fit(lambda: ID3Classifier(max_depth=8).fit(X_train_id3, y_train))
    results.append(evaluate("ID3", name, model, X_train_id3, y_train, X_test_id3, y_test,
                             model.get_depth(), model.n_nodes_, model.n_leaves_, fit_time))

    # ---------------- C4.5 (native continuous + gain ratio + pruning) ---
    model, fit_time = timed_fit(
        lambda: C45Classifier(max_depth=8, prune=True, cf=0.25).fit(
            X_train, y_train, continuous_cols=continuous_cols))
    results.append(evaluate("C4.5", name, model, X_train, y_train, X_test, y_test,
                             model.get_depth(), model.n_nodes_, model.n_leaves_, fit_time))

    # ---------------- CART (Gini, binary splits, CCP) --------------------
    model, fit_time = timed_fit(
        lambda: CARTClassifier(max_depth=8, ccp_alpha=0.0).fit(
            X_train, y_train, continuous_cols=continuous_cols))
    results.append(evaluate("CART", name, model, X_train, y_train, X_test, y_test,
                             model.get_depth(), model.n_nodes_, model.n_leaves_, fit_time))

    # ---------------- sklearn reference (CART-based, gini) ---------------
    X_train_enc = pd.get_dummies(X_train)
    X_test_enc = pd.get_dummies(X_test).reindex(columns=X_train_enc.columns, fill_value=0)
    model, fit_time = timed_fit(
        lambda: DecisionTreeClassifier(max_depth=8, criterion="gini", random_state=seed).fit(
            X_train_enc, y_train))
    results.append(evaluate("sklearn-CART", name, model, X_train_enc, y_train, X_test_enc, y_test,
                             model.get_depth(), model.tree_.node_count,
                             model.get_n_leaves(), fit_time))

    df = pd.DataFrame([r.as_dict() for r in results])
    print(df[["model", "accuracy", "precision", "recall", "f1", "tree_depth", "n_nodes"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return results


def plot_dataset(df: pd.DataFrame, dataset_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    sub = df[df.dataset == dataset_name]
    sns.barplot(data=sub, x="model", y="accuracy", ax=axes[0], palette="viridis")
    axes[0].set_title(f"Accuracy — {dataset_name}")
    axes[0].set_ylim(0, 1.05)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                          ha="center", va="bottom", fontsize=9)

    sns.barplot(data=sub, x="model", y="n_nodes", ax=axes[1], palette="magma")
    axes[1].set_title(f"Tree size (#nodes) — {dataset_name}")

    plt.tight_layout()
    out = RESULTS_DIR / f"comparison_{dataset_name}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved {out}")


def plot_summary(df: pd.DataFrame):
    pivot = df.pivot_table(index="model", columns="dataset", values="accuracy")
    plt.figure(figsize=(9, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1)
    plt.title("Accuracy across algorithms and datasets")
    plt.tight_layout()
    out = RESULTS_DIR / "summary.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved {out}")


def main():
    all_results = []
    for name, loader in DATASETS.items():
        all_results.extend(run_dataset(name, loader))

    df = pd.DataFrame([r.as_dict() for r in all_results])
    csv_path = RESULTS_DIR / "metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw metrics to {csv_path}")

    for name in DATASETS:
        plot_dataset(df, name)
    plot_summary(df)

    print(f"\n{'=' * 70}\nOverall mean accuracy by model\n{'=' * 70}")
    print(df.groupby("model")["accuracy"].mean().sort_values(ascending=False)
          .to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
