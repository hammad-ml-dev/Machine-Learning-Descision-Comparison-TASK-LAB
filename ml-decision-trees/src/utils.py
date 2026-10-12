"""Shared helpers for loading datasets and reporting metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


@dataclass
class RunResult:
    model: str
    dataset: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    train_time_sec: float
    predict_time_sec: float
    tree_depth: int
    n_nodes: int
    n_leaves: int

    def as_dict(self):
        return asdict(self)


def evaluate(model_name: str, dataset_name: str, model, X_train, y_train, X_test, y_test,
             depth: int, n_nodes: int, n_leaves: int, fit_time: float) -> RunResult:
    t0 = time.perf_counter()
    preds = model.predict(X_test)
    predict_time = time.perf_counter() - t0

    # Macro averaging works uniformly for binary and multi-class string
    # labels alike, avoiding sklearn's pos_label=1 assumption for "binary".
    return RunResult(
        model=model_name,
        dataset=dataset_name,
        accuracy=accuracy_score(y_test, preds),
        precision=precision_score(y_test, preds, average="macro", zero_division=0),
        recall=recall_score(y_test, preds, average="macro", zero_division=0),
        f1=f1_score(y_test, preds, average="macro", zero_division=0),
        train_time_sec=fit_time,
        predict_time_sec=predict_time,
        tree_depth=depth,
        n_nodes=n_nodes,
        n_leaves=n_leaves,
    )


def timed_fit(fit_callable: Callable):
    t0 = time.perf_counter()
    model = fit_callable()
    elapsed = time.perf_counter() - t0
    return model, elapsed
