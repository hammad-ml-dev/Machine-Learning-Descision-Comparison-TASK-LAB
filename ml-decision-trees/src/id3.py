"""
ID3 (Iterative Dichotomiser 3)
------------------------------
Classic decision tree induction algorithm (Quinlan, 1986).

Key properties implemented here:
- Splitting criterion : Information Gain (based on Shannon entropy)
- Attribute handling  : Categorical only (continuous features are bucketed
                         by the caller, e.g. via pandas.cut, before fitting)
- Branching           : Multi-way (one branch per distinct attribute value)
- Missing values      : Not handled (rows are expected to be complete)
- Pruning             : None (ID3 has no built-in pruning step)
- Stopping conditions : Pure node, no attributes left, or max_depth reached

This file has zero dependency on scikit-learn's tree internals -- the
algorithm is implemented from first principles using numpy/pandas so the
mechanics of ID3 are fully visible and can be compared fairly against the
CART and C4.5 implementations in this project.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class Node:
    """A node in the ID3 decision tree."""
    is_leaf: bool = False
    prediction: Any = None                 # majority class, set on leaves
    attribute: Optional[str] = None        # splitting attribute (internal node)
    children: dict = field(default_factory=dict)   # value -> Node
    depth: int = 0
    n_samples: int = 0
    class_counts: dict = field(default_factory=dict)


def entropy(y: pd.Series) -> float:
    """Shannon entropy H(S) = -sum(p_i * log2(p_i))."""
    counts = Counter(y)
    n = len(y)
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


def information_gain(X: pd.DataFrame, y: pd.Series, attribute: str) -> float:
    """IG(S, A) = H(S) - sum_v ( |S_v|/|S| * H(S_v) )"""
    base_entropy = entropy(y)
    n = len(y)
    weighted_entropy = 0.0
    for value, idx in X.groupby(attribute).groups.items():
        subset_y = y.loc[idx]
        weighted_entropy += (len(subset_y) / n) * entropy(subset_y)
    return base_entropy - weighted_entropy


class ID3Classifier:
    """
    A from-scratch ID3 decision tree classifier with a scikit-learn-like
    `fit` / `predict` interface.

    Parameters
    ----------
    max_depth : int, optional
        Maximum depth of the tree. None means grow until pure / no
        attributes remain.
    min_samples_split : int, default=2
        Minimum number of samples required to attempt a split.
    """

    def __init__(self, max_depth: Optional[int] = None, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Optional[Node] = None
        self.n_nodes_ = 0
        self.n_leaves_ = 0

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ID3Classifier":
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        self.root = self._build(X, y, attributes=list(X.columns), depth=0)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self._predict_row(row, self.root) for _, row in X.iterrows()])

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == np.asarray(y)))

    def get_depth(self) -> int:
        return self._depth(self.root)

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _build(self, X: pd.DataFrame, y: pd.Series, attributes: list, depth: int) -> Node:
        self.n_nodes_ += 1
        class_counts = dict(Counter(y))

        # Stopping condition 1: pure node
        if len(set(y)) == 1:
            self.n_leaves_ += 1
            return Node(is_leaf=True, prediction=y.iloc[0], depth=depth,
                        n_samples=len(y), class_counts=class_counts)

        majority = Counter(y).most_common(1)[0][0]

        # Stopping condition 2: no attributes left, max depth, or too few samples
        if (not attributes or
                (self.max_depth is not None and depth >= self.max_depth) or
                len(y) < self.min_samples_split):
            self.n_leaves_ += 1
            return Node(is_leaf=True, prediction=majority, depth=depth,
                        n_samples=len(y), class_counts=class_counts)

        # Choose attribute with highest information gain
        gains = {a: information_gain(X, y, a) for a in attributes}
        best_attr = max(gains, key=gains.get)

        if gains[best_attr] <= 0:
            self.n_leaves_ += 1
            return Node(is_leaf=True, prediction=majority, depth=depth,
                        n_samples=len(y), class_counts=class_counts)

        node = Node(is_leaf=False, attribute=best_attr, depth=depth,
                    n_samples=len(y), class_counts=class_counts)
        remaining_attrs = [a for a in attributes if a != best_attr]

        for value, idx in X.groupby(best_attr).groups.items():
            sub_X = X.loc[idx].drop(columns=[best_attr])
            sub_y = y.loc[idx]
            if len(sub_y) == 0:
                self.n_leaves_ += 1
                node.children[value] = Node(is_leaf=True, prediction=majority,
                                            depth=depth + 1, n_samples=0)
            else:
                node.children[value] = self._build(sub_X, sub_y, remaining_attrs, depth + 1)

        return node

    def _predict_row(self, row: pd.Series, node: Node):
        while not node.is_leaf:
            value = row.get(node.attribute, None)
            if value in node.children:
                node = node.children[value]
            else:
                # Unseen attribute value at prediction time -> fall back to
                # the majority class recorded at this node.
                return Counter(node.class_counts).most_common(1)[0][0] if node.class_counts else None
        return node.prediction

    def _depth(self, node: Optional[Node]) -> int:
        if node is None or node.is_leaf:
            return node.depth if node else 0
        return max(self._depth(c) for c in node.children.values())
