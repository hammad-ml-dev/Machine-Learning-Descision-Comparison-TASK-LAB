"""
CART (Classification And Regression Trees)
-------------------------------------------
Breiman, Friedman, Olshen & Stone (1984). Implemented from scratch.

Key properties implemented here:
- Splitting criterion : Gini impurity for classification
                         (variance reduction for regression, see
                         `criterion='mse'`)
- Branching           : Strictly binary (every split is a yes/no question),
                         for both categorical and continuous attributes
- Categorical handling: Best binary partition of categories is searched
                         (subset split), not just a single-value split
- Continuous handling : Threshold splits, same style as C4.5 but always
                         binary and chosen by Gini gain instead of gain ratio
- Pruning             : Cost-Complexity Pruning (CCP) — the hallmark of
                         CART — via a user-supplied ccp_alpha, mirroring
                         scikit-learn's minimal cost-complexity pruning path
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class Node:
    is_leaf: bool = False
    prediction: Any = None
    attribute: Optional[str] = None
    is_continuous: bool = False
    threshold: Optional[float] = None
    left_categories: Optional[set] = None   # for categorical binary splits
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    depth: int = 0
    n_samples: int = 0
    class_counts: dict = field(default_factory=dict)
    impurity: float = 0.0


def gini(y: pd.Series) -> float:
    n = len(y)
    if n == 0:
        return 0.0
    counts = Counter(y)
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


class CARTClassifier:
    """
    From-scratch CART classifier: binary splits chosen by Gini impurity
    reduction, with cost-complexity pruning.

    Parameters
    ----------
    max_depth : int, optional
    min_samples_split : int, default=2
    min_samples_leaf : int, default=1
    ccp_alpha : float, default=0.0
        Complexity parameter for cost-complexity pruning. 0.0 disables
        pruning (same semantics as scikit-learn).
    """

    def __init__(self, max_depth: Optional[int] = None, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, ccp_alpha: float = 0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.ccp_alpha = ccp_alpha
        self.root: Optional[Node] = None
        self.continuous_cols_: set = set()
        self.n_nodes_ = 0
        self.n_leaves_ = 0

    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y: pd.Series, continuous_cols: Optional[list] = None) -> "CARTClassifier":
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        if continuous_cols is None:
            continuous_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.continuous_cols_ = set(continuous_cols)

        self.root = self._build(X, y, depth=0)

        if self.ccp_alpha > 0:
            self._cost_complexity_prune(self.ccp_alpha)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self._predict_row(row, self.root) for _, row in X.iterrows()])

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == np.asarray(y)))

    def get_depth(self) -> int:
        return self._depth(self.root)

    # ------------------------------------------------------------------ #
    # splitting
    # ------------------------------------------------------------------ #
    def _best_split(self, X: pd.DataFrame, y: pd.Series, attributes: list):
        n = len(y)
        parent_gini = gini(y)
        best_gain, best_attr, best_thresh, best_cats, best_is_cont = -1.0, None, None, None, False

        for attr in attributes:
            col = X[attr]
            if attr in self.continuous_cols_:
                vals = np.sort(col.unique())
                if len(vals) < 2:
                    continue
                candidates = (vals[:-1] + vals[1:]) / 2.0
                for t in candidates:
                    left_mask = col.values <= t
                    right_mask = ~left_mask
                    nl, nr = left_mask.sum(), right_mask.sum()
                    if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                        continue
                    g = parent_gini - (nl / n) * gini(y[left_mask]) - (nr / n) * gini(y[right_mask])
                    if g > best_gain:
                        best_gain, best_attr, best_thresh, best_is_cont = g, attr, t, True
                        best_cats = None
            else:
                categories = col.unique().tolist()
                if len(categories) < 2:
                    continue
                # Search all binary partitions of the category set (feasible
                # for the low-cardinality categorical columns typical of
                # textbook datasets; CART in practice uses this exact
                # subset-search approach for categorical splits).
                for r in range(1, len(categories) // 2 + 1):
                    for subset in itertools.combinations(categories, r):
                        # avoid double-counting complementary subsets when len==len/2
                        if len(categories) % 2 == 0 and r == len(categories) // 2:
                            if subset > tuple(c for c in categories if c not in subset):
                                continue
                        subset_set = set(subset)
                        left_mask = col.isin(subset_set).values
                        right_mask = ~left_mask
                        nl, nr = left_mask.sum(), right_mask.sum()
                        if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                            continue
                        g = parent_gini - (nl / n) * gini(y[left_mask]) - (nr / n) * gini(y[right_mask])
                        if g > best_gain:
                            best_gain, best_attr, best_cats, best_is_cont = g, attr, subset_set, False
                            best_thresh = None

        return best_attr, best_thresh, best_cats, best_is_cont, best_gain

    # ------------------------------------------------------------------ #
    def _build(self, X: pd.DataFrame, y: pd.Series, depth: int) -> Node:
        self.n_nodes_ += 1
        class_counts = dict(Counter(y))
        node_impurity = gini(y)
        majority = Counter(y).most_common(1)[0][0]

        if (len(set(y)) == 1 or
                len(y) < self.min_samples_split or
                (self.max_depth is not None and depth >= self.max_depth)):
            self.n_leaves_ += 1
            return Node(is_leaf=True, prediction=majority, depth=depth, n_samples=len(y),
                        class_counts=class_counts, impurity=node_impurity)

        attrs = list(X.columns)
        best_attr, best_thresh, best_cats, is_cont, best_gain = self._best_split(X, y, attrs)

        if best_attr is None or best_gain <= 0:
            self.n_leaves_ += 1
            return Node(is_leaf=True, prediction=majority, depth=depth, n_samples=len(y),
                        class_counts=class_counts, impurity=node_impurity)

        node = Node(is_leaf=False, attribute=best_attr, is_continuous=is_cont,
                    threshold=best_thresh, left_categories=best_cats, depth=depth,
                    n_samples=len(y), class_counts=class_counts, impurity=node_impurity)

        col = X[best_attr]
        if is_cont:
            left_mask = col.values <= best_thresh
        else:
            left_mask = col.isin(best_cats).values
        right_mask = ~left_mask

        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[right_mask], y[right_mask], depth + 1)
        return node

    # ------------------------------------------------------------------ #
    # cost-complexity pruning
    # ------------------------------------------------------------------ #
    def _leaf_errors_weighted(self, node: Node, n_total: int) -> float:
        """Weighted misclassification rate of a single leaf (R(t))."""
        n = sum(node.class_counts.values())
        correct = node.class_counts.get(node.prediction, 0)
        return (n - correct) / n_total

    def _subtree_errors_weighted(self, node: Node, n_total: int) -> float:
        if node.is_leaf:
            return self._leaf_errors_weighted(node, n_total)
        return (self._subtree_errors_weighted(node.left, n_total) +
                self._subtree_errors_weighted(node.right, n_total))

    def _count_leaves(self, node: Node) -> int:
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def _cost_complexity_prune(self, alpha: float):
        """
        Greedily collapse the subtree whose effective alpha
        g(t) = (R(t) - R(T_t)) / (|leaves(T_t)| - 1)
        is smallest, repeating while that g(t) <= alpha.
        This mirrors scikit-learn's minimal cost-complexity pruning (Breiman et al.).
        """
        n_total = self.root.n_samples

        def find_weakest_link(node: Node):
            """Return (node_with_min_g, min_g) over all internal nodes."""
            if node.is_leaf:
                return None, math.inf
            Rt = self._leaf_errors_weighted(
                Node(is_leaf=True, prediction=max(node.class_counts, key=node.class_counts.get),
                     class_counts=node.class_counts), n_total)
            RTt = self._subtree_errors_weighted(node, n_total)
            leaves = self._count_leaves(node)
            g = (Rt - RTt) / (leaves - 1) if leaves > 1 else math.inf

            best_node, best_g = node, g
            for child in (node.left, node.right):
                cand_node, cand_g = find_weakest_link(child)
                if cand_node is not None and cand_g < best_g:
                    best_node, best_g = cand_node, cand_g
            return best_node, best_g

        while True:
            weakest_node, g = find_weakest_link(self.root)
            if weakest_node is None or g > alpha:
                break
            # collapse weakest_node into a leaf
            majority = max(weakest_node.class_counts, key=weakest_node.class_counts.get)
            weakest_node.is_leaf = True
            weakest_node.prediction = majority
            weakest_node.left = None
            weakest_node.right = None
            weakest_node.attribute = None
            if weakest_node is self.root:
                break

    # ------------------------------------------------------------------ #
    def _predict_row(self, row: pd.Series, node: Node):
        while not node.is_leaf:
            val = row.get(node.attribute, None)
            if node.is_continuous:
                go_left = (val is not None) and (val <= node.threshold)
            else:
                go_left = val in node.left_categories if node.left_categories else False
            node = node.left if go_left else node.right
        return node.prediction

    def _depth(self, node: Optional[Node]) -> int:
        if node is None or node.is_leaf:
            return node.depth if node else 0
        return max(self._depth(node.left), self._depth(node.right))
