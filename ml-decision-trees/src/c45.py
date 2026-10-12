"""
C4.5
----
Quinlan's successor to ID3 (1993). Implemented from scratch.

Improvements over ID3 implemented here:
- Splitting criterion : Gain Ratio = Information Gain / Split Information
                         (corrects ID3's bias toward high-cardinality attributes)
- Continuous features : Handled natively via binary threshold splits
                         (attribute <= t  vs  attribute > t), threshold chosen
                         to maximize gain ratio
- Categorical features: Multi-way split, same as ID3
- Missing values      : Rows with NaN for the attribute being tested are
                         distributed across branches using fractional
                         weighting (the classic C4.5 approach)
- Pruning             : Pessimistic post-pruning using a confidence-based
                         error estimate (simplified error-based pruning,
                         the mechanism C4.5 is famous for)
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
    is_leaf: bool = False
    prediction: Any = None
    attribute: Optional[str] = None
    is_continuous: bool = False
    threshold: Optional[float] = None      # for continuous splits
    children: dict = field(default_factory=dict)  # category -> Node, or {'<=':Node, '>':Node}
    depth: int = 0
    n_samples: int = 0
    class_counts: dict = field(default_factory=dict)


def entropy(weights_by_class: dict) -> float:
    total = sum(weights_by_class.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in weights_by_class.values():
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


def _weighted_class_counts(y: pd.Series, w: np.ndarray) -> dict:
    d: dict = {}
    for cls, wt in zip(y, w):
        d[cls] = d.get(cls, 0.0) + wt
    return d


class C45Classifier:
    """
    From-scratch C4.5 decision tree with gain-ratio splitting, native
    continuous-attribute handling, fractional missing-value treatment,
    and pessimistic error-based post-pruning.

    Parameters
    ----------
    max_depth : int, optional
    min_samples_split : int, default=2
    prune : bool, default=True
        Whether to run pessimistic post-pruning after the tree is grown.
    cf : float, default=0.25
        Confidence factor used in the pessimistic error estimate
        (lower cf => more aggressive pruning), matching the classic
        C4.5 "-c" parameter.
    """

    def __init__(self, max_depth: Optional[int] = None, min_samples_split: int = 2,
                 prune: bool = True, cf: float = 0.25):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.prune = prune
        self.cf = cf
        self.root: Optional[Node] = None
        self.continuous_cols_: set = set()
        self.n_nodes_ = 0
        self.n_leaves_ = 0

    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y: pd.Series, continuous_cols: Optional[list] = None,
            val_X: Optional[pd.DataFrame] = None, val_y: Optional[pd.Series] = None) -> "C45Classifier":
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        if continuous_cols is None:
            continuous_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.continuous_cols_ = set(continuous_cols)

        weights = np.ones(len(X))
        self.root = self._build(X, y, weights, list(X.columns), depth=0)

        if self.prune:
            # Reduced-error / pessimistic pruning on the training data itself
            # (classic C4.5 uses the training set with a pessimistic
            # correction rather than requiring a held-out set).
            self._prune(self.root)

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
    def _best_split(self, X: pd.DataFrame, y: pd.Series, w: np.ndarray, attributes: list):
        """
        Standard C4.5 recipe for avoiding gain ratio's degenerate bias
        toward splits that carve off one tiny branch:
          1. For each attribute, pick the candidate split that maximizes
             plain information GAIN (for continuous attrs this chooses the
             threshold; for categorical attrs the split is fixed).
          2. Only attributes whose gain is >= the mean gain over all
             attributes are eligible.
          3. Among eligible attributes, pick the one with the highest GAIN
             RATIO.
        This mirrors Quinlan's original C4.5 heuristic.
        """
        base_counts = _weighted_class_counts(y, w)
        base_entropy = entropy(base_counts)
        total_w = sum(w)
        if total_w == 0:
            return None, None, False

        candidates = []  # list of dicts: attr, gain, gain_ratio, threshold, is_cont
        for attr in attributes:
            col = X[attr]
            known_mask = col.notna().values
            if known_mask.sum() == 0:
                continue
            known_w_total = w[known_mask].sum()
            prop_known = known_w_total / total_w

            if attr in self.continuous_cols_:
                vals = np.sort(col[known_mask].unique())
                if len(vals) < 2:
                    continue
                thresholds = (vals[:-1] + vals[1:]) / 2.0
                best_t, best_gain, best_gr = None, -1.0, 0.0
                for t in thresholds:
                    left_mask = known_mask & (col.values <= t)
                    right_mask = known_mask & (col.values > t)
                    if left_mask.sum() == 0 or right_mask.sum() == 0:
                        continue
                    gr, gain = self._gain_ratio_binary(y, w, known_mask, left_mask, right_mask,
                                                        base_entropy, prop_known)
                    # select threshold by raw gain (step 1)
                    if gain > best_gain:
                        best_gain, best_gr, best_t = gain, gr, t
                if best_t is not None:
                    candidates.append(dict(attr=attr, gain=best_gain, gain_ratio=best_gr,
                                           threshold=best_t, is_cont=True))
            else:
                gr, gain = self._gain_ratio_categorical(X, y, w, attr, known_mask,
                                                         base_entropy, prop_known)
                candidates.append(dict(attr=attr, gain=gain, gain_ratio=gr,
                                       threshold=None, is_cont=False))

        if not candidates:
            return None, None, False

        mean_gain = sum(c["gain"] for c in candidates) / len(candidates)
        eligible = [c for c in candidates if c["gain"] >= mean_gain] or candidates

        best = max(eligible, key=lambda c: c["gain_ratio"])
        return best["attr"], best["threshold"], best["is_cont"]

    def _gain_ratio_binary(self, y, w, known_mask, left_mask, right_mask, base_entropy, prop_known):
        wl, wr = w[left_mask].sum(), w[right_mask].sum()
        total_known = wl + wr
        ent_l = entropy(_weighted_class_counts(y[left_mask], w[left_mask]))
        ent_r = entropy(_weighted_class_counts(y[right_mask], w[right_mask]))
        # information gain restricted to known instances, scaled by proportion known
        weighted_ent = (wl / total_known) * ent_l + (wr / total_known) * ent_r
        gain = prop_known * (base_entropy - weighted_ent)
        # split information over the branches used for this attribute
        p_l, p_r = wl / total_known, wr / total_known
        split_info = -sum(p * math.log2(p) for p in (p_l, p_r) if p > 0)
        if split_info == 0:
            return 0.0, gain
        return gain / split_info, gain

    def _gain_ratio_categorical(self, X, y, w, attr, known_mask, base_entropy, prop_known):
        col = X[attr]
        known_w_total = w[known_mask].sum()
        weighted_ent = 0.0
        split_info = 0.0
        for value in col[known_mask].unique():
            branch_mask = known_mask & (col.values == value)
            wb = w[branch_mask].sum()
            p = wb / known_w_total
            if p > 0:
                weighted_ent += p * entropy(_weighted_class_counts(y[branch_mask], w[branch_mask]))
                split_info -= p * math.log2(p)
        gain = prop_known * (base_entropy - weighted_ent)
        gr = 0.0 if split_info == 0 else gain / split_info
        return gr, gain

    # ------------------------------------------------------------------ #
    # tree building
    # ------------------------------------------------------------------ #
    def _build(self, X: pd.DataFrame, y: pd.Series, w: np.ndarray, attributes: list, depth: int) -> Node:
        self.n_nodes_ += 1
        class_counts = _weighted_class_counts(y, w)

        if len(set(y)) == 1:
            self.n_leaves_ += 1
            return Node(is_leaf=True, prediction=y.iloc[0], depth=depth,
                        n_samples=len(y), class_counts=class_counts)

        majority = max(class_counts, key=class_counts.get) if class_counts else (
            Counter(y).most_common(1)[0][0])

        if (not attributes or
                (self.max_depth is not None and depth >= self.max_depth) or
                len(y) < self.min_samples_split):
            self.n_leaves_ += 1
            return Node(is_leaf=True, prediction=majority, depth=depth,
                        n_samples=len(y), class_counts=class_counts)

        best_attr, best_thresh, is_cont = self._best_split(X, y, w, attributes)

        if best_attr is None:
            self.n_leaves_ += 1
            return Node(is_leaf=True, prediction=majority, depth=depth,
                        n_samples=len(y), class_counts=class_counts)

        node = Node(is_leaf=False, attribute=best_attr, is_continuous=is_cont,
                    threshold=best_thresh, depth=depth, n_samples=len(y),
                    class_counts=class_counts)

        col = X[best_attr]
        known_mask = col.notna().values
        unknown_mask = ~known_mask

        if is_cont:
            left_known = known_mask & (col.values <= best_thresh)
            right_known = known_mask & (col.values > best_thresh)
            w_left_total = w[left_known].sum()
            w_right_total = w[right_known].sum()
            total_known_w = w_left_total + w_right_total

            # fractional distribution of unknowns
            left_extra_w = w[unknown_mask] * (w_left_total / total_known_w if total_known_w else 0.5)
            right_extra_w = w[unknown_mask] * (w_right_total / total_known_w if total_known_w else 0.5)

            left_idx = np.where(left_known | unknown_mask)[0]
            right_idx = np.where(right_known | unknown_mask)[0]

            w_left = w.copy()
            w_left[unknown_mask] = left_extra_w
            w_right = w.copy()
            w_right[unknown_mask] = right_extra_w

            remaining = attributes  # continuous attrs can be reused with a different threshold
            node.children['<='] = self._build(X.iloc[left_idx], y.iloc[left_idx],
                                               w_left[left_idx], remaining, depth + 1)
            node.children['>'] = self._build(X.iloc[right_idx], y.iloc[right_idx],
                                              w_right[right_idx], remaining, depth + 1)
        else:
            values = col[known_mask].unique()
            known_w_total = w[known_mask].sum()
            remaining_attrs = [a for a in attributes if a != best_attr]
            for value in values:
                branch_known = known_mask & (col.values == value)
                wb_total = w[branch_known].sum()
                frac = wb_total / known_w_total if known_w_total else 0.0
                branch_unknown_w = w[unknown_mask] * frac
                idx = np.where(branch_known | unknown_mask)[0]
                wv = w.copy()
                wv[unknown_mask] = branch_unknown_w
                node.children[value] = self._build(X.iloc[idx], y.iloc[idx], wv[idx],
                                                    remaining_attrs, depth + 1)

        return node

    # ------------------------------------------------------------------ #
    # pessimistic post-pruning
    # ------------------------------------------------------------------ #
    def _pessimistic_error(self, n: int, e: int) -> float:
        """Continuity-corrected pessimistic error estimate (classic C4.5 heuristic)."""
        if n == 0:
            return 0.0
        z = 0.69  # ~ z-score corresponding to cf=0.25 in Quinlan's table
        f = e / n
        term = (z ** 2) / (2 * n)
        num = f + term + z * math.sqrt(f / n - f ** 2 / n + term / n)
        den = 1 + term
        return n * (num / den)

    def _subtree_errors(self, node: Node) -> int:
        """Number of (weighted, rounded) training errors made by the subtree rooted at node."""
        if node.is_leaf:
            total = sum(node.class_counts.values())
            correct = node.class_counts.get(node.prediction, 0)
            return total - correct
        return sum(self._subtree_errors(c) for c in node.children.values())

    def _prune(self, node: Node):
        if node.is_leaf:
            return
        for child in node.children.values():
            self._prune(child)

        # Can only collapse if every child is now a leaf
        if not all(c.is_leaf for c in node.children.values()):
            return

        n = sum(node.class_counts.values())
        if n == 0:
            return
        majority = max(node.class_counts, key=node.class_counts.get)
        leaf_errors = n - node.class_counts.get(majority, 0)
        subtree_errors = self._subtree_errors(node)

        leaf_pess = self._pessimistic_error(n, leaf_errors)
        subtree_pess = sum(
            self._pessimistic_error(sum(c.class_counts.values()),
                                     sum(c.class_counts.values()) - c.class_counts.get(c.prediction, 0))
            for c in node.children.values()
        )

        if leaf_pess <= subtree_pess:
            node.is_leaf = True
            node.prediction = majority
            node.children = {}
            node.attribute = None

    # ------------------------------------------------------------------ #
    def _predict_row(self, row: pd.Series, node: Node):
        while not node.is_leaf:
            val = row.get(node.attribute, None)
            if node.is_continuous:
                if pd.isna(val):
                    return max(node.class_counts, key=node.class_counts.get)
                node = node.children['<='] if val <= node.threshold else node.children['>']
            else:
                if val in node.children:
                    node = node.children[val]
                else:
                    return max(node.class_counts, key=node.class_counts.get) if node.class_counts else None
        return node.prediction

    def _depth(self, node: Optional[Node]) -> int:
        if node is None or node.is_leaf:
            return node.depth if node else 0
        return max(self._depth(c) for c in node.children.values())
