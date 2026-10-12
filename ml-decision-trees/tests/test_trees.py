"""
Basic correctness and sanity tests for ID3, C4.5, and CART.

Run with:  pytest -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.id3 import ID3Classifier, entropy, information_gain
from src.c45 import C45Classifier
from src.cart import CARTClassifier, gini


# --------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------- #
@pytest.fixture
def play_tennis():
    data = {
        "outlook":     ["sunny", "sunny", "overcast", "rain", "rain", "rain", "overcast",
                         "sunny", "sunny", "rain", "sunny", "overcast", "overcast", "rain"],
        "humidity":    ["high", "high", "high", "high", "normal", "normal", "normal",
                         "high", "normal", "normal", "normal", "high", "normal", "high"],
        "wind":        ["weak", "strong", "weak", "weak", "weak", "strong", "strong",
                         "weak", "weak", "weak", "strong", "strong", "weak", "strong"],
        "play":        ["no", "no", "yes", "yes", "yes", "no", "yes",
                         "no", "yes", "yes", "yes", "yes", "yes", "no"],
    }
    df = pd.DataFrame(data)
    return df.drop(columns=["play"]), df["play"]


@pytest.fixture
def xor_continuous():
    """A simple 2D dataset that is NOT linearly separable, to check that
    the trees can actually carve out nonlinear decision regions."""
    rng = np.random.RandomState(0)
    n = 200
    X = pd.DataFrame({
        "x1": rng.uniform(-1, 1, n),
        "x2": rng.uniform(-1, 1, n),
    })
    y = pd.Series(np.where((X["x1"] > 0) ^ (X["x2"] > 0), "A", "B"))
    return X, y


# --------------------------------------------------------------------- #
# metric building blocks
# --------------------------------------------------------------------- #
def test_entropy_pure_is_zero():
    assert entropy(pd.Series(["yes", "yes", "yes"])) == 0.0


def test_entropy_balanced_is_one():
    assert entropy(pd.Series(["yes", "no"])) == pytest.approx(1.0)


def test_gini_pure_is_zero():
    assert gini(pd.Series(["a", "a", "a"])) == 0.0


def test_gini_balanced_binary_is_half():
    assert gini(pd.Series(["a", "b"])) == pytest.approx(0.5)


def test_information_gain_nonnegative(play_tennis):
    X, y = play_tennis
    for col in X.columns:
        assert information_gain(X, y, col) >= 0


# --------------------------------------------------------------------- #
# ID3
# --------------------------------------------------------------------- #
def test_id3_fits_and_perfectly_memorizes_training_set(play_tennis):
    X, y = play_tennis
    clf = ID3Classifier()
    clf.fit(X, y)
    preds = clf.predict(X)
    # ID3 with no pruning / no depth cap should perfectly fit noiseless
    # categorical training data it has fully seen.
    assert (preds == y.values).all()


def test_id3_root_split_is_outlook(play_tennis):
    """Outlook has the highest information gain on the classic Play-Tennis
    dataset; this is the textbook result for ID3."""
    X, y = play_tennis
    clf = ID3Classifier()
    clf.fit(X, y)
    assert clf.root.attribute == "outlook"


def test_id3_respects_max_depth(play_tennis):
    X, y = play_tennis
    clf = ID3Classifier(max_depth=1)
    clf.fit(X, y)
    assert clf.get_depth() <= 1


# --------------------------------------------------------------------- #
# C4.5
# --------------------------------------------------------------------- #
def test_c45_handles_continuous_features(xor_continuous):
    X, y = xor_continuous
    clf = C45Classifier(max_depth=5, prune=False)
    clf.fit(X, y)
    preds = clf.predict(X)
    acc = (preds == y.values).mean()
    assert acc > 0.9  # should carve out the XOR pattern well


def test_c45_pruning_does_not_increase_tree_size(xor_continuous):
    X, y = xor_continuous
    unpruned = C45Classifier(max_depth=6, prune=False).fit(X, y)
    pruned = C45Classifier(max_depth=6, prune=True).fit(X, y)
    assert pruned.n_nodes_ <= unpruned.n_nodes_


def test_c45_handles_missing_values():
    X = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": ["x", "y", "x", "y"]})
    y = pd.Series(["c1", "c2", "c1", "c2"])
    clf = C45Classifier(max_depth=3, prune=False)
    clf.fit(X, y, continuous_cols=["a"])
    preds = clf.predict(X)
    assert len(preds) == len(y)


# --------------------------------------------------------------------- #
# CART
# --------------------------------------------------------------------- #
def test_cart_produces_strictly_binary_splits(xor_continuous):
    X, y = xor_continuous
    clf = CARTClassifier(max_depth=5)
    clf.fit(X, y)

    def check(node):
        if node.is_leaf:
            return
        assert node.left is not None and node.right is not None
        check(node.left)
        check(node.right)

    check(clf.root)


def test_cart_handles_continuous_features(xor_continuous):
    X, y = xor_continuous
    clf = CARTClassifier(max_depth=5)
    clf.fit(X, y)
    preds = clf.predict(X)
    acc = (preds == y.values).mean()
    assert acc > 0.9


def test_cart_ccp_alpha_prunes_tree(play_tennis):
    X, y = play_tennis
    unpruned = CARTClassifier(ccp_alpha=0.0).fit(X, y)
    pruned = CARTClassifier(ccp_alpha=0.3).fit(X, y)
    assert pruned.n_nodes_ <= unpruned.n_nodes_


# --------------------------------------------------------------------- #
# cross-algorithm sanity
# --------------------------------------------------------------------- #
def test_all_three_agree_reasonably_on_easy_dataset():
    """On a trivially separable dataset, all three algorithms should reach
    perfect or near-perfect training accuracy."""
    X = pd.DataFrame({"f": [1, 2, 3, 4, 5, 6, 7, 8]})
    y = pd.Series(["lo"] * 4 + ["hi"] * 4)

    id3_X = pd.DataFrame({"f": pd.cut(X["f"], bins=2, labels=["lo", "hi"])})
    id3 = ID3Classifier().fit(id3_X, y)
    c45 = C45Classifier(prune=False).fit(X, y, continuous_cols=["f"])
    cart = CARTClassifier().fit(X, y, continuous_cols=["f"])

    assert (id3.predict(id3_X) == y.values).mean() == 1.0
    assert (c45.predict(X) == y.values).mean() == 1.0
    assert (cart.predict(X) == y.values).mean() == 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
