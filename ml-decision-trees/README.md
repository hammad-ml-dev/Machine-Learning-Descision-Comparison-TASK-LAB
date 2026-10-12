# Decision Trees from Scratch: ID3 vs. C4.5 vs. CART

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-15%20passing-brightgreen.svg)](tests/test_trees.py)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Dependencies](https://img.shields.io/badge/core-NumPy%20%2B%20pandas-informational.svg)](requirements.txt)

From-scratch implementations of the three classical decision-tree induction
algorithms, **ID3**, **C4.5** and **CART**, built with only NumPy and pandas
(no `sklearn.tree` internals), plus a reproducible benchmark that compares
them against each other and against scikit-learn's `DecisionTreeClassifier`
on four datasets.

Metrics reported: accuracy, precision, recall, F1, tree depth, node count,
leaf count, training time and prediction time.

![Accuracy summary](results/summary.png)

---

## Table of contents

- [Why three implementations?](#why-three-implementations)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Library usage](#library-usage)
- [Datasets](#datasets)
- [Results](#results)
- [Implementation notes](#implementation-notes)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Why three implementations?

All three algorithms greedily choose, at every node, the feature (and, for
C4.5 and CART, the threshold) that best separates the classes. They differ in
what "best" means and in how a split is shaped:

| | **ID3** (Quinlan, 1986) | **C4.5** (Quinlan, 1993) | **CART** (Breiman et al., 1984) |
|---|---|---|---|
| Split criterion | Information Gain | Gain Ratio | Gini impurity |
| Branching | Multi-way | Multi-way (categorical), binary (continuous) | Strictly binary |
| Continuous features | Not supported natively (pre-bucketed) | Native threshold search | Native threshold search |
| Categorical features | Multi-way | Multi-way | Binary subset search |
| Missing values | Not handled | Fractional instance weighting | Not handled |
| Pruning | None | Pessimistic error-based post-pruning | Cost-complexity pruning (`ccp_alpha`) |
| Known bias | Favors high-cardinality attributes | Corrects the cardinality bias | Favors attributes with many possible binary splits |

The math behind each criterion:

```
Entropy        H(S)        = -Σ p_i · log2(p_i)
Info Gain      IG(S, A)    = H(S) - Σ_v (|S_v| / |S|) · H(S_v)
Split Info     SI(S, A)    = -Σ_v (|S_v| / |S|) · log2(|S_v| / |S|)
Gain Ratio     GR(S, A)    = IG(S, A) / SI(S, A)
Gini           Gini(S)     = 1 - Σ p_i²
```

All three classifiers expose the same scikit-learn-style `fit` / `predict` /
`score` API, so they are interchangeable in the benchmark.

## Project structure

```
.
├── src/
│   ├── id3.py        # ID3: information gain, categorical, multi-way splits
│   ├── c45.py        # C4.5: gain ratio, continuous features, missing values,
│   │                 #       pessimistic post-pruning
│   ├── cart.py       # CART: Gini, strictly binary splits, cost-complexity pruning
│   └── utils.py      # shared evaluation helpers (metrics, timing)
├── tests/
│   └── test_trees.py # 15 unit tests (pytest)
├── compare.py        # end-to-end benchmark + plots
├── results/          # generated metrics.csv and PNG charts
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick start

```bash
git clone https://github.com/hammad-ml-dev/Machine-Learning-Descision-Comparison-TASK-LAB.git
cd Machine-Learning-Descision-Comparison-TASK-LAB

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
python compare.py               # run the benchmark
pytest -v                       # run the test suite
```

`compare.py` prints a results table per dataset, writes everything to
`results/metrics.csv`, and saves `comparison_<dataset>.png` bar charts and a
`summary.png` heatmap to `results/`. It needs no network access: every dataset
is either bundled with scikit-learn or generated locally.

## Library usage

```python
from src.id3 import ID3Classifier
from src.c45 import C45Classifier
from src.cart import CARTClassifier

# X_train / X_test: pandas DataFrames, y_train / y_test: pandas Series
id3 = ID3Classifier(max_depth=5).fit(X_train, y_train)
print("ID3 accuracy:", id3.score(X_test, y_test))

c45 = C45Classifier(max_depth=5, prune=True, cf=0.25).fit(
    X_train, y_train, continuous_cols=["age", "income"])
print("C4.5 accuracy:", c45.score(X_test, y_test))

cart = CARTClassifier(max_depth=5, ccp_alpha=0.01).fit(
    X_train, y_train, continuous_cols=["age", "income"])
print("CART accuracy:", cart.score(X_test, y_test))
```

## Datasets

| Dataset | Rows | Features | Classes | Purpose |
|---|---|---|---|---|
| `play_tennis` | 14 | 4 categorical | 2 | Classic textbook set; sanity-checks ID3's `outlook` root split |
| `car_evaluation` | 1200 | 6 categorical | 4 | Locally generated, UCI-style; stresses multi-way categorical splits |
| `iris` | 150 | 4 continuous | 3 | Easy, near-linearly-separable baseline |
| `breast_cancer` | 569 | 30 continuous | 2 | Highest-dimensional dataset in the suite |

All models use a 70/30 train/test split and `max_depth=8`. Continuous columns
are bucketed with `pd.cut` before being passed to ID3, which only handles
categorical attributes.

## Results

Test-set accuracy per model and dataset (full numbers, including precision,
recall, F1 and timings, are in [`results/metrics.csv`](results/metrics.csv)):

| Dataset | ID3 | C4.5 | CART (ours) | sklearn CART |
|---|---|---|---|---|
| play_tennis | 0.600 | 0.400 | 0.600 | 0.400 |
| car_evaluation | 0.869 | **0.889** | 0.772 | 0.764 |
| iris | 0.889 | 0.911 | 0.911 | **0.933** |
| breast_cancer | **0.924** | **0.924** | 0.901 | 0.901 |
| **Mean** | **0.821** | 0.781 | 0.796 | 0.750 |

Tree size (number of nodes):

| Dataset | ID3 | C4.5 | CART (ours) | sklearn CART |
|---|---|---|---|---|
| play_tennis | 8 | 8 | 7 | 7 |
| car_evaluation | 397 | 392 | 259 | 239 |
| iris | 14 | 15 | 15 | 15 |
| breast_cancer | 84 | **41** | 37 | 37 |

### Key takeaways

- **C4.5 gives the most compact accurate trees.** On `breast_cancer` it
  matches ID3's accuracy (0.924) with about half the nodes (41 vs. 84), which
  is pessimistic post-pruning doing its job. On `car_evaluation` it is both
  the most accurate and slightly smaller than ID3 (392 vs. 397 nodes).
- **Our CART matches scikit-learn's CART.** On `breast_cancer` the two produce
  identical accuracy and node counts (37 nodes each), a good sign that the
  Gini, binary-split and cost-complexity-pruning logic is correct.
- **Strictly binary splits cost CART on categorical data.** A 4-way attribute
  needs several binary splits, so under the same `max_depth` budget CART trails
  ID3 and C4.5 on `car_evaluation`.
- **`iris` doesn't discriminate.** All four models land in a 0.889-0.933 band.
- **`play_tennis` is not a benchmark.** With 14 rows, the test split is 4-5
  rows and one error swings accuracy by 20-25 points. It exists to verify the
  ID3 root split, not to rank algorithms.
- **Speed.** The pure-Python implementations are slower than scikit-learn's
  compiled Cython trees (e.g. 5-9 s for C4.5 and CART vs. about 5 ms for
  scikit-learn on `breast_cancer`), as expected for a readable reference
  implementation.

### Caveats

Treat the mean accuracy ranking with caution: none of the models are
hyperparameter-tuned, all share one split and one seed, and two of the four
datasets are categorical, which favors ID3's native multi-way splits. The goal
is to make *why* the algorithms diverge visible, not to crown a universal
winner.

## Implementation notes

- **ID3** (`src/id3.py`) grows a multi-way tree using information gain until
  nodes are pure, attributes run out, or `max_depth` is reached. Unseen
  categories at prediction time fall back to the node's majority class.
- **C4.5** (`src/c45.py`) follows Quinlan's heuristic: only attributes with
  above-average information gain are ranked by gain ratio, which avoids gain
  ratio over-rewarding splits that carve off one tiny branch. Missing values
  are handled by distributing fractional instance weights across branches.
  Pruning uses a continuity-corrected pessimistic error estimate (`cf=0.25`
  by default, matching C4.5's classic `-c` parameter).
- **CART** (`src/cart.py`) searches midpoint thresholds for continuous
  attributes and all binary partitions of the category set for categorical
  ones. Minimal cost-complexity pruning repeatedly collapses the weakest-link
  node while its effective alpha is at most `ccp_alpha`, mirroring
  scikit-learn's pruning path.

## Testing

```bash
pytest -v
```

The 15 tests cover:

- Entropy, Gini and information-gain correctness and non-negativity
- ID3: textbook `outlook` root split on Play-Tennis, memorization of noiseless
  training data, `max_depth` handling
- C4.5: XOR-style non-linear data, pruning never increasing tree size,
  prediction with missing values
- CART: every internal node has exactly two children, continuous features,
  `ccp_alpha` shrinking the tree
- A cross-algorithm sanity check on a trivially separable dataset

## Troubleshooting

**Windows: `ImportError: DLL load failed ... An Application Control policy has
blocked this file.`** This is not a project bug. Windows Smart App Control (or a
corporate WDAC policy) can block the unsigned compiled `.pyd`/`.dll` files that
NumPy, pandas and scikit-learn ship. Turn off Smart App Control under
*Windows Security -> App & browser control* and restart, ask IT to allow-list
Python, or install the packages through conda/Miniconda.

## License

Released under the [MIT License](LICENSE).

**Author:** Hammad Durrani
