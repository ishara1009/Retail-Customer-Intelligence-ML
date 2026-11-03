
# Retail Customer Intelligence ML

A beginner-friendly Python 3.11+ portfolio project that uses fictional retail
data to answer three business questions: how much a customer may spend in the
target month, whether they may buy, and which geometrically similar customer
groups appear in a single snapshot. A separate GridWorld teaches reinforcement
learning. This is educational work, not a production decision system.

## Dataset and leakage-safe design

The synthetic dataset has 1,000 customers, 22,135 transactions, and 12,000
customer-month rows. One monthly row contains identifiers, seven historical
features, and two targets. At snapshot `2025-04-01`, history is strictly before
April 1 (the 90-day window is `[2025-01-01, 2025-04-01)`) and the target is the
immediately following month window `[2025-04-01, 2025-05-01)`. Zero-spend rows
and blank recency for customers without a previous purchase are retained.

Models use an explicit seven-feature allowlist. IDs, dates, and **both** targets
are excluded from every supervised input. Imputation, scaling, and feature
selection live inside pipelines fitted on training data. Splits are chronological:

- Train: 2025-04-01 through 2025-11-01 (8,000 rows)
- Validation: 2025-12-01 through 2026-01-01 (2,000 rows)
- Test: 2026-02-01 through 2026-03-01 (2,000 rows)

The same customers recur, so test results estimate future monthly outcomes for
existing customers—not generalization to entirely unseen customers. Earlier
transactions may legitimately become historical inputs for a later forecast.
No ordinary row-level K-fold split is used.

## Setup

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run from the repository root:

```bash
python main.py --task regression
python main.py --task classification
python main.py --task clustering
python main.py --task experiments
python main.py --task rl
python main.py --task all
python -m reinforcement_learning.run_demo
pytest
```

`python scripts/generate_dataset.py` safely preserves the supplied processed
table. The original random-data generator was not supplied, so the script never
replaces raw synthetic data. An explicit `--force-rebuild` can reconstruct the
monthly table from rounded raw amounts; rare derived values may differ by LKR
0.01 from values originally calculated before rounding.

## Models and measured results

Regression compares a mean baseline, Linear Regression, and small Decision
Trees using validation MAE; the winner is refitted on train + validation and
tested once. Classification similarly compares a most-frequent baseline,
Logistic Regression, and trees using validation F1 at threshold 0.5. Clustering
standardizes recency, order count, and spend at 2025-11-01, then selects `k`
from 2–6 using silhouette score. Cluster IDs are arbitrary; silhouette measures
geometric separation, not business usefulness.

The verified run in this environment produced the following results (seed 42):

| Task | Validation choice | Reserved test / evaluation result |
|---|---|---|
| Regression | Linear Regression, validation MAE 7,249.37 LKR | MAE 6,596.27 LKR; RMSE 10,779.12 LKR; R² 0.2314 |
| Classification | depth-2 tree, validation F1 0.7850 | accuracy 0.7000; precision 0.7633; recall 0.7265; F1 0.7445; ROC-AUC 0.7486 |
| Clustering | k=3, silhouette 0.5123 | cluster sizes 141, 672, and 187 at 2025-11-01 |
| GridWorld | greedy policy after 1,000 episodes | 100% success, mean reward 5.0, mean length 6.0 (100 seeded episodes) |

The uniform-random RL baseline had 63% success, mean reward -29.91, and mean
length 36.84 in the same seeded 100-episode evaluation. These are measured
educational results, not promised future performance. Full values are written to
`outputs/metrics/`. No negative regression predictions occurred in this run;
the code would report rather than silently clip them. For classification, false
positives may waste outreach while false negatives may miss interested
customers; this demonstration is not deployment-ready.

The RL demo trains tabular Q-learning in a 4×4 GridWorld and compares its greedy
policy with uniform random actions. It is independent of retail transactions.
Reaching the goal is terminal; hitting 50 steps is truncation.

## Project map

- `src/`: validated data access, pipelines, models, evaluation, and plots.
- `experiments/`: complexity, learning-curve, bias/variance, and selection demos.
- `reinforcement_learning/`: GridWorld and tabular Q-learning.
- `notebooks/`: exploration only; model logic remains in Python modules.
- `docs/`: English fundamentals and an interview-oriented Sinhala walkthrough.
- `tests/`: data semantics, preprocessing leakage, and GridWorld behavior.
- `outputs/`: reproducible generated metrics, predictions, and figures.

## Syllabus coverage

| Topic | Where to learn it |
|---|---|
| What is ML; AI vs ML vs DL | `docs/ml_fundamentals.md` |
| Supervised learning | regression/classification modules |
| Unsupervised learning | `src/clustering.py` |
| Reinforcement learning | `reinforcement_learning/` |
| Regression | `src/regression.py` |
| Classification | `src/classification.py` |
| Clustering | `src/clustering.py` |
| Training vs testing | `src/preprocessing.py` |
| Over/underfitting | `experiments/overfitting_underfitting.py` |
| Bias and variance | `experiments/bias_variance.py` |
| Feature selection | `experiments/feature_selection.py` |
| Model evaluation | `src/evaluation.py`, generated metrics |

## Limitations

The data are synthetic, history is short, customer behavior can change, and no
causal or fairness claim is made. Repeated customers make this a rolling future
forecast evaluation. Hyperparameter searches are intentionally small. Cluster
quality is not business validation. The RL environment is only a teaching toy.
