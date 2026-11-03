
# Machine Learning Fundamentals

## 1. What is Machine Learning?

Machine Learning (ML) lets a computer learn useful patterns from examples rather
than receiving a separate hand-written rule for every case. In this project,
`src/regression.py` learns spending patterns from historical customer-month rows.

## 2. AI vs ML vs DL

Artificial Intelligence (AI) is the broad field of systems performing tasks
associated with intelligence. ML is a subset of AI. Deep Learning (DL), which
uses multilayer neural networks, is a subset of ML: **DL ⊂ ML ⊂ AI**. This small,
structured tabular dataset does not need a neural network. Simpler models train
quickly, need fewer data, and are easier for an intern to inspect and explain.

## 3–4. Supervised and unsupervised learning

Supervised learning has known targets. Regression predicts a number
(`next_month_spend_lkr`); classification predicts a class
(`will_purchase_next_month`). Unsupervised learning has no known target.
`src/clustering.py` asks KMeans to find geometric groups, without pretending
that cluster numbers are ground-truth customer types.

## 5. Reinforcement Learning

An agent chooses actions, receives rewards, and learns a policy. In
`reinforcement_learning/q_learning.py`, the agent updates a Q-table while moving
through GridWorld. This educational environment is separate from retail data.

## 6–8. Regression, classification, and clustering

- Regression predicts continuous spend and is evaluated with MAE, RMSE, and R².
- Classification predicts purchase/no purchase and uses accuracy, precision,
  recall, F1, ROC-AUC, and a confusion matrix.
- Clustering groups standardized observations; silhouette score describes
  separation and cohesion, not commercial value.

Baselines matter. A complex model should be compared with a simple mean or
most-frequent predictor before its score is considered useful.

## 9. Training, validation, and testing

Training data teach parameters such as regression coefficients or tree splits.
Validation data choose hyperparameters such as tree depth. Test data estimate
the chosen workflow once at the end. `src/preprocessing.py` uses time order,
because a forecast must not learn from its future. Preprocessors are fitted
inside pipelines, so training medians and scaling statistics do not leak.

## 10. Overfitting and underfitting

Underfitting occurs when a model is too simple to capture useful structure.
Overfitting occurs when it learns training details that do not generalize.
`experiments/overfitting_underfitting.py` measures training/validation RMSE at
several tree depths and with increasing chronological training-month prefixes.
Interpret the measured curves rather than forcing a textbook story.

## 11. Bias and variance

Bias is systematic error in the average learned prediction. Variance is how
much predictions change across independently sampled training sets. The
one-dimensional `sin(x)` simulation in `experiments/bias_variance.py` can
estimate both because it knows the noise-free truth and creates many datasets.
A single retail train/validation gap does not directly measure variance.

## 12. Feature selection

Feature selection keeps a subset of inputs. `SelectKBest` in
`experiments/feature_selection.py` is fitted only on training rows. It may help,
hurt, or preserve validation performance. Some features are redundant: total
spend is approximately order count times mean order value.

## 13. Model evaluation

MAE is average absolute currency error; RMSE penalizes large errors more; R²
compares regression with a mean prediction. Precision asks how many predicted
buyers bought; recall asks how many actual buyers were found; F1 balances both.
ROC-AUC ranks positive examples across thresholds and is undefined if only one
class is present. Metrics are implemented in `src/evaluation.py` and outputs are
saved as strict JSON.

Finally, distinguish **parameters**, which fitting learns (coefficients, tree
splits, Q-values), from **hyperparameters**, which a person or validation process
chooses (tree depth, learning rate, discount factor, or cluster count).
