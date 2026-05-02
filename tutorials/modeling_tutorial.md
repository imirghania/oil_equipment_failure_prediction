# Oilfield Equipment Failure Prediction — Modelling Tutorial

This document walks through every modelling step in `notebooks/model_oilfield.ipynb`
and explains the reasoning behind each decision in plain language alongside the
technical detail.

---

## Table of Contents

1. [Problem framing](#1-problem-framing)
2. [Feature engineering — drawdown and failure_pressure_index](#2-feature-engineering)
3. [Hyperparameter search with Optuna — and why we negate AUC](#3-hyperparameter-search-with-optuna)
4. [Model calibration with CalibratedClassifierCV](#4-model-calibration-with-calibratedclassifiercv)
5. [Out-of-fold (OOF) probabilities](#5-out-of-fold-oof-probabilities)
6. [The Brier score](#6-the-brier-score)
7. [Calibration curves](#7-calibration-curves)
8. [Risk scoring figures explained](#8-risk-scoring-figures-explained)
9. [End-to-end summary](#9-end-to-end-summary)

---

## 1. Problem framing

The dataset contains 796 oilfield wells. The original column `status` is
`1 = operational` and `0 = failed`. Because the convention in machine learning
is that the **positive class** (the thing we want to detect) equals 1, we flip
it:

```python
df["failure"] = 1 - df["status"]   # 1 = failed, 0 = operational
```

Two columns — `failures_last_year` and `downtime_days` — are **leaky**: they are
recorded _after_ a failure has already happened, so they would let the model
"cheat" during training. Using them would produce artificially perfect accuracy
that does not generalise to predicting failures before they occur. Both are
dropped before any modelling.

The goal is to output a **calibrated probability** of failure for each well, not
just a binary label. A probability is far more actionable than a label: an
engineer can prioritise the top-20 wells by probability rather than treating all
"failed = predicted" wells equally.

---

## 2. Feature engineering

### 2.1 `drawdown`

```python
df["drawdown"] = df["reservoir_pressure_MPa"] - df["pressure_bottomhole_MPa"]
```

**What it is physically.**
Reservoir pressure is the pressure of fluids far out in the rock formation.
Bottomhole pressure is the pressure measured at the bottom of the wellbore.
The difference is the **drawdown** — the pressure gradient that "pulls" fluids
from the formation into the well.

- A **high** drawdown means the well is pulling very aggressively. This places
  mechanical stress on the completion equipment (ESPs, packers, tubing) and can
  cause formation sand to migrate into the wellbore, accelerating equipment wear.
- A **low or negative** drawdown (bottomhole pressure ≥ reservoir pressure)
  means the well is not flowing efficiently, which can indicate a damaged pump or
  near-wellbore blockage.

Neither raw pressure column alone carries this information as cleanly. The ratio
or difference between them is what physically matters, which is why we compute
it explicitly rather than letting the model try to infer it from three separate
correlated columns. EDA already showed that high-drawdown wells have elevated
failure rates, confirming its predictive value.

### 2.2 `failure_pressure_index`

```python
df["failure_pressure_index"] = df["drawdown"] * df["scaling_flag"]
```

**What it is.**
`scaling_flag` is a binary indicator: 1 means scale deposits have been detected
in the wellbore, 0 means clean. From EDA we know that scaling wells fail at
roughly twice the rate of clean wells.

`failure_pressure_index` is a **multiplicative interaction term** between two
risk factors:
- High drawdown alone → mechanical stress risk
- Scaling alone → chemical/blockage risk
- High drawdown **and** scaling → both risk mechanisms active simultaneously

When `scaling_flag = 0` the term collapses to 0 regardless of drawdown. When
`scaling_flag = 1` the term equals the drawdown value, giving the model an
explicit signal for the worst-case combination. A tree model can in principle
discover this interaction on its own, but providing it explicitly makes the
relationship transparent and helps in low-data regimes.

---

## 3. Hyperparameter search with Optuna

### 3.1 What Optuna does

[Optuna](https://optuna.org) is an automatic hyperparameter optimisation
framework. Instead of a fixed grid of combinations, it uses a **Tree-structured
Parzen Estimator (TPE)** sampler that learns from earlier trials to focus the
search on promising regions of the hyperparameter space. Over 50 trials it
typically outperforms a brute-force grid search of hundreds of combinations.

The search works like this for each trial:

1. Optuna proposes a set of hyperparameter values.
2. We train a LightGBM (or XGBoost) model with those values using 5-fold
   cross-validation.
3. We compute the mean ROC-AUC across the 5 folds.
4. We return a scalar score back to Optuna.
5. Optuna updates its internal model of the landscape and proposes the next trial.

### 3.2 Why we return the negative AUC

```python
aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
return -aucs.mean()   # minimize negative AUC
```

Optuna studies are always configured to **minimise** their objective:

```python
study_lgbm = optuna.create_study(direction="minimize", ...)
```

ROC-AUC is a metric where **higher is better**. To make Optuna minimise
something that is equivalent to maximising AUC, we simply negate it. Minimising
`-AUC` is mathematically identical to maximising `AUC`.

When we retrieve the best result, we negate again to recover the real AUC value:

```python
best_auc_lgbm = -study_lgbm.best_value
```

`study_lgbm.best_value` stores the best (minimum) objective value seen, which
in our case is a negative number like `-0.8445`. Negating it gives `0.8445` —
the actual AUC.

Optuna does have a `direction="maximize"` option, but the negate pattern is
equally common and makes the objective function self-contained: anyone reading
`return -aucs.mean()` instantly sees that higher AUC is being sought.

---

## 4. Model calibration with `CalibratedClassifierCV`

### 4.1 The problem: raw probabilities are not calibrated

A gradient boosting model outputs a `predict_proba` value, but this value is
**not** a true probability by default. The model is optimised for ranking
(which is what AUC measures), not for the absolute value of its score. In
practice this means:

- A well that is 70% likely to fail might get a score of 0.95 from an
  uncalibrated LGBM model.
- A well that is 10% likely to fail might get a score of 0.30.

The _ranking_ is correct (0.95 > 0.30), but the _magnitude_ is wrong. If you
use these scores directly for risk tiering or maintenance scheduling, you will
mis-prioritise: everything gets bucketed into "Critical" or "Low" when many
wells actually belong in the middle tiers.

### 4.2 What CalibratedClassifierCV does

```python
lgbm_cal = CalibratedClassifierCV(lgbm_base, method="isotonic", cv=5)
```

`CalibratedClassifierCV` wraps the base model and learns a **post-hoc mapping**
from raw model scores to calibrated probabilities. The procedure is:

1. Split the data into `cv=5` folds (stratified).
2. For each fold: train the base model on the other 4 folds, produce raw scores
   on the held-out fold.
3. Fit a calibration function (isotonic regression in our case) that maps those
   raw scores to the true positive rate observed on the held-out fold.
4. Store 5 fitted base models and 5 fitted calibrators.

At prediction time, the wrapped model averages the calibrated probabilities from
all 5 (base model, calibrator) pairs.

**Why 5 folds and not the whole dataset?**
If we trained the calibrator on the same data the base model was trained on, the
base model would have already overfit to it — its raw scores on training data are
over-confident, not representative of scores it would produce on new data. Using
held-out folds prevents this leakage and gives the calibrator an honest view of
the score distribution.

### 4.3 Isotonic regression vs sigmoid (Platt scaling)

`method="isotonic"` fits a **non-parametric step function** that is constrained
to be monotonically non-decreasing:

```
raw score 0.1  →  calibrated 0.04
raw score 0.4  →  calibrated 0.18
raw score 0.7  →  calibrated 0.52
raw score 0.9  →  calibrated 0.88
```

The only constraint is that if score A > score B, then calibrated(A) ≥
calibrated(B). There is no fixed shape — it learns the shape from data.

The alternative, `method="sigmoid"`, fits a two-parameter logistic function
(Platt scaling). It is more reliable when the training set is small (< ~1000
samples) because it has fewer degrees of freedom. With 796 samples we are right
at the boundary, but isotonic was chosen because the calibration curves
(Section 7) showed it produced good results in practice.

### 4.4 A concrete mental model

Think of calibration as a **translation table**. The base model says "I give
this well a score of 0.85." The calibrator looks up its table and says "When the
base model historically scored wells around 0.85, about 72% of them actually
failed — so the calibrated probability is 0.72." The output is now directly
interpretable as a probability.

---

## 5. Out-of-fold (OOF) probabilities

### 5.1 The problem with in-sample evaluation

If you fit a model on all 796 rows and then evaluate it on the same 796 rows,
the model has already "seen" every example during training. Its predictions will
be over-optimistic — especially for tree ensembles that can memorise training
data when deep enough.

### 5.2 What OOF means

Out-of-fold (OOF) probabilities are a way to get **one probability estimate per
sample that was never influenced by that sample's own training**. The procedure:

1. Split all 796 rows into 5 folds.
2. For fold 1: train on folds 2–5, predict fold 1. Fold 1's predictions are OOF.
3. For fold 2: train on folds 1, 3–5, predict fold 2. Fold 2's predictions are OOF.
4. Repeat for all 5 folds.

After all 5 rounds, every row has exactly one OOF prediction. Concatenated, they
form a vector of 796 probabilities — one per well — none of which was produced
by a model that was trained on that well.

```python
probs_lgbm = cross_val_predict(lgbm_cal, X, y, cv=cv5, method="predict_proba")[:, 1]
```

`cross_val_predict` handles all the folding internally. The `[:, 1]` selects the
probability of the positive class (failure = 1) from the two-column output.

### 5.3 Why OOF with the calibrated wrapper

We pass `lgbm_cal` (the `CalibratedClassifierCV` wrapper) rather than
`lgbm_base` to `cross_val_predict`. This means that inside each outer fold:
- The calibrated wrapper refits its 5 inner calibration folds using only the
  outer training data.
- Predictions on the outer held-out fold are produced by the freshly calibrated
  model that never saw that fold.

The result is **doubly held-out** probabilities: honest estimates of how the
calibrated model would perform on truly unseen wells.

### 5.4 OOF as a proxy for generalisation

When we compute AUC, Brier score, and log-loss on OOF probabilities, we get an
unbiased estimate of the model's generalisation performance — equivalent to what
we would see on a completely separate test set, but making use of all available
data. This is especially valuable here because 796 samples is small by ML
standards, and holding out a fixed 20% test set would waste 160 samples.

---

## 6. The Brier score

### 6.1 Definition

The Brier score measures the **mean squared error between predicted probabilities
and actual outcomes**:

```
Brier = (1/N) × Σ (p_i − y_i)²
```

where:
- `p_i` is the model's predicted probability of failure for well `i`
- `y_i` is 1 if the well actually failed, 0 if it is operational
- `N` is the number of wells

### 6.2 What makes it useful

AUC only measures **ranking quality** — whether the model ranks failed wells
above operational ones. It does not care about the actual probability values.
You could multiply all probabilities by 10 and AUC would not change.

Brier score measures both **discrimination** (does the model separate the two
classes?) and **calibration** (are the probability values themselves accurate?).
A model that says "probability = 0.8" for every single well will have poor Brier
score even if AUC is decent, because it is never near 0 for operational wells.

### 6.3 Interpretation

- **Brier = 0.0** → perfect predictions (every `p_i` equals `y_i` exactly)
- **Brier = 0.25** → baseline: a model that always predicts the population
  failure rate of 33.4% achieves approximately this score (it is the variance
  of a Bernoulli with p = 0.334 ≈ 0.222, but predicting a constant 0.334
  yields `(0.334−1)²×0.334 + (0.334−0)²×0.666 ≈ 0.222`)
- **Brier = 1.0** → worst possible (every probability is maximally wrong)

A rule of thumb: Brier < 0.15 indicates well-calibrated probabilities. Our best
model achieves **0.1349**, comfortably below this threshold.

### 6.4 How `brier_score_loss` works in scikit-learn

```python
from sklearn.metrics import brier_score_loss
brier_score_loss(y_true, y_prob)
```

- `y_true`: the actual binary labels (0 or 1)
- `y_prob`: the model's predicted probability of the **positive class** (failure)

Internally it computes exactly `(1/N) × Σ (y_prob_i − y_true_i)²`. Despite
being called "loss" (which convention often means lower is better), it returns
a non-negative number — lower is better.

In our metrics table, we highlight the minimum Brier value in green for the same
reason as maximising AUC.

---

## 7. Calibration curves

### 7.1 The concept

A calibration curve answers: **"When the model predicts a probability of X%,
what fraction of wells at that predicted probability actually fail?"**

A perfectly calibrated model has this property: among all wells where it predicts
p = 0.7, exactly 70% fail. Among all wells where it predicts p = 0.3, exactly
30% fail. This ideal relationship is the **diagonal line** on the calibration
plot.

### 7.2 How `calibration_curve` works

```python
from sklearn.calibration import calibration_curve
frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="uniform")
```

The function:

1. Divides the [0, 1] probability range into `n_bins=10` equal-width buckets
   (because `strategy="uniform"`): [0, 0.1), [0.1, 0.2), …, [0.9, 1.0].
2. For each bucket, collects all wells whose predicted probability falls in that
   bucket.
3. Computes two numbers for each bucket:
   - `mean_pred`: the average predicted probability in this bucket
   - `frac_pos`: the fraction of wells in this bucket that actually failed

The output arrays `mean_pred` and `frac_pos` are then plotted as x and y.

### 7.3 Reading the plot

```
y-axis (frac_pos): actual failure rate in each bin
x-axis (mean_pred): average predicted probability in each bin
diagonal:  perfect calibration
```

| Curve position | Meaning |
|---|---|
| **Above** the diagonal | Model is **underconfident** — it predicts p = 0.4 but 60% of wells in that bin actually fail. You should trust higher predictions. |
| **Below** the diagonal | Model is **overconfident** — it predicts p = 0.7 but only 40% of those wells actually fail. Predictions are inflated. |
| **On** the diagonal | Perfectly calibrated — predicted probabilities equal observed rates. |

Raw LightGBM (without `CalibratedClassifierCV`) tends to cluster predictions
near 0 and 1, bowing the curve below the diagonal. After isotonic calibration
the curve hugs the diagonal closely, which is exactly what we want for credible
risk tiers.

### 7.4 In our notebook

```python
fig_cal.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode="lines",
    name="Perfect calibration",
    line=dict(color="gray", dash="dash", width=1.5),
))
```

The dashed diagonal is the "ideal" reference line. Each coloured trace is one
model. The Brier score is shown in the legend label — it summarises the total
deviation from calibrated probabilities in a single number, while the curve
shows _where_ the deviation occurs.

---

## 8. Risk scoring figures explained

After the best calibrated model (LightGBM) is fitted on the full training set,
it assigns a failure probability to every well:

```python
lgbm_cal.fit(X, y)
df["failure_prob"] = lgbm_cal.predict_proba(X)[:, 1]
df["risk_tier"] = pd.cut(
    df["failure_prob"],
    bins=[0, 0.20, 0.50, 0.80, 1.0],
    labels=["Low", "Medium", "High", "Critical"],
    include_lowest=True,
)
```

Two figures visualise the results.

### 8.1 `fig_prob_hist` — Distribution of predicted failure probability

```python
fig_prob_hist = px.histogram(
    df, x="failure_prob",
    color="risk_tier",
    color_discrete_map=TIER_COLORS,
    nbins=40,
    title="Distribution of predicted failure probability",
    ...
)
```

**What it shows.**
A histogram of all 796 wells' predicted failure probabilities, with each bar
coloured by the risk tier that probability falls into:

| Tier | Probability range | Colour |
|---|---|---|
| Low | 0 – 0.20 | Teal |
| Medium | 0.20 – 0.50 | Blue |
| High | 0.50 – 0.80 | Amber |
| Critical | 0.80 – 1.00 | Red |

**What to look for.**

- The shape of the distribution tells you how much "signal" the model has. A
  bimodal distribution (many wells near 0, many near 1, few in the middle)
  indicates the model can confidently separate the two classes. A flat or
  bell-shaped distribution indicates more uncertainty.
- The tier boundaries let you count how many wells fall in each action category.
  In our dataset: Low = 340 wells (43%), Medium = 341 (43%), High = 5 (1%),
  Critical = 110 (14%). The Critical group has a 100% actual failure rate —
  those wells truly are in the worst state according to the model.
- This figure is the first thing an operations manager would look at to decide
  how many wells to prioritise for inspection.

### 8.2 `fig_risk_scatter` — Reservoir Pressure vs Drawdown coloured by failure probability

```python
fig_risk_scatter = px.scatter(
    df,
    x="reservoir_pressure_MPa",
    y="drawdown",
    color="failure_prob",
    color_continuous_scale="RdYlGn_r",
    hover_data=["well_id", "well_type", "equipment_type", "risk_tier"],
    title="Reservoir Pressure vs Drawdown — coloured by failure probability",
    ...
)
```

**What it shows.**
A scatter plot in the engineering operating space — reservoir pressure on the
x-axis, drawdown (reservoir pressure minus bottomhole pressure) on the y-axis —
with each point coloured on a green-to-red scale by predicted failure
probability. Green = safe, red = high risk.

**Why these two axes?**
Both are primary engineered features. Drawdown is defined directly from
reservoir pressure, so the scatter is plotted in the space where the two most
physically meaningful variables live. This is also the space that a well
engineer thinks in: "where does my well sit on the pressure drawdown curve?"

**What to look for.**

- **Clusters of red points** identify high-risk operating regimes: e.g., high
  drawdown at moderate reservoir pressure, or low reservoir pressure (depleted
  reservoir) combined with any drawdown. The model has learned which corners of
  this space are dangerous.
- **Green points** with high reservoir pressure and low drawdown represent
  wells that are comfortably within a safe operating envelope.
- **Hover information** (`well_id`, `equipment_type`, `risk_tier`) lets an
  engineer click on a red point and immediately know which physical well and
  equipment type to inspect next.
- Because the colour scale is continuous (not discrete tiers), you can see
  **gradient transitions** — areas where risk increases gradually rather than
  jumping sharply, which indicates the model is interpolating sensibly rather
  than producing abrupt decision boundaries.

Together the two figures provide complementary views: the histogram tells you
**how many** wells fall in each risk category, while the scatter tells you
**which operational conditions** are driving that risk.

---

## 9. End-to-end summary

| Step | What happens | Why |
|---|---|---|
| **Target flip** | `failure = 1 − status` | Convention: positive class = the event we detect |
| **Drop leaky cols** | Remove `downtime_days`, `failures_last_year` | They are post-failure records — using them makes the model cheat |
| **Feature engineering** | Add `drawdown`, `failure_pressure_index` | Encode physical domain knowledge explicitly |
| **LabelEncode** | `equipment_type`, `well_type` → integers | Tree models handle integer categories natively |
| **Optuna 50 trials** | TPE search for LightGBM and XGBoost hyperparameters | More efficient than grid search; finds a good configuration within a practical budget |
| **Negative objective** | `return -aucs.mean()` | Turns "maximise AUC" into "minimise negative AUC" for Optuna's minimisation engine |
| **CalibratedClassifierCV** | Isotonic post-hoc calibration inside CV | Corrects raw score inflation so that p=0.7 truly means "70% chance of failure" |
| **OOF probabilities** | `cross_val_predict` on the calibrated wrapper | Honest generalisation estimates without wasting a fixed test set |
| **Brier score** | Mean squared error of probabilities | Jointly measures calibration quality and discrimination |
| **Calibration curve** | Predicted probability bin vs actual failure rate | Shows visually whether probabilities are trustworthy |
| **Risk tiers** | Cut at 0.20 / 0.50 / 0.80 | Actionable categories for maintenance scheduling |
| **fig_prob_hist** | Histogram of failure_prob coloured by tier | Shows tier distribution and model confidence across all wells |
| **fig_risk_scatter** | Reservoir pressure vs drawdown coloured by failure_prob | Maps risk onto the physical operating envelope engineers understand |
