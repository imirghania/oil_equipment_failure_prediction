# Oilfield Equipment EDA — Tutorial

This document walks through every section of `notebooks/eda_oilfield.ipynb` and
explains what each analysis does, why it was chosen, and what conclusions it
leads to.

---

## Table of Contents

1. [Notebook structure — how the Jupyter notebook is organised](#1-notebook-structure)
2. [Setup — imports and shared constants](#2-setup)
3. [Section 1 · Data Overview](#3-section-1--data-overview)
4. [Section 2 · Target Distribution](#4-section-2--target-distribution)
5. [Section 3 · Numeric Feature Distributions](#5-section-3--numeric-feature-distributions)
6. [Section 4 · Correlation Analysis](#6-section-4--correlation-analysis)
7. [Section 5 · Categorical Breakdowns](#7-section-5--categorical-breakdowns)
8. [Section 6 · Scaling Flag Deep-Dive](#8-section-6--scaling-flag-deep-dive)
9. [Section 7 · Pressure & Production Analysis](#9-section-7--pressure--production-analysis)
10. [Section 8 · Injection Wells Analysis](#10-section-8--injection-wells-analysis)
11. [Section 9 · Key Findings Summary](#11-section-9--key-findings-summary)

---

## 1. Notebook structure

### Layout

The notebook is a standard JupyterLab notebook (`*.ipynb`). It consists of
alternating **markdown cells** (section headings and explanatory text) and
**code cells** (analysis and visualisation). Cells are executed top-to-bottom;
variables defined in earlier cells are available to all later cells in the same
kernel session.

### Display pattern

In Jupyter the last expression in a cell is rendered as output automatically,
but for compound outputs (multiple figures + a table + printed text in one
cell) the code uses explicit `display()` calls from IPython.

```python
from IPython.display import display
display(dataframe.style.background_gradient(...))
```

`display()` forces immediate rendering of a Styler or figure in the cell
output area before execution continues, which allows a single cell to emit
several outputs in a specific order.

### Styled tables with `pandas.Styler`

Rather than a Marimo interactive table, the notebook uses `pandas.Styler` to
produce HTML-rendered tables with conditional formatting:

```python
display(
    column_overview.style
    .apply(lambda col: ["background: #FFF3CD" if v == "⚠️  LEAKY" else "" for v in col],
           subset=["Note"])
    .hide(axis="index")
)
```

`.hide(axis="index")` removes the default numeric row index from the rendered
table. `.apply()` accepts a function that receives a Series (one column) and
returns a list of CSS strings, one per cell in that column.

### Plotly figures

Plotly figures call `.show()` at the end of the cell to render them inline:

```python
fig_status_pie.show()
```

In JupyterLab with the `plotly` extension, `.show()` renders the interactive
figure in the cell output. Without the extension it falls back to a static PNG.

---

## 2. Setup

### Imports cell

```python
%matplotlib inline

import warnings
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from pathlib import Path

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
```

`%matplotlib inline` tells Jupyter to embed Matplotlib figures directly in the
notebook output cells rather than opening a separate GUI window.

Two plotting libraries are used in parallel:

| Library | Use case |
| --- | --- |
| **Plotly Express / Graph Objects** | Interactive charts — hover, zoom, and click work in JupyterLab |
| **Matplotlib / Seaborn** | Static figures for pairplots and histogram grids where tight multi-subplot layout is easier to control |

`make_subplots` (from `plotly.subplots`) is used to build the violin grid — a
single Plotly figure containing multiple violin subplots laid out in a grid.
`gaussian_kde` (from `scipy.stats`) is imported at the top level so it is
available to the histogram-grid cell without a nested import.

`sns.set_theme(style="whitegrid")` sets a clean background with subtle
gridlines for all Matplotlib/Seaborn plots globally.

### Constants cell

```python
COLOR_OPERATIONAL = "#1D9E75"   # teal
COLOR_FAILED      = "#D85A30"   # coral

PLOTLY_CONSTRUCTOR_KW = dict(template="plotly_white", height=450)
PLOTLY_LAYOUT_KW      = dict(margin=dict(l=50, r=20, t=50, b=40), font_size=12)
```

**Why two dicts — `PLOTLY_CONSTRUCTOR_KW` and `PLOTLY_LAYOUT_KW`?**
Plotly Express constructors (`px.bar`, `px.pie`, etc.) only accept a subset
of arguments — they do not accept layout-level keys like `margin` or
`font_size`. Passing those directly would raise
`TypeError: got an unexpected keyword argument`.
The split solves this:

```python
fig = px.bar(df, ..., **PLOTLY_CONSTRUCTOR_KW)   # template + height — accepted
fig.update_layout(**PLOTLY_LAYOUT_KW)            # margin + font_size — applied after
```

### Feature lists cell

```python
LEAKY_COLUMNS = ["failures_last_year", "downtime_days"]
SAFE_NUMERIC_FEATURES = [
    column for column in wells_df.select_dtypes(include="number").columns
    if column not in LEAKY_COLUMNS + ["status"]
]
```

`LEAKY_COLUMNS` names the two columns that must never be used as model inputs.
`SAFE_NUMERIC_FEATURES` is computed once and reused in four later cells
(violin grid, pairplot, histogram grid, correlation bar chart). It selects all
numeric columns except the leaky ones and the target, representing exactly the
feature set that would be safe to pass to a model.

---

## 3. Section 1 · Data Overview

### Column inventory table

```python
column_overview = pd.DataFrame({
    "Column":        wells_df.columns,
    "Dtype":         wells_df.dtypes.astype(str).values,
    "Null count":    wells_df.isnull().sum().values,
    "Unique values": wells_df.nunique().values,
    "Note": [
        "⚠️  LEAKY"  if column_name in LEAKY_COLUMNS
        else ("TARGET" if column_name == "status" else "—")
        for column_name in wells_df.columns
    ],
})

display(
    column_overview.style
    .apply(
        lambda note_column: [
            "background: #FFF3CD; font-weight:bold" if note_value == "⚠️  LEAKY"
            else ("background: #D5F5E3; font-weight:bold" if note_value == "TARGET" else "")
            for note_value in note_column
        ],
        subset=["Note"],
    )
    .hide(axis="index")
)
```

This builds a summary DataFrame covering every column at a glance: its data
type (to catch any implicit strings that should be numeric), whether it has
nulls (796 rows, 0 nulls — confirmed clean), how many unique values it has
(helpful for spotting binary flags vs continuous features), and a colour-coded
badge for leaky and target columns.

The `.apply()` call on the Styler receives the entire `"Note"` column as a
pandas Series and returns a list of CSS strings — one per cell. Amber
(`#FFF3CD`) highlights leaky columns; green (`#D5F5E3`) highlights the target.

### Dataset statistics table

```python
dataset_stats = pd.DataFrame({
    "Metric": ["Total wells", "Operational wells", "Failed wells",
               "Operational rate", "Failure rate",
               "Equipment types", "Well types"],
    "Value":  [total_wells, int(wells_df["status"].sum()), ...],
})
display(dataset_stats.style.hide(axis="index"))
```

`dataset_stats` gives the same quick-reference snapshot as stat cards would in
a dashboard: 796 wells · 66.6% operational · 33.4% failed · 5 equipment
types · 2 well types. The Styler's `.hide(axis="index")` removes the
auto-generated 0-based index column from the rendered table.

### Leaky-column warning

```python
print("⚠️  LEAKY columns — never use in any predictive model:")
for leaky_column in LEAKY_COLUMNS:
    print(f"   • {leaky_column!r} — recorded after a failure has occurred")
```

**What is data leakage?**
`failures_last_year` counts how many times a well failed in the past year.
`downtime_days` records how many days it was non-operational. Both are recorded
*after* the current `status` observation is known — they are consequences of the
outcome, not predictors of it. A model that uses them learns "failed wells have
non-zero downtime" which is tautologically true but useless for predicting
failures in the future when downtime has not yet been observed.

---

## 4. Section 2 · Target Distribution

### Pie chart

```python
status_counts = wells_df["status"].value_counts().reset_index()
status_counts.columns = ["status", "count"]
status_counts["label"] = status_counts["status"].map({1: "Operational", 0: "Failed"})

fig_status_pie = px.pie(
    status_counts, values="count", names="label",
    title="Operational vs Failed Wells",
    color="label",
    color_discrete_map={"Operational": COLOR_OPERATIONAL, "Failed": COLOR_FAILED},
    **PLOTLY_CONSTRUCTOR_KW,
)
fig_status_pie.update_traces(textinfo="percent+label+value", pull=[0.03, 0.03])
fig_status_pie.update_layout(**PLOTLY_LAYOUT_KW)
fig_status_pie.show()
```

The pie shows the class split: 530 operational (66.6%) vs 266 failed (33.4%).
This is a **moderately imbalanced** dataset — not severe enough to require
oversampling (like SMOTE), but important to note:

- Accuracy is not a useful metric here (a model that always predicts
  "operational" achieves 66.6% accuracy while being useless for detecting
  failures).
- AUC, F1, Brier score, and recall on the minority class are the appropriate
  metrics.

`pull=[0.03, 0.03]` slightly separates both slices for visual clarity.
`textinfo="percent+label+value"` overlays the slice with percentage, category
name, and raw count simultaneously.

### Countplot by well type and equipment type (Matplotlib)

```python
fig_status_bars, (ax_well_type, ax_equipment_type) = plt.subplots(1, 2, figsize=(13, 5))

for group_column, subplot_ax, chart_title, y_axis_label in [
    ("well_type",      ax_well_type,      "Status by Well Type",      "Well Type"),
    ("equipment_type", ax_equipment_type, "Status by Equipment Type", "Equipment Type"),
]:
    group_counts = (
        wells_df.groupby([group_column, "status"])
        .size().reset_index(name="count")
        .pivot(index=group_column, columns="status", values="count")
        .fillna(0)
        .rename(columns={0: "Failed", 1: "Operational"})
    )
    group_counts.plot(kind="barh", ax=subplot_ax,
                      color=[COLOR_FAILED, COLOR_OPERATIONAL])
    for bar_patch in subplot_ax.patches:
        bar_width = bar_patch.get_width()
        if bar_width > 0:
            subplot_ax.text(
                bar_width + 1,
                bar_patch.get_y() + bar_patch.get_height() / 2,
                str(int(bar_width)), va="center", fontsize=9,
            )
```

A `for` loop drives both subplots using the same logic. The approach uses
pandas `groupby → pivot` to build a crosstab, then plots it as a horizontal
bar chart directly onto a Matplotlib axis.

Value labels are added by iterating over `subplot_ax.patches` — the list of
bar rectangles. `bar_patch.get_width()` is the bar length (the count value).
The text is placed just outside the bar end (`x = bar_width + 1`) and centred
vertically at `bar_patch.get_y() + height / 2`.

**What this reveals:** Injection Pump and Gas Lift equipment have high absolute
failure counts relative to their fleet sizes. Injection wells fail more than
production wells. This motivates the dedicated injection-well deep-dive in
Section 8.

---

## 5. Section 3 · Numeric Feature Distributions

Three complementary views are shown for the numeric features.

### Violin grid (Plotly — `make_subplots` + `go.Violin`)

```python
fig_violin_grid = make_subplots(
    rows=violin_row_count, cols=violin_cols_per_row,
    subplot_titles=SAFE_NUMERIC_FEATURES,
    horizontal_spacing=0.08,
    vertical_spacing=0.14,
)

for feature_index, feature_name in enumerate(SAFE_NUMERIC_FEATURES):
    subplot_row = feature_index // violin_cols_per_row + 1
    subplot_col = feature_index %  violin_cols_per_row + 1
    for status_class, violin_color, status_label in [
        (1, COLOR_OPERATIONAL, "Operational"),
        (0, COLOR_FAILED,      "Failed"),
    ]:
        feature_values = wells_df.loc[wells_df["status"] == status_class, feature_name]
        fig_violin_grid.add_trace(
            go.Violin(
                y=feature_values,
                name=status_label,
                box_visible=True,
                meanline_visible=True,
                line_color=violin_color,
                fillcolor=violin_color,
                opacity=0.55,
                showlegend=(feature_index == 0),
                legendgroup=status_label,
            ),
            row=subplot_row, col=subplot_col,
        )

fig_violin_grid.update_layout(
    height=280 * violin_row_count,
    template="plotly_white",
    violinmode="group",
    ...
)
fig_violin_grid.show()
```

**Why `make_subplots` instead of `px.violin`?**
`px.violin` creates one standalone figure per feature. In a Jupyter notebook
there is no equivalent of Marimo's `mo.hstack` for laying out multiple Plotly
figures side-by-side in a grid. The solution is to use `make_subplots` to
allocate a grid of subplot axes within a single figure, then add raw
`go.Violin` traces to the appropriate row/col position.

The grid coordinates are computed arithmetically:

- `subplot_row = feature_index // violin_cols_per_row + 1`
- `subplot_col = feature_index %  violin_cols_per_row + 1`

With `violin_cols_per_row = 3`, the first three features go into row 1
(columns 1–3), the next three into row 2, and so on. Both `//` (floor
division) and `%` (modulo) produce 0-based indices; adding 1 converts to
Plotly's 1-based row/col addressing.

`showlegend=(feature_index == 0)` ensures that only the first feature's two
violin traces add entries to the legend — subsequent features reuse the same
`legendgroup` labels without creating duplicate legend items.

`box_visible=True` overlays the box-and-whisker inside each violin.
`meanline_visible=True` adds a line at the mean position.

**What to look for:** If the median lines of the two violins are clearly
separated vertically, that feature is a strong univariate predictor. If the
violins overlap completely, the feature may still be useful in combination with
others but is weak alone.

### Pairplot (Seaborn)

```python
target_correlations = (
    wells_df[SAFE_NUMERIC_FEATURES + ["status"]]
    .corr()["status"].drop("status")
    .abs().sort_values(ascending=False)
)
top_four_features = target_correlations.head(4).index.tolist()
pairplot_df       = wells_df[top_four_features + ["status"]].copy()
pairplot_df["Status"] = pairplot_df["status"].map({1: "Operational", 0: "Failed"})

pairplot_grid = sns.pairplot(
    pairplot_df, vars=top_four_features, hue="Status",
    palette={"Operational": COLOR_OPERATIONAL, "Failed": COLOR_FAILED},
    diag_kind="kde",
    plot_kws={"alpha": 0.45, "s": 12},
    height=2.4,
)
```

First, `target_correlations` ranks every safe numeric feature by its absolute
Pearson correlation with `status`. `top_four_features` takes the top 4. A
pairplot then shows every pairwise scatter between those four features plus
their univariate KDE densities on the diagonal, with points coloured by actual
failure status.

**What this reveals:**

- **Off-diagonal scatter plots:** Clusters of teal (operational) and coral
  (failed) points that are spatially separated indicate that the pair of
  features, used together, can linearly separate the two classes.
- **Diagonal KDE:** Shows the per-class distribution of each individual feature.
  Separated peaks → strong predictor. Overlapping peaks → weak predictor.
- **Why the top 4 and not all 8?** A pairplot of 8 features would produce 64
  subplots, making each tiny and unreadable. The top-4 by |r| captures the most
  informative features while keeping the figure legible.

`alpha=0.45` is set because with 796 points, overplotting would obscure density
differences between the two classes. `s=12` makes the markers small enough
that the overall cloud shape is visible.

### Histogram grid with KDE (Matplotlib)

```python
fig_histogram_grid, histogram_axes = plt.subplots(
    histogram_row_count, histogram_cols_per_row,
    figsize=(14, histogram_row_count * 3.5),
)
histogram_axes = histogram_axes.flatten()

for feature_idx, feature_name in enumerate(SAFE_NUMERIC_FEATURES):
    subplot_ax = histogram_axes[feature_idx]
    for status_class, class_color, class_label in [
        (1, COLOR_OPERATIONAL, "Operational"),
        (0, COLOR_FAILED,      "Failed"),
    ]:
        feature_data = wells_df.loc[wells_df["status"] == status_class, feature_name].dropna()
        subplot_ax.hist(feature_data, bins=25, color=class_color,
                        alpha=0.55, label=class_label, density=True)
        # Guard: gaussian_kde fails on constant data (e.g. scaling_flag filtered by class)
        if len(feature_data) > 1 and feature_data.std() > 1e-10:
            kernel_density_estimator = gaussian_kde(feature_data)
            kde_x_values = np.linspace(feature_data.min(), feature_data.max(), 200)
            subplot_ax.plot(kde_x_values, kernel_density_estimator(kde_x_values),
                            color=class_color, lw=1.8)

for empty_panel_idx in range(feature_idx + 1, len(histogram_axes)):
    histogram_axes[empty_panel_idx].set_visible(False)
```

`density=True` normalises each histogram to a probability density (area = 1)
so the operational class (530 wells) and failed class (266 wells) can be
overlaid on the same scale without the operational bars dwarfing the others.

The KDE (Kernel Density Estimate) is a smooth curve fitted over the histogram.
`gaussian_kde` places a small Gaussian "bump" at each data point and sums them
to produce a smooth density estimate.

**Why the guard `if len(feature_data) > 1 and feature_data.std() > 1e-10`?**
`scaling_flag` is binary (values 0 and 1 only). When filtered to a single
class, it may contain only one unique value — e.g., all wells with
`scaling_flag = 0` happen to be operational in one class split. In that case,
the data has zero variance, and `gaussian_kde` fails with a
"singular covariance matrix" error. The guard skips the KDE in this case;
the histogram still renders correctly.

`histogram_axes.flatten()` converts the 2D array of axes returned by
`plt.subplots` into a 1D array so it can be indexed with a single integer
`feature_idx`. The `empty_panel_idx` loop hides any unused subplot panels
(when `len(SAFE_NUMERIC_FEATURES)` is not a multiple of `histogram_cols_per_row`).

`sns.despine(ax=subplot_ax)` removes the top and right axis borders for a
cleaner look consistent with the `style="whitegrid"` theme.

---

## 6. Section 4 · Correlation Analysis

### Full correlation heatmap (Seaborn)

```python
excluded_columns   = LEAKY_COLUMNS + ["well_id"]
correlation_matrix = (
    wells_df.drop(columns=[col for col in excluded_columns if col in wells_df.columns])
    .select_dtypes(include="number")
    .corr()
)
upper_triangle_mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

fig_correlation_heatmap, ax_heatmap = plt.subplots(figsize=(9, 7))
sns.heatmap(
    correlation_matrix, mask=upper_triangle_mask, annot=True, fmt=".2f",
    cmap="coolwarm", center=0, vmin=-1, vmax=1,
    ax=ax_heatmap, linewidths=0.4, annot_kws={"size": 8},
)
```

`.corr()` computes the pairwise Pearson correlation matrix across all retained
numeric columns. Pearson r measures **linear** association:

- r = +1: perfect positive linear relationship
- r = 0: no linear relationship (may still have nonlinear association)
- r = −1: perfect negative linear relationship

`np.triu(np.ones_like(..., dtype=bool))` creates a boolean mask for the upper
triangle, including the diagonal. Passing it to `mask=` in `sns.heatmap` hides
those cells, showing only the lower triangle. This removes redundancy — the
correlation matrix is symmetric, so the upper triangle is a mirror image of the
lower.

`cmap="coolwarm"` with `center=0` maps negative correlations to blue,
near-zero to white, and positive to red. `vmin=-1, vmax=1` anchors the scale
to the full range so all heatmaps in the project use the same colour meaning.

**Key finding:** The dataset shows that `pressure_bottomhole_MPa` and
`pressure_wellhead_MPa` are moderately correlated (r ≈ 0.40). The notebook
prints the inter-pressure correlation matrix explicitly after the bar chart to
make this visible, and recommends using the derived `drawdown_MPa` feature
(reservoir pressure minus bottomhole pressure) to consolidate redundant signal
before training linear or regularised models.

### Feature–status correlation bar chart (Plotly)

```python
feature_status_correlations = (
    wells_df[SAFE_NUMERIC_FEATURES + ["status"]]
    .corr()["status"].drop("status")
    .sort_values(key=abs, ascending=False)
)
correlation_bar_df = feature_status_correlations.reset_index()
correlation_bar_df.columns = ["Feature", "Correlation"]
correlation_bar_df["Direction"] = correlation_bar_df["Correlation"].apply(
    lambda correlation_value:
        "Positive (→ operational)" if correlation_value >= 0 else "Negative (→ failure)"
)

fig_correlation_bar = px.bar(
    correlation_bar_df, x="Correlation", y="Feature", orientation="h",
    color="Direction",
    color_discrete_map={
        "Positive (→ operational)": COLOR_OPERATIONAL,
        "Negative (→ failure)":     COLOR_FAILED,
    },
    ...
)
fig_correlation_bar.update_layout(
    yaxis={"categoryorder": "total ascending"},
    **PLOTLY_LAYOUT_KW,
)
fig_correlation_bar.show()
```

This extracts just the column of correlations with `status` (the target) and
ranks features by absolute correlation magnitude. Colour encodes direction:
teal for positive (higher feature value → more likely operational), coral for
negative (higher value → more likely failed).

`yaxis={"categoryorder": "total ascending"}` sorts the horizontal bars so
the longest bar is at the top — making the most predictive features immediately
visible without having to scan the chart.

The `Direction` label includes the engineering interpretation
("→ operational" / "→ failure") directly in the legend, so the chart is
self-explanatory without needing to recall that `status = 1` is operational.

---

## 7. Section 5 · Categorical Breakdowns

### Failure rate by equipment type and well type

```python
for group_column, chart_title, y_axis_label in [
    ("equipment_type", "Failure Rate (%) by Equipment Type", "Equipment Type"),
    ("well_type",      "Failure Rate (%) by Well Type",      "Well Type"),
]:
    failure_rate_table = (
        wells_df.groupby(group_column)["status"]
        .agg(failed_count=lambda x: (x == 0).sum(), total_count="count")
        .assign(failure_rate_pct=lambda d: d["failed_count"] / d["total_count"] * 100)
        .sort_values("failure_rate_pct", ascending=False)
        .reset_index()
    )
    failure_rate_fig = px.bar(
        failure_rate_table,
        x="failure_rate_pct", y=group_column, orientation="h",
        text=failure_rate_table["failure_rate_pct"].round(1).astype(str) + "%",
        ...
    )
    failure_rate_fig.update_layout(
        xaxis_range=[0, failure_rate_table["failure_rate_pct"].max() * 1.25],
        **PLOTLY_LAYOUT_KW,
    )
    failure_rate_fig.show()
```

A single `for` loop generates two separate figures — one for equipment type and
one for well type — using identical logic. The `groupby().agg()` pipeline
computes two aggregations simultaneously using named aggregation:
`failed_count` counts wells where `status == 0`, and `total_count` counts all
wells in the group. `.assign(failure_rate_pct=...)` then adds the rate as a
derived column in a single chain.

`xaxis_range=[0, max * 1.25]` extends the x-axis 25% beyond the longest bar,
giving space for the text label to appear without being clipped by the chart
border.

**What this reveals:**
Injection Pump has the highest failure rate among equipment types. The
per-category rates immediately show which equipment deserves priority inspection
and maintenance scheduling.

### Equipment Type × Well Type pivot heatmap

```python
failure_rate_pivot = wells_df.pivot_table(
    values="status",
    index="equipment_type",
    columns="well_type",
    aggfunc=lambda status_values: round((status_values == 0).mean() * 100, 1),
)

fig_failure_rate_heatmap, ax_pivot_heatmap = plt.subplots(figsize=(7, 4))
sns.heatmap(
    failure_rate_pivot, annot=True, fmt=".1f", cmap="YlOrRd",
    ax=ax_pivot_heatmap, linewidths=0.5,
    cbar_kws={"label": "Failure Rate (%)"},
)
```

`pd.pivot_table` reshapes the data from long format (one row per well) to wide
format (one row per equipment type, one column per well type). The custom
`aggfunc` computes failure rate directly:
`(status_values == 0).mean() * 100` gives the percentage of failed wells in
each cell.

`cmap="YlOrRd"` (yellow → orange → red) maps low failure rates to pale yellow
and high failure rates to deep red, making the dangerous equipment–well-type
combinations immediately visible as dark cells.

Not every combination may exist in the dataset (some cells could be NaN if a
given equipment type is never used with a given well type), but `pivot_table`
handles missing combinations gracefully by leaving them as NaN, which Seaborn
renders as white.

---

## 8. Section 6 · Scaling Flag Deep-Dive

### Stacked bar chart

```python
scaling_status_counts = (
    wells_df.groupby(["scaling_flag", "status"])
    .size().reset_index(name="well_count")
)
scaling_status_counts["Status"]  = scaling_status_counts["status"].map(
    {1: "Operational", 0: "Failed"}
)
scaling_status_counts["Scaling"] = scaling_status_counts["scaling_flag"].map(
    {0: "No Scaling  (flag = 0)", 1: "Scaling Present  (flag = 1)"}
)

fig_scaling_bar = px.bar(
    scaling_status_counts,
    x="Scaling", y="well_count", color="Status",
    barmode="stack",
    ...
)
fig_scaling_bar.show()
```

The `groupby(["scaling_flag", "status"]).size()` creates a count for each of
the four combinations: (no scaling, operational), (no scaling, failed),
(scaling, operational), (scaling, failed). The stacked bar puts operational and
failed segments on top of each other so both total fleet size and within-group
composition are visible simultaneously.

`barmode="stack"` is what tells Plotly to stack segments; the alternative
`barmode="group"` would place them side by side.

### Automatic rate computation

```python
scaling_failure_rates = (
    wells_df.groupby("scaling_flag")["status"]
    .agg(failed_count=lambda x: (x == 0).sum(), total_count="count")
    .assign(failure_rate_pct=lambda d: (d["failed_count"] / d["total_count"] * 100).round(1))
)
no_scaling_failure_rate    = scaling_failure_rates.loc[0, "failure_rate_pct"]
with_scaling_failure_rate  = scaling_failure_rates.loc[1, "failure_rate_pct"]
scaling_failure_multiplier = (
    round(with_scaling_failure_rate / no_scaling_failure_rate, 1)
    if no_scaling_failure_rate > 0 else "N/A"
)

print(f"  → Scaling wells fail at {scaling_failure_multiplier}× the rate of clean wells.")
```

The rates are computed programmatically and printed as a plain summary, so the
finding is always accurate even if the dataset changes. The
`no_scaling_failure_rate > 0` guard prevents a division-by-zero if,
hypothetically, there were no failed non-scaling wells.

**Finding:** In this dataset, `no_scaling_failure_rate = 22.7%` and
`with_scaling_failure_rate = 100.0%`, giving
`scaling_failure_multiplier = 4.4`. Scaling-affected wells fail at **4.4× the
rate** of clean wells. This makes `scaling_flag` one of the highest-weight
features in the downstream model's feature importance ranking.

---

## 9. Section 7 · Pressure & Production Analysis

### The drawdown feature

```python
pressure_analysis_df = wells_df.copy()
pressure_analysis_df["drawdown_MPa"] = (
    pressure_analysis_df["reservoir_pressure_MPa"]
    - pressure_analysis_df["pressure_bottomhole_MPa"]
)
pressure_analysis_df["Status"] = pressure_analysis_df["status"].map(
    {1: "Operational", 0: "Failed"}
)

display(
    pressure_analysis_df.groupby("Status")["drawdown_MPa"]
    .describe().round(2)
    .style.background_gradient(cmap="RdYlGn_r", axis=None)
)
```

`pressure_analysis_df` is a copy of `wells_df` augmented with the
`drawdown_MPa` derived feature. Because it is assigned to its own variable, the
original `wells_df` is unchanged and all earlier cells continue to work
correctly.

The `.describe()` table shows mean, std, min, quartiles, and max for each
status class. The `.background_gradient(cmap="RdYlGn_r", axis=None)` colours
cells across the entire table (not column-by-column) so high drawdown values
appear red, immediately confirming that failed wells have higher drawdown
statistics.

**Physical meaning:** Drawdown is the pressure gradient driving fluids from the
reservoir rock into the wellbore:

```text
drawdown = reservoir_pressure_MPa − pressure_bottomhole_MPa
```

A large drawdown means the bottomhole pressure is much lower than the reservoir
pressure, so fluids rush in quickly. This imposes high mechanical stress on
downhole equipment (pumps, valves, tubing) and causes faster wear.
See Section 2 of `tutorial.md` for a fuller physical explanation.

### Reservoir vs Bottomhole pressure scatter

```python
fig_pressure_bubble_scatter = px.scatter(
    pressure_analysis_df,
    x="reservoir_pressure_MPa",
    y="pressure_bottomhole_MPa",
    color="Status",
    size="oil_rate_tpd",
    size_max=18,
    hover_data=["well_id", "equipment_type"],
    opacity=0.7,
    ...
)
fig_pressure_bubble_scatter.update_layout(**PLOTLY_LAYOUT_KW)
fig_pressure_bubble_scatter.show()
```

Plotting reservoir pressure on x and bottomhole pressure on y creates a
**two-pressure plane** where drawdown is the vertical distance from each point
to the `y = x` diagonal. Points far below the diagonal (high x, low y) have
high drawdown.

`size="oil_rate_tpd"` encodes oil production rate as bubble size, adding a
third dimension to the 2D scatter. High-rate wells with high drawdown are the
most mechanically stressed.

`hover_data=["well_id", "equipment_type"]` makes each bubble interactive:
hovering over it in JupyterLab shows the well identifier and equipment type,
letting an engineer identify specific high-risk wells directly from the plot.

`opacity=0.7` prevents overplotting from fully obscuring points in dense
regions.

### Drawdown box plots by status and well type

```python
fig_drawdown_boxplot = px.box(
    pressure_analysis_df,
    x="Status",
    y="drawdown_MPa",
    facet_col="well_type",
    color="Status",
    ...
)
fig_drawdown_boxplot.update_layout(showlegend=False, **PLOTLY_LAYOUT_KW)
fig_drawdown_boxplot.show()
```

`facet_col="well_type"` splits the figure into two side-by-side subplots —
one for injection wells and one for production wells — without needing to
filter the dataframe manually. Plotly handles the split internally.

A box plot shows five statistics: minimum, first quartile (Q1), median, third
quartile (Q3), and maximum (with outliers as dots). Comparing the two boxes
(operational vs failed) within each well type panel reveals:

- The median drawdown is higher for failed wells in both well types.
- Production wells have a wider drawdown spread than injection wells (more
  variability in how aggressively they are produced).

---

## 10. Section 8 · Injection Wells Analysis

### Injection-only scatter

```python
injection_wells_df = wells_df[wells_df["well_type"] == "injection"].copy()
injection_wells_df["Status"] = injection_wells_df["status"].map(
    {1: "Operational", 0: "Failed"}
)

fig_injection_scatter = px.scatter(
    injection_wells_df,
    x="injection_rate_m3_day",
    y="water_cut_pct",
    color="Status",
    opacity=0.7,
    ...
)
fig_injection_scatter.update_layout(**PLOTLY_LAYOUT_KW)
fig_injection_scatter.show()
```

By filtering to `injection_wells_df` only, this scatter focuses on the two
features most relevant to injection operations:

- **Injection rate (m³/day):** Volume of fluid being pumped downhole. Higher
  rates increase erosion and mechanical fatigue on pumps and valves.
- **Water cut (%):** Fraction of produced/injected fluid that is water.
  High water cut indicates corrosive conditions (water + dissolved salts) and
  can accelerate equipment corrosion.

Clusters of coral (failed) points in the high-rate, high-water-cut region of
the plot confirm that these two conditions together create a particularly
damaging environment for injection-well equipment.

### Equipment failure rate table

```python
injection_equipment_rates = (
    injection_wells_df.groupby("equipment_type")["status"]
    .agg(
        total_count="count",
        failed_count=lambda x: (x == 0).sum(),
        operational_count=lambda x: (x == 1).sum(),
    )
    .assign(
        failure_rate_pct=lambda d: (d["failed_count"] / d["total_count"] * 100).round(1)
    )
    .reset_index()
    .sort_values("failure_rate_pct", ascending=False)
    .rename(columns={
        "equipment_type":    "Equipment Type",
        "total_count":       "Total",
        "failed_count":      "Failed",
        "operational_count": "Operational",
        "failure_rate_pct":  "Failure Rate (%)",
    })
)

display(
    injection_equipment_rates.style
    .background_gradient(subset=["Failure Rate (%)"], cmap="Reds")
    .format({"Failure Rate (%)": "{:.1f}%"})
    .hide(axis="index")
)
```

This is a scoped version of the Section 5 equipment-type breakdown, restricted
to injection wells. It surfaces which equipment types are most vulnerable in
injection-specific conditions and gives absolute counts (Total, Failed,
Operational) alongside the rate so the reader can judge statistical confidence.
A category with 2 wells and 1 failure (50% rate) is less reliable than a
category with 80 wells and 40 failures (also 50% rate).

`.background_gradient(subset=["Failure Rate (%)"], cmap="Reds")` applies a
red colour gradient only to the failure rate column — the higher the rate, the
darker the red — while leaving all other columns with the default white
background.

---

## 11. Section 9 · Key Findings Summary

### Top 5 findings (markdown cell)

The five findings are stated as engineering conclusions, not statistical
observations. Each one is directly traceable to a specific earlier section:

| Finding | Evidence |
| --- | --- |
| Scaling is the strongest binary driver | Section 6: 4.4× failure rate |
| Injection Pump has highest failure rate | Section 5: categorical breakdown |
| High drawdown correlates with failure | Sections 3 & 7: violin + box plots |
| Injection wells fail more | Sections 2 & 8: countplots + scatter |
| Pressure features are multi-collinear | Section 4: heatmap, inter-pressure correlations |

### Heuristic risk score

```python
risk_scoring_df = wells_df.copy()
risk_scoring_df["risk_score"] = (
    risk_scoring_df["scaling_flag"] * 0.40
    + (
        1.0 - (
            risk_scoring_df["pressure_bottomhole_MPa"]
            / risk_scoring_df["reservoir_pressure_MPa"].clip(lower=0.01)
        ).clip(upper=1.0)
    ) * 0.30
    + (risk_scoring_df["water_cut_pct"] / 100.0) * 0.30
).round(4)
```

This formula is a **domain-knowledge weighted score**, not a trained model. It
explicitly encodes three risk factors and their relative importance:

| Component | Weight | What it measures |
| --- | --- | --- |
| `scaling_flag` | 40% | Binary presence of scale deposits |
| `1 − (BHP / reservoir_P)` | 30% | Normalised drawdown: 0 when BHP = reservoir_P (no drawdown), 1 when BHP = 0 (maximum drawdown) |
| `water_cut_pct / 100` | 30% | Normalised water fraction: 0 = pure oil, 1 = pure water |

**Why `.clip(lower=0.01)` on reservoir pressure?**
Division by zero protection: if `reservoir_pressure_MPa` is ever 0 or negative
(sensor fault, data error), dividing by it would produce infinity or NaN.
Clipping to a minimum of 0.01 MPa is a safe floor.

**Why `.clip(upper=1.0)` on the ratio?**
If `pressure_bottomhole_MPa > reservoir_pressure_MPa` (which can happen in
injection wells where the pump pressure exceeds reservoir pressure), the ratio
exceeds 1 and the term `1 − ratio` becomes negative. Clipping the ratio to 1
keeps the drawdown component non-negative and bounded to [0, 1].

**Why this is labelled a heuristic and not a prediction:**
The weights (0.40 / 0.30 / 0.30) are chosen by engineering judgement, not
optimised against the actual failure labels. A well ranked top-10 by this score
is genuinely high-risk, but the scores are not calibrated probabilities.

### Top-10 well table (Pandas Styler)

```python
top_10_risk_wells = (
    risk_scoring_df.sort_values("risk_score", ascending=False)
    .head(10)[["well_id", "well_type", "equipment_type",
               "scaling_flag", "reservoir_pressure_MPa",
               "pressure_bottomhole_MPa", "water_cut_pct",
               "risk_score", "status"]]
    .reset_index(drop=True)
)
top_10_risk_wells["actual_status"] = top_10_risk_wells["status"].map(
    {1: "Operational", 0: "Failed"}
)
top_10_risk_wells = top_10_risk_wells.drop(columns=["status"])

display(
    top_10_risk_wells.style
    .background_gradient(subset=["risk_score"], cmap="Reds")
    .apply(
        lambda status_column: [
            "background-color: #FADBD8" if status_value == "Failed"
            else "background-color: #D5F5E3"
            for status_value in status_column
        ],
        subset=["actual_status"],
    )
    .format({
        "risk_score":               "{:.4f}",
        "reservoir_pressure_MPa":   "{:.2f}",
        "pressure_bottomhole_MPa":  "{:.2f}",
        "water_cut_pct":            "{:.1f}",
    })
    .hide(axis="index")
)
```

`pandas.Styler` is used (rather than `go.Table`) because it integrates
naturally with Jupyter's HTML output and provides clean conditional formatting
with minimal code.

Two separate styling rules are applied to the same Styler object by chaining
`.background_gradient()` and `.apply()`:

- `background_gradient(subset=["risk_score"], cmap="Reds")` — red gradient
  on the risk score column; the highest scores appear darkest.
- `apply(lambda ..., subset=["actual_status"])` — per-row colouring on the
  status column: pale red (`#FADBD8`) for failed wells, pale green
  (`#D5F5E3`) for operational wells. This makes it easy to spot operational
  wells in the high-risk list at a glance (which would indicate false alarms
  from the heuristic score).

`.format()` controls the number of decimal places displayed in each numeric
column without altering the underlying data.

---

## End-to-end EDA narrative

The nine sections are structured to answer progressively deeper questions:

| Section | Question answered |
| --- | --- |
| 1 — Overview | What is in the dataset? Are there any quality issues? |
| 2 — Target | How severe is the class imbalance? Which groups fail most? |
| 3 — Distributions | Which numeric features separate failed from operational wells? |
| 4 — Correlation | Which features are most predictive? Are any redundant? |
| 5 — Categoricals | Which equipment types and well types fail at the highest rates? |
| 6 — Scaling | How much do scale deposits amplify failure risk? |
| 7 — Pressure | How does drawdown relate to failure? |
| 8 — Injection | Are injection wells a special risk population? |
| 9 — Summary | What do all of the above findings say in plain language? |

Each section builds on the previous: the findings in Section 6 (scaling flag)
motivated including `scaling_flag` as a multiplicative factor in the
`failure_pressure_index` feature in the modelling notebook. The multi-collinearity
finding in Section 4 motivated replacing the three pressure columns with the
single `drawdown` derived feature.
