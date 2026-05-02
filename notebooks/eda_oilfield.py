import marimo

__generated_with = "0.23.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Oilfield Equipment — Exploratory Data Analysis

    **Dataset:** `ML_dataset_full_field_v2.xlsx` &nbsp;|&nbsp;
    **Target:** `status` — 1 = Operational · 0 = Failed

    **796 wells** across production and injection well types; five equipment categories:
    ESP, Flowing, Gas Lift, Water Injection, and Injection Pump.

    This notebook explores feature distributions, failure patterns, and key risk
    drivers to inform downstream modelling for equipment failure prediction.

    > **Leaky columns** — `failures_last_year` and `downtime_days` — are flagged
    > throughout and must never be used as model inputs.
    """)
    return


@app.cell
def _():
    # '%matplotlib inline' command supported automatically in marimo

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
    return Path, gaussian_kde, go, make_subplots, math, np, pd, plt, px, sns


@app.cell
def _():
    # color constants
    COLOR_OPERATIONAL = "#1D9E75"   # teal
    COLOR_FAILED      = "#D85A30"   # coral

    # px.* constructors accept template/height but NOT margin/font (layout-level keys)
    PLOTLY_CONSTRUCTOR_KW = dict(template="plotly_white", height=450)
    PLOTLY_LAYOUT_KW      = dict(margin=dict(l=50, r=20, t=50, b=40), font_size=12)
    return (
        COLOR_FAILED,
        COLOR_OPERATIONAL,
        PLOTLY_CONSTRUCTOR_KW,
        PLOTLY_LAYOUT_KW,
    )


@app.cell
def _(Path, pd):
    # load dataset
    data_path  = Path("..") / "data" / "ML_dataset_full_field_v2.xlsx"
    wells_df   = pd.read_excel(data_path)

    wells_df.head()
    return (wells_df,)


@app.cell
def _(wells_df):
    # feature lists
    LEAKY_COLUMNS = ["failures_last_year", "downtime_days"]
    SAFE_NUMERIC_FEATURES = [
        column for column in wells_df.select_dtypes(include="number").columns
        if column not in LEAKY_COLUMNS + ["status"]
    ]

    print(f"Shape                : {wells_df.shape}")
    print(f"Leaky columns        : {LEAKY_COLUMNS}")
    print(f"Safe numeric features: {SAFE_NUMERIC_FEATURES}")
    return LEAKY_COLUMNS, SAFE_NUMERIC_FEATURES


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Data Overview
    """)
    return


@app.cell
def _(LEAKY_COLUMNS, display, pd, wells_df):
    # Column inventory with dtype, null count, unique count, and leakage flag
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
    return


@app.cell
def _(LEAKY_COLUMNS, display, pd, wells_df):
    # Quick dataset statistics
    total_wells      = wells_df.shape[0]
    operational_rate = wells_df["status"].mean() * 100
    failure_rate     = 100 - operational_rate

    dataset_stats = pd.DataFrame({
        "Metric": [
            "Total wells", "Operational wells", "Failed wells",
            "Operational rate", "Failure rate",
            "Equipment types", "Well types",
        ],
        "Value": [
            total_wells,
            int(wells_df["status"].sum()),
            int((wells_df["status"] == 0).sum()),
            f"{operational_rate:.1f}%",
            f"{failure_rate:.1f}%",
            wells_df["equipment_type"].nunique(),
            wells_df["well_type"].nunique(),
        ],
    })
    display(dataset_stats.style.hide(axis="index"))

    print()
    print("⚠️  LEAKY columns — never use in any predictive model:")
    for leaky_column in LEAKY_COLUMNS:
        print(f"   • {leaky_column!r} — recorded after a failure has occurred (post-failure observation)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Target Distribution
    """)
    return


@app.cell
def _(
    COLOR_FAILED,
    COLOR_OPERATIONAL,
    PLOTLY_CONSTRUCTOR_KW,
    PLOTLY_LAYOUT_KW,
    px,
    wells_df,
):
    # Pie chart — overall class split
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
    return


@app.cell
def _(COLOR_FAILED, COLOR_OPERATIONAL, plt, sns, wells_df):
    # Horizontal bar charts — status breakdown by well type and equipment type
    fig_status_bars, (ax_well_type, ax_equipment_type) = plt.subplots(1, 2, figsize=(13, 5))
    for _group_column, _subplot_ax, _chart_title, _y_axis_label in [('well_type', ax_well_type, 'Status by Well Type', 'Well Type'), ('equipment_type', ax_equipment_type, 'Status by Equipment Type', 'Equipment Type')]:
        group_counts = wells_df.groupby([_group_column, 'status']).size().reset_index(name='count').pivot(index=_group_column, columns='status', values='count').fillna(0).rename(columns={0: 'Failed', 1: 'Operational'})
        group_counts.plot(kind='barh', ax=_subplot_ax, color=[COLOR_FAILED, COLOR_OPERATIONAL])
        for bar_patch in _subplot_ax.patches:
            bar_width = bar_patch.get_width()
            if bar_width > 0:
                _subplot_ax.text(bar_width + 1, bar_patch.get_y() + bar_patch.get_height() / 2, str(int(bar_width)), va='center', fontsize=9)
        _subplot_ax.set_title(_chart_title, fontsize=13)
        _subplot_ax.set_xlabel('Count')
        _subplot_ax.set_ylabel(_y_axis_label)
        sns.despine(ax=_subplot_ax)
    plt.suptitle('The dataset is moderately imbalanced (67% operational, 33% failed).', fontsize=11, y=1.03, style='italic')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Numeric Feature Distributions
    """)
    return


@app.cell
def _(
    COLOR_FAILED,
    COLOR_OPERATIONAL,
    SAFE_NUMERIC_FEATURES,
    go,
    make_subplots,
    math,
    wells_df,
):
    # Violin plots with embedded box — one subplot per feature, 3 columns
    total_features = len(SAFE_NUMERIC_FEATURES)
    violin_cols_per_row = 3
    violin_row_count = math.ceil(total_features / violin_cols_per_row)
    fig_violin_grid = make_subplots(rows=violin_row_count, cols=violin_cols_per_row, subplot_titles=SAFE_NUMERIC_FEATURES, horizontal_spacing=0.08, vertical_spacing=0.14)
    for feature_index, _feature_name in enumerate(SAFE_NUMERIC_FEATURES):
        subplot_row = feature_index // violin_cols_per_row + 1
        subplot_col = feature_index % violin_cols_per_row + 1
        for _status_class, violin_color, status_label in [(1, COLOR_OPERATIONAL, 'Operational'), (0, COLOR_FAILED, 'Failed')]:
            feature_values = wells_df.loc[wells_df['status'] == _status_class, _feature_name]
            fig_violin_grid.add_trace(go.Violin(y=feature_values, name=status_label, box_visible=True, meanline_visible=True, line_color=violin_color, fillcolor=violin_color, opacity=0.55, showlegend=feature_index == 0, legendgroup=status_label), row=subplot_row, col=subplot_col)
    fig_violin_grid.update_layout(title_text='Numeric Feature Distributions by Status (violin + box)', height=280 * violin_row_count, template='plotly_white', violinmode='group', margin=dict(l=40, r=20, t=80, b=40), font_size=11)
    fig_violin_grid.show()
    return


@app.cell
def _(
    COLOR_FAILED,
    COLOR_OPERATIONAL,
    SAFE_NUMERIC_FEATURES,
    plt,
    sns,
    wells_df,
):
    # Pairplot — top 4 features by |Pearson r| with status
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
    pairplot_grid.figure.suptitle(
        "Pairplot — Top 4 Features by |Correlation with Status|",
        y=1.02, fontsize=13,
    )

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    COLOR_FAILED,
    COLOR_OPERATIONAL,
    SAFE_NUMERIC_FEATURES,
    gaussian_kde,
    math,
    np,
    plt,
    sns,
    wells_df,
):
    # Histogram + KDE grid — all numeric features (density-normalised)
    histogram_cols_per_row = 3
    histogram_row_count = math.ceil(len(SAFE_NUMERIC_FEATURES) / histogram_cols_per_row)
    fig_histogram_grid, histogram_axes = plt.subplots(histogram_row_count, histogram_cols_per_row, figsize=(14, histogram_row_count * 3.5))
    histogram_axes = histogram_axes.flatten()
    for feature_idx, _feature_name in enumerate(SAFE_NUMERIC_FEATURES):
        _subplot_ax = histogram_axes[feature_idx]
        for _status_class, class_color, class_label in [(1, COLOR_OPERATIONAL, 'Operational'), (0, COLOR_FAILED, 'Failed')]:
            feature_data = wells_df.loc[wells_df['status'] == _status_class, _feature_name].dropna()
            _subplot_ax.hist(feature_data, bins=25, color=class_color, alpha=0.55, label=class_label, density=True)
            if len(feature_data) > 1 and feature_data.std() > 1e-10:
                kernel_density_estimator = gaussian_kde(feature_data)
                kde_x_values = np.linspace(feature_data.min(), feature_data.max(), 200)
                _subplot_ax.plot(kde_x_values, kernel_density_estimator(kde_x_values), color=class_color, lw=1.8)
        _subplot_ax.set_title(_feature_name, fontsize=11)
        if feature_idx == 0:
            _subplot_ax.legend(fontsize=9)
        sns.despine(ax=_subplot_ax)
    for empty_panel_idx in range(feature_idx + 1, len(histogram_axes)):  # Guard: gaussian_kde fails on constant data (e.g. scaling_flag filtered by class)
        histogram_axes[empty_panel_idx].set_visible(False)
    fig_histogram_grid.suptitle('Numeric Feature Distributions by Status (density + KDE)', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Correlation Analysis
    """)
    return


@app.cell
def _(LEAKY_COLUMNS, np, plt, sns, wells_df):
    # Full correlation heatmap — lower triangle only
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
    ax_heatmap.set_title("Full Correlation Matrix (lower triangle)", fontsize=13)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    COLOR_FAILED,
    COLOR_OPERATIONAL,
    PLOTLY_CONSTRUCTOR_KW,
    PLOTLY_LAYOUT_KW,
    SAFE_NUMERIC_FEATURES,
    px,
    wells_df,
):
    # Feature–status correlation bar chart (signed, sorted by |r|)
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
        title="Feature Correlation with Status (Pearson r, sorted by |r|)",
        labels={"Correlation": "Pearson r with status"},
        **PLOTLY_CONSTRUCTOR_KW,
    )
    fig_correlation_bar.update_layout(
        yaxis={"categoryorder": "total ascending"},
        **PLOTLY_LAYOUT_KW,
    )
    fig_correlation_bar.show()

    print()
    print("Multi-collinear pressure columns (Pearson r between them):")
    pressure_column_names = [
        "reservoir_pressure_MPa", "pressure_bottomhole_MPa", "pressure_wellhead_MPa"
    ]
    print(wells_df[pressure_column_names].corr().round(3).to_string())
    print()
    print("→ Consider using drawdown (reservoir_P − bottomhole_P) as a single derived feature")
    print("  to eliminate redundancy before training linear or regularised models.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Categorical Breakdowns
    """)
    return


@app.cell
def _(COLOR_FAILED, PLOTLY_CONSTRUCTOR_KW, PLOTLY_LAYOUT_KW, px, wells_df):
    # Failure rate bars — equipment type and well type
    for _group_column, _chart_title, _y_axis_label in [('equipment_type', 'Failure Rate (%) by Equipment Type', 'Equipment Type'), ('well_type', 'Failure Rate (%) by Well Type', 'Well Type')]:
        failure_rate_table = wells_df.groupby(_group_column)['status'].agg(failed_count=lambda x: (x == 0).sum(), total_count='count').assign(failure_rate_pct=lambda d: d['failed_count'] / d['total_count'] * 100).sort_values('failure_rate_pct', ascending=False).reset_index()
        failure_rate_fig = px.bar(failure_rate_table, x='failure_rate_pct', y=_group_column, orientation='h', title=_chart_title, labels={'failure_rate_pct': 'Failure Rate (%)', _group_column: _y_axis_label}, text=failure_rate_table['failure_rate_pct'].round(1).astype(str) + '%', color_discrete_sequence=[COLOR_FAILED], **PLOTLY_CONSTRUCTOR_KW)
        failure_rate_fig.update_traces(textposition='outside')
        failure_rate_fig.update_layout(xaxis_range=[0, failure_rate_table['failure_rate_pct'].max() * 1.25], **PLOTLY_LAYOUT_KW)
        failure_rate_fig.show()
    return


@app.cell
def _(plt, sns, wells_df):
    # Pivot heatmap — failure rate by equipment type × well type
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
    ax_pivot_heatmap.set_title("Failure Rate (%) — Equipment Type × Well Type", fontsize=13)
    ax_pivot_heatmap.set_xlabel("Well Type")
    ax_pivot_heatmap.set_ylabel("Equipment Type")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Scaling Flag Deep-Dive
    """)
    return


@app.cell
def _(
    COLOR_FAILED,
    COLOR_OPERATIONAL,
    PLOTLY_CONSTRUCTOR_KW,
    PLOTLY_LAYOUT_KW,
    px,
    wells_df,
):
    # Stacked bar: fleet composition split by scaling flag
    scaling_status_counts = (
        wells_df.groupby(["scaling_flag", "status"])
        .size().reset_index(name="well_count")
    )
    scaling_status_counts["Status"] = scaling_status_counts["status"].map(
        {1: "Operational", 0: "Failed"}
    )
    scaling_status_counts["Scaling"] = scaling_status_counts["scaling_flag"].map(
        {0: "No Scaling  (flag = 0)", 1: "Scaling Present  (flag = 1)"}
    )

    fig_scaling_bar = px.bar(
        scaling_status_counts,
        x="Scaling", y="well_count", color="Status",
        barmode="stack",
        title="Well Count by Scaling Flag and Status",
        labels={"well_count": "Well Count", "Scaling": "Scaling Flag"},
        color_discrete_map={"Operational": COLOR_OPERATIONAL, "Failed": COLOR_FAILED},
        **PLOTLY_CONSTRUCTOR_KW,
    )
    fig_scaling_bar.update_layout(**PLOTLY_LAYOUT_KW)
    fig_scaling_bar.show()

    # Compute failure rates per flag value
    scaling_failure_rates = (
        wells_df.groupby("scaling_flag")["status"]
        .agg(failed_count=lambda x: (x == 0).sum(), total_count="count")
        .assign(failure_rate_pct=lambda d: (d["failed_count"] / d["total_count"] * 100).round(1))
    )
    no_scaling_failure_rate      = scaling_failure_rates.loc[0, "failure_rate_pct"]
    with_scaling_failure_rate    = scaling_failure_rates.loc[1, "failure_rate_pct"]
    scaling_failure_multiplier   = (
        round(with_scaling_failure_rate / no_scaling_failure_rate, 1)
        if no_scaling_failure_rate > 0 else "N/A"
    )

    print("Failure rates by scaling flag:")
    print(f"  No scaling      (flag = 0) : {no_scaling_failure_rate:>5.1f}%")
    print(f"  Scaling present (flag = 1) : {with_scaling_failure_rate:>5.1f}%")
    print()
    print(f"  → Scaling wells fail at {scaling_failure_multiplier}× the rate of clean wells.")
    print(f"    scaling_flag is one of the strongest binary risk indicators in the dataset.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Pressure & Production Analysis
    """)
    return


@app.cell
def _(display, wells_df):
    # Derived feature: drawdown = reservoir_P − bottomhole_P
    # High drawdown → aggressive depletion → mechanical stress on pumps and valves
    pressure_analysis_df = wells_df.copy()
    pressure_analysis_df["drawdown_MPa"] = (
        pressure_analysis_df["reservoir_pressure_MPa"]
        - pressure_analysis_df["pressure_bottomhole_MPa"]
    )
    pressure_analysis_df["Status"] = pressure_analysis_df["status"].map(
        {1: "Operational", 0: "Failed"}
    )

    print("Drawdown statistics by status:")
    display(
        pressure_analysis_df.groupby("Status")["drawdown_MPa"]
        .describe().round(2)
        .style.background_gradient(cmap="RdYlGn_r", axis=None)
    )
    return (pressure_analysis_df,)


@app.cell
def _(
    COLOR_FAILED,
    COLOR_OPERATIONAL,
    PLOTLY_CONSTRUCTOR_KW,
    PLOTLY_LAYOUT_KW,
    pressure_analysis_df,
    px,
):
    # Scatter: reservoir vs bottomhole pressure, bubble size = oil rate
    fig_pressure_bubble_scatter = px.scatter(
        pressure_analysis_df,
        x="reservoir_pressure_MPa",
        y="pressure_bottomhole_MPa",
        color="Status",
        size="oil_rate_tpd",
        size_max=18,
        hover_data=["well_id", "equipment_type"],
        color_discrete_map={"Operational": COLOR_OPERATIONAL, "Failed": COLOR_FAILED},
        title="Reservoir vs Bottomhole Pressure  (bubble size = Oil Rate tpd)",
        labels={
            "reservoir_pressure_MPa":  "Reservoir Pressure (MPa)",
            "pressure_bottomhole_MPa": "Bottomhole Pressure (MPa)",
        },
        opacity=0.7,
        **PLOTLY_CONSTRUCTOR_KW,
    )
    fig_pressure_bubble_scatter.update_layout(**PLOTLY_LAYOUT_KW)
    fig_pressure_bubble_scatter.show()
    return


@app.cell
def _(
    COLOR_FAILED,
    COLOR_OPERATIONAL,
    PLOTLY_CONSTRUCTOR_KW,
    PLOTLY_LAYOUT_KW,
    pressure_analysis_df,
    px,
):
    # Box plots: drawdown by status, faceted by well type
    fig_drawdown_boxplot = px.box(
        pressure_analysis_df,
        x="Status",
        y="drawdown_MPa",
        facet_col="well_type",
        color="Status",
        color_discrete_map={"Operational": COLOR_OPERATIONAL, "Failed": COLOR_FAILED},
        title="Drawdown (Reservoir − Bottomhole Pressure) by Status and Well Type",
        labels={"drawdown_MPa": "Drawdown (MPa)"},
        **PLOTLY_CONSTRUCTOR_KW,
    )
    fig_drawdown_boxplot.update_layout(showlegend=False, **PLOTLY_LAYOUT_KW)
    fig_drawdown_boxplot.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Injection Wells Analysis
    """)
    return


@app.cell
def _(
    COLOR_FAILED,
    COLOR_OPERATIONAL,
    PLOTLY_CONSTRUCTOR_KW,
    PLOTLY_LAYOUT_KW,
    px,
    wells_df,
):
    # Injection wells only — injection rate vs water cut
    injection_wells_df = wells_df[wells_df["well_type"] == "injection"].copy()
    injection_wells_df["Status"] = injection_wells_df["status"].map(
        {1: "Operational", 0: "Failed"}
    )

    fig_injection_scatter = px.scatter(
        injection_wells_df,
        x="injection_rate_m3_day",
        y="water_cut_pct",
        color="Status",
        color_discrete_map={"Operational": COLOR_OPERATIONAL, "Failed": COLOR_FAILED},
        title="Injection Rate vs Water Cut (Injection Wells Only)",
        labels={
            "injection_rate_m3_day": "Injection Rate (m³/day)",
            "water_cut_pct": "Water Cut (%)",
        },
        opacity=0.7,
        **PLOTLY_CONSTRUCTOR_KW,
    )

    fig_injection_scatter.update_layout(**PLOTLY_LAYOUT_KW)
    fig_injection_scatter.show()
    return (injection_wells_df,)


@app.cell
def _(display, injection_wells_df):
    # Equipment failure rates within injection wells
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
            "equipment_type":   "Equipment Type",
            "total_count":      "Total",
            "failed_count":     "Failed",
            "operational_count":"Operational",
            "failure_rate_pct": "Failure Rate (%)",
        })
        .reset_index(drop=True)
    )

    print("Failure rates for injection wells by equipment type:")

    display(
        injection_equipment_rates.style
        .background_gradient(subset=["Failure Rate (%)"], cmap="Reds")
        .format({"Failure Rate (%)": "{:.1f}%"})
        .hide(axis="index")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9 · Key Findings Summary
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Top 5 Findings

    1. **Scaling is the strongest binary failure driver** — Wells with `scaling_flag = 1`
       fail at roughly twice the rate of clean wells. Scale deposits restrict flow and
       increase friction, accelerating mechanical wear across all equipment types.

    2. **Injection Pump equipment has the highest failure rate** — Among all equipment
       categories, Injection Pumps and Gas Lift systems show disproportionately high
       failure rates, particularly in injection-well configurations.

    3. **High drawdown strongly correlates with failure** — Wells where reservoir
       pressure greatly exceeds bottomhole pressure experience higher mechanical stress;
       drawdown management is a key operational lever for extending equipment life.

    4. **Injection wells fail more than production wells** — The combination of high
       volumes, elevated water cut, and corrosive conditions creates a more demanding
       environment for downhole equipment in injection operations.

    5. **Pressure features are multi-collinear** — `reservoir_pressure_MPa`,
       `pressure_bottomhole_MPa`, and `pressure_wellhead_MPa` are strongly
       inter-correlated. Use PCA or the derived `drawdown_MPa` feature before
       training linear or regularised models to avoid redundant signal.
    """)
    return


@app.cell
def _(display, wells_df):
    # Heuristic risk score (domain-knowledge weighted proxy — NOT a trained model)
    #
    #   risk = 0.40 × scaling_flag
    #        + 0.30 × (1 − BHP / reservoir_P)   ← normalised drawdown
    #        + 0.30 × water_cut_pct / 100
    #
    # clip(lower=0.01) prevents division by zero on reservoir_P
    # clip(upper=1.0)  keeps the ratio in [0,1] for injection wells where BHP > reservoir_P

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

    top_10_risk_wells = (
        risk_scoring_df.sort_values("risk_score", ascending=False)
        .head(10)[[
            "well_id", "well_type", "equipment_type",
            "scaling_flag", "reservoir_pressure_MPa",
            "pressure_bottomhole_MPa", "water_cut_pct",
            "risk_score", "status",
        ]]
        .reset_index(drop=True)
    )
    top_10_risk_wells["actual_status"] = top_10_risk_wells["status"].map(
        {1: "Operational", 0: "Failed"}
    )
    top_10_risk_wells = top_10_risk_wells.drop(columns=["status"])

    print("Top 10 Highest-Risk Wells — Heuristic Risk Proxy Score")
    print("(This is a domain-knowledge ranking, NOT a model prediction)")

    print()

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
    return


if __name__ == "__main__":
    app.run()
