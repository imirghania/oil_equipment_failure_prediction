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
    # Oilfield Well Failure — Probability Prediction Model

    **Objective:** Train and calibrate machine-learning models that output a
    well-calibrated *probability* of equipment failure for each well.

    **Target variable:** `failure = 1 − status` (1 = failed, 0 = operational)

    **Dropped features (leaky / identifiers):**
    - `well_id` — identifier, no predictive signal
    - `downtime_days` — post-failure observation, leaky
    - `failures_last_year` — post-failure observation, leaky

    **Approach:**
    1. Optuna hyperparameter search (50 trials each) for LightGBM and XGBoost
    2. Calibrate both models with `CalibratedClassifierCV(method="isotonic")`
    3. Compare calibrated LightGBM / XGBoost / Logistic Regression via Brier score,
       log-loss, AUC, and calibration curves
    4. Score the full dataset and assign risk tiers
    """)
    return


@app.cell
def _():
    # requires: jupyterlab pandas openpyxl numpy lightgbm xgboost optuna
    #            scikit-learn matplotlib seaborn plotly nbconvert
    # '%matplotlib inline' command supported automatically in marimo

    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    import optuna

    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.metrics import (
        brier_score_loss, log_loss, roc_auc_score, f1_score,
        confusion_matrix, ConfusionMatrixDisplay,
    )

    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sns.set_theme(style="whitegrid", palette="muted")
    return (
        CalibratedClassifierCV,
        ConfusionMatrixDisplay,
        LGBMClassifier,
        LabelEncoder,
        LogisticRegression,
        Pipeline,
        StandardScaler,
        StratifiedKFold,
        XGBClassifier,
        brier_score_loss,
        calibration_curve,
        cross_val_predict,
        cross_val_score,
        f1_score,
        go,
        log_loss,
        optuna,
        pd,
        plt,
        px,
        roc_auc_score,
    )


@app.cell
def _():
    # Display constants
    PALETTE = {"0": "#1D9E75", "1": "#D85A30"}
    COLORS  = {"lgbm": "#1D9E75", "xgb": "#378ADD", "logreg": "#BA7517"}
    PLOTLY_KW   = dict(template="plotly_white", height=450,
                   margin=dict(l=50, r=20, t=50, b=40), font_size=12)
    return COLORS, PLOTLY_KW


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Data Preparation
    """)
    return


@app.cell
def _(pd):
    from pathlib import Path

    # Load data
    path = Path("..") / "data" / "ML_dataset_full_field_v2.xlsx"
    df = pd.read_excel(path)
    df.head()
    return (df,)


@app.cell
def _(LabelEncoder, df):
    # Target
    df["failure"] = 1 - df["status"]   # flip: 1 = failed

    # Feature engineering
    df["drawdown"] = df["reservoir_pressure_MPa"] - df["pressure_bottomhole_MPa"]
    df["failure_pressure_index"] = df["drawdown"] * df["scaling_flag"]

    # Encode categoricals
    le_equip = LabelEncoder()
    le_well  = LabelEncoder()
    df["equipment_type_enc"] = le_equip.fit_transform(df["equipment_type"])
    df["well_type_enc"]      = le_well.fit_transform(df["well_type"])

    # Drop identifiers, leaky columns, original target
    COL_TO_DROP = ["well_id", "status", "downtime_days", "failures_last_year",
            "equipment_type", "well_type", "failure"]
    X = df.drop(columns=COL_TO_DROP)
    y = df["failure"]

    print("Feature matrix shape:", X.shape)
    print("Features:", X.columns.tolist())

    print('\n')

    print("Class balance:")
    balance = y.value_counts().rename({0: "operational", 1: "failed"})
    for label, n in balance.items():
        print(f"  {label}: {n:>4d}  ({n/len(y)*100:.1f}%)")

    print('\n')
    X.head()
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Optuna Search — LightGBM
    """)
    return


@app.cell
def _(LGBMClassifier, StratifiedKFold, X, cross_val_score, y):
    # Optuna objective — LightGBM
    def objective_lgbm(trial):
        params = dict(
            num_leaves        = trial.suggest_int("num_leaves", 20, 150),
            max_depth         = trial.suggest_int("max_depth", 3, 8),
            learning_rate     = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            n_estimators      = trial.suggest_int("n_estimators", 50, 400),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha         = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            min_child_samples = trial.suggest_int("min_child_samples", 5, 50),
            random_state=42, verbose=-1,
        )
        model = LGBMClassifier(**params)
        cv    = StratifiedKFold(5, shuffle=True, random_state=42)
        aucs  = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return -aucs.mean()   # minimize negative AUC

    return (objective_lgbm,)


@app.cell
def _(objective_lgbm, optuna):
    # Run LightGBM Optuna study — 50 trials
    study_lgbm = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study_lgbm.optimize(objective_lgbm, n_trials=50, show_progress_bar=True)
    best_params_lgbm = study_lgbm.best_params
    best_auc_lgbm = -study_lgbm.best_value
    print(f'LightGBM best AUC (5-fold CV): {best_auc_lgbm:.4f}')
    print('Best params:')
    for _k, _v in best_params_lgbm.items():
        print(f'  {_k}: {_v}')
    return best_params_lgbm, study_lgbm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Optuna Search — XGBoost
    """)
    return


@app.cell
def _(StratifiedKFold, X, XGBClassifier, cross_val_score, y):
    # Optuna objective — XGBoost
    def objective_xgb(trial):
        params = dict(
            max_depth        = trial.suggest_int("max_depth", 3, 8),
            learning_rate    = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            n_estimators     = trial.suggest_int("n_estimators", 50, 400),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha        = trial.suggest_float("reg_alpha", 0.0, 10.0),
            reg_lambda       = trial.suggest_float("reg_lambda", 0.0, 10.0),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
            gamma            = trial.suggest_float("gamma", 0.0, 5.0),
            eval_metric="logloss", verbosity=0, random_state=42,
        )
        model = XGBClassifier(**params)
        cv    = StratifiedKFold(5, shuffle=True, random_state=42)
        aucs  = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return -aucs.mean()

    return (objective_xgb,)


@app.cell
def _(objective_xgb, optuna):
    # Run XGBoost Optuna study — 50 trials
    study_xgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study_xgb.optimize(objective_xgb, n_trials=50, show_progress_bar=True)
    best_params_xgb = study_xgb.best_params
    best_auc_xgb = -study_xgb.best_value
    print(f'XGBoost best AUC (5-fold CV): {best_auc_xgb:.4f}')
    print('Best params:')
    for _k, _v in best_params_xgb.items():
        print(f'  {_k}: {_v}')
    return best_params_xgb, study_xgb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Model Training & Calibration
    """)
    return


@app.cell
def _(
    CalibratedClassifierCV,
    LGBMClassifier,
    LogisticRegression,
    Pipeline,
    StandardScaler,
    StratifiedKFold,
    X,
    XGBClassifier,
    best_params_lgbm,
    best_params_xgb,
    cross_val_predict,
    y,
):
    # Instantiate base models with tuned params
    lgbm_base = LGBMClassifier(**best_params_lgbm, random_state=42, verbose=-1)
    xgb_base  = XGBClassifier(**best_params_xgb,  random_state=42,
                               eval_metric="logloss", verbosity=0)
    logreg    = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=2000, C=0.5, random_state=42)),
    ])

    # Wrap with isotonic calibration
    lgbm_cal = CalibratedClassifierCV(lgbm_base, method="isotonic", cv=5)
    xgb_cal  = CalibratedClassifierCV(xgb_base,  method="isotonic", cv=5)

    # Out-of-fold probability estimates
    cv5 = StratifiedKFold(5, shuffle=True, random_state=42)

    print("Computing OOF probabilities...")
    probs_lgbm = cross_val_predict(lgbm_cal, X, y, cv=cv5, method="predict_proba")[:, 1]
    print("  LightGBM  done")
    probs_xgb  = cross_val_predict(xgb_cal,  X, y, cv=cv5, method="predict_proba")[:, 1]
    print("  XGBoost   done")
    probs_lr   = cross_val_predict(logreg,   X, y, cv=cv5, method="predict_proba")[:, 1]
    print("  LogReg    done")

    probs = {"lgbm": probs_lgbm, "xgb": probs_xgb, "logreg": probs_lr}
    print("\nOOF probabilities computed.")
    return lgbm_base, lgbm_cal, probs, xgb_base


@app.cell
def _(brier_score_loss, f1_score, log_loss, pd, probs, roc_auc_score, y):
    # Build metrics comparison table
    rows = []
    for _name, _p in probs.items():
        pred = (_p >= 0.5).astype(int)
        rows.append({'Model': _name.upper(), 'AUC': round(roc_auc_score(y, _p), 4), 'F1': round(f1_score(y, pred), 4), 'Brier': round(brier_score_loss(y, _p), 4), 'Log-loss': round(log_loss(y, _p), 4)})
    metrics_df = pd.DataFrame(rows).set_index('Model')
    # Highlight best values
    metrics_df.style.highlight_max(subset=['AUC', 'F1'], color='lightgreen').highlight_min(subset=['Brier', 'Log-loss'], color='lightgreen').format('{:.4f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Calibration Curves
    """)
    return


@app.cell
def _(COLORS, PLOTLY_KW, brier_score_loss, calibration_curve, go, probs, y):
    # Calibration curves for all 3 models
    fig_cal = go.Figure()
    for _name, _p in probs.items():
        frac_pos, mean_pred = calibration_curve(y, _p, n_bins=10, strategy='uniform')
        brier = brier_score_loss(y, _p)
        fig_cal.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode='lines+markers', name=f'{_name.upper()}  (Brier={brier:.4f})', line=dict(color=COLORS[_name], width=2), marker=dict(size=7)))
    fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfect calibration', line=dict(color='gray', dash='dash', width=1.5)))
    fig_cal.update_layout(title='Calibration curve — predicted vs actual failure rate', xaxis_title='Mean predicted probability', yaxis_title='Fraction of positives', **PLOTLY_KW)
    # Perfect calibration diagonal
    fig_cal.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. ROC Curves
    """)
    return


@app.cell
def _(COLORS, PLOTLY_KW, go, probs, roc_auc_score, y):
    # ROC curves — OOF probabilities
    from sklearn.metrics import roc_curve
    fig_roc = go.Figure()
    for _name, _p in probs.items():
        fpr, tpr, _ = roc_curve(y, _p)
        auc = roc_auc_score(y, _p)
        kw = dict(color=COLORS[_name], width=2)
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{_name.upper()}  AUC={auc:.4f}', line=kw))
        if _name == 'lgbm':
            fig_roc.add_trace(go.Scatter(x=list(fpr) + [1, 0], y=list(tpr) + [0, 0], fill='toself', fillcolor='rgba(29,158,117,0.10)', line=dict(width=0), showlegend=False))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(color='gray', dash='dash', width=1.5)))
    fig_roc.update_layout(title='ROC curves — out-of-fold predictions', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', **PLOTLY_KW)
    # Chance line
    fig_roc.show()  # fill area under best model
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Confusion Matrix
    """)
    return


@app.cell
def _(ConfusionMatrixDisplay, f1_score, plt, probs, roc_auc_score, y):
    from sklearn.metrics import precision_score, recall_score

    # Use LightGBM (best model) OOF probs
    best_name  = max(probs, key=lambda n: roc_auc_score(y, probs[n]))
    best_probs = probs[best_name]
    preds = (best_probs >= 0.5).astype(int)

    print(f"Best model: {best_name.upper()}")

    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y, preds, 
                                            ax=ax_cm, 
                                            cmap="Blues",
                                            display_labels=["Operational", "Failed"],
                                           )
    ax_cm.set_title(f"Confusion Matrix — {best_name.upper()} (OOF, thresh=0.5)")
    plt.tight_layout()
    plt.show()

    print(f"\nPrecision : {precision_score(y, preds):.4f}")
    print(f"Recall    : {recall_score(y, preds):.4f}")
    print(f"F1        : {f1_score(y, preds):.4f}")
    print(f"Support   : failed={int(y.sum())}  operational={int((y==0).sum())}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Feature Importance
    """)
    return


@app.cell
def _(X, lgbm_base, xgb_base, y):
    # Fit models on full dataset for importance extraction
    lgbm_base.fit(X, y)
    xgb_base.fit(X, y)
    return


@app.cell
def _(go, pd):
    def importance_fig(importances, feature_names, title, shades):
        imp_df = (pd.DataFrame({"feature": feature_names, "importance": importances})
                    .sort_values("importance", ascending=False)
                    .reset_index(drop=True))
        # Color by quartile
        q = pd.qcut(imp_df["importance"], 4, labels=False, duplicates="drop")
        imp_df["color"] = q.map(lambda i: shades[int(i)] if pd.notna(i) else shades[0])
        fig = go.Figure(go.Bar(
            x=imp_df["importance"],
            y=imp_df["feature"],
            orientation="h",
            marker_color=imp_df["color"].tolist(),
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Importance (gain)",
            yaxis=dict(autorange="reversed"),
            template="plotly_white", height=400,
            margin=dict(l=50, r=20, t=50, b=40), font_size=12,
        )
        return fig

    return (importance_fig,)


@app.cell
def _(X, importance_fig, lgbm_base, xgb_base):
    # LightGBM importance
    lgbm_imp = lgbm_base.feature_importances_
    lgbm_shades = ["#9FE1CB", "#5DCAA5", "#1D9E75", "#085041"]
    fig_imp_lgbm = importance_fig(lgbm_imp, X.columns, "LightGBM — Feature Importance (gain)", lgbm_shades)
    fig_imp_lgbm.show()

    # XGBoost importance
    xgb_imp = xgb_base.feature_importances_
    xgb_shades = ["#BDD7F5", "#7AAFEC", "#378ADD", "#154F8A"]
    fig_imp_xgb = importance_fig(xgb_imp, X.columns, "XGBoost — Feature Importance (gain)", xgb_shades)
    fig_imp_xgb.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Optuna Optimization History
    """)
    return


@app.cell
def _(go, pd, study_lgbm, study_xgb):
    # Optimization history — best AUC per trial
    fig_hist = go.Figure()
    for _study, _name, _color in [(study_lgbm, 'LightGBM', '#1D9E75'), (study_xgb, 'XGBoost', '#378ADD')]:
        values = [-t.value for t in _study.trials]
        best_so_far = pd.Series(values).cummax().tolist()
        fig_hist.add_trace(go.Scatter(x=list(range(1, len(best_so_far) + 1)), y=best_so_far, mode='lines', name=_name, line=dict(color=_color, width=2)))
    fig_hist.update_layout(title='Optuna optimization history — best AUC per trial', xaxis_title='Trial', yaxis_title='Best AUC (running max)', template='plotly_white', height=380, margin=dict(l=50, r=20, t=50, b=40), font_size=12)
    fig_hist.show()  # convert back to AUC
    return


@app.cell
def _(go, study_lgbm, study_xgb):
    # Top-5 hyperparameter importances per study
    from optuna.importance import get_param_importances
    fig_imp_hp = go.Figure()
    for _study, _name, _color in [(study_lgbm, 'LightGBM', '#1D9E75'), (study_xgb, 'XGBoost', '#378ADD')]:
        try:
            hp_imp = get_param_importances(_study)
            hp_top = dict(sorted(hp_imp.items(), key=lambda x: x[1], reverse=True)[:5])
            fig_imp_hp.add_trace(go.Bar(name=_name, x=list(hp_top.keys()), y=list(hp_top.values()), marker_color=_color))
        except Exception as e:
            print(f'Could not compute {_name} param importances: {e}')
    fig_imp_hp.update_layout(barmode='group', title='Top-5 hyperparameter importances (Optuna FAnova)', xaxis_title='Hyperparameter', yaxis_title='Importance', template='plotly_white', height=400, margin=dict(l=50, r=20, t=50, b=40), font_size=12)
    fig_imp_hp.show()  # Keep top 5
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Risk Scoring — Full Dataset
    """)
    return


@app.cell
def _(X, df, lgbm_cal, pd, y):
    # Fit best calibrated model on full training data
    lgbm_cal.fit(X, y)

    # Predict failure probability for every well
    df["failure_prob"] = lgbm_cal.predict_proba(X)[:, 1]
    df["risk_tier"] = pd.cut(
        df["failure_prob"],
        bins=[0, 0.20, 0.50, 0.80, 1.0],
        labels=["Low", "Medium", "High", "Critical"],
        include_lowest=True,
    )
    return


@app.cell
def _():
    TIER_COLORS = {
        "Low":      "#1D9E75",
        "Medium":   "#378ADD",
        "High":     "#BA7517",
        "Critical": "#E24B4A",
    }
    return (TIER_COLORS,)


@app.cell
def _(TIER_COLORS, df, px):
    # Histogram of predicted failure probabilities
    fig_prob_hist = px.histogram(
        df, x="failure_prob",
        color="risk_tier",
        color_discrete_map=TIER_COLORS,
        nbins=40,
        title="Distribution of predicted failure probability",
        labels={"failure_prob": "Predicted failure probability", "risk_tier": "Risk tier"},
        template="plotly_white", height=400,
    )

    fig_prob_hist.update_layout(margin=dict(l=50, r=20, t=50, b=40), font_size=12)
    fig_prob_hist.show()
    return


@app.cell
def _(df, px):
    # Scatter: reservoir pressure vs drawdown, coloured by failure_prob
    fig_risk_scatter = px.scatter(
        df,
        x="reservoir_pressure_MPa",
        y="drawdown",
        color="failure_prob",
        color_continuous_scale="RdYlGn_r",
        size_max=8,
        hover_data=["well_id", "well_type", "equipment_type", "risk_tier"],
        title="Reservoir Pressure vs Drawdown — coloured by failure probability",
        labels={
            "reservoir_pressure_MPa": "Reservoir Pressure (MPa)",
            "drawdown": "Drawdown (MPa)",
            "failure_prob": "Failure prob.",
        },
        template="plotly_white", height=460,
    )
    fig_risk_scatter.update_traces(marker=dict(size=8, opacity=0.75))
    fig_risk_scatter.update_layout(margin=dict(l=50, r=20, t=50, b=40), font_size=12)
    fig_risk_scatter.show()
    return


@app.cell
def _(df):
    # Risk tier summary
    print("\nRisk tier distribution:")
    tier_summary = (
        df.groupby("risk_tier", observed=True)
        .agg(count=("failure_prob", "size"),
             actual_failure_rate=("failure", "mean"))
        .assign(actual_failure_rate=lambda d: (d["actual_failure_rate"] * 100).round(1))
    )
    print(tier_summary.to_string())
    return


@app.cell
def _(df):
    # Top-20 highest-risk wells
    top20 = (
        df.sort_values("failure_prob", ascending=False)
        .head(20)[["well_id", "well_type", "equipment_type",
                   "scaling_flag", "failure_prob", "risk_tier"]]
        .reset_index(drop=True)
    )
    top20.style.background_gradient(subset=["failure_prob"], 
                                    cmap="Reds"
                                   ).format({"failure_prob": "{:.4f}"})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Model Summary
    """)
    return


@app.cell
def _(X, brier_score_loss, df, lgbm_base, pd, probs, roc_auc_score, y):
    # Build and print a structured deployment summary
    best_model = max(probs, key=lambda n: roc_auc_score(y, probs[n]))
    best_auc = roc_auc_score(y, probs[best_model])
    best_brier = brier_score_loss(y, probs[best_model])
    fi = pd.Series(lgbm_base.feature_importances_, index=X.columns)
    # Top-3 feature importances from LightGBM
    top3 = fi.sort_values(ascending=False).head(3)
    tier_fr = df.groupby('risk_tier', observed=True)['failure'].agg(['sum', 'count']).assign(rate=lambda d: (d['sum'] / d['count'] * 100).round(1))
    print('=' * 60)
    # Tier-level failure rates
    print('MODEL DEPLOYMENT SUMMARY')
    print('=' * 60)
    print(f'\nBest model     : {best_model.upper()} (calibrated, isotonic, 5-fold)')
    print(f'AUC            : {best_auc:.4f}')
    print(f'Brier score    : {best_brier:.4f}')
    print(f'\nTop-3 features :')
    for feat, imp in top3.items():
        print(f'  {feat:<35s}  gain={imp:.1f}')
    print('\nRisk tier actual failure rates:')
    for tier, row in tier_fr.iterrows():
        bar = '█' * int(row['rate'] / 5)
        print(f'  {str(tier):<10s}  {row['rate']:>5.1f}%  {bar}')
    print()
    print('Calibration quality:')
    brier_map = {n: brier_score_loss(y, _p) for n, _p in probs.items()}
    best_brier_name = min(brier_map, key=brier_map.get)
    print(f'  Best Brier score: {best_brier_name.upper()} = {brier_map[best_brier_name]:.4f}')
    print(f'  (Brier < 0.15 indicates well-calibrated probabilities)')
    print()
    print('Operational recommendation:')
    print("  • Flag 'High' and 'Critical' tier wells for preventive inspection")
    print('  • Prioritise wells with scaling_flag=1 and high drawdown')
    print('  • Re-score quarterly; retrain every 6 months with new failure data')
    print('=' * 60)
    return


if __name__ == "__main__":
    app.run()
