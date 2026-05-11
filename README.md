# Oilfield Equipment Failure Predictor

A machine-learning project that predicts the **probability of equipment failure** for oil and gas wells, covering exploratory data analysis, Optuna hyperparameter search, probability calibration, and a standalone PyQt6 desktop application for single-well and batch inference.

---

## Environment Setup

The project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
pip install uv
uv sync
```

---

## Converting Marimo Notebooks to JupyterLab

The notebooks in this repo are Marimo `.py` files. They must be converted to `.ipynb` before opening in JupyterLab.

```bash
uv run marimo export ipynb notebooks/eda_oilfield.py \
    --output notebooks/eda_oilfield.ipynb

uv run marimo export ipynb notebooks/model_oilfield.py \
    --output notebooks/model_oilfield.ipynb
```

Open the executed notebooks in JupyterLab:

```bash
uv run jupyter lab
```

---

## Saving the Model

The script `save_model.py` train and save the model bundle using the best parameters found from expirements:

```bash
uv run python save_model.py
```

This creates `models/lgbm_calibrated.joblib` (the directory is created automatically). The app raises `FileNotFoundError` on startup if this file is missing, so **always run `save_model.py` once after a fresh clone**.
