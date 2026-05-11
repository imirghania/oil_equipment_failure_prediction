from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

DATA_PATH = Path(__file__).parent / "data" / "ML_dataset_full_field_v2.xlsx"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "lgbm_calibrated.joblib"

BEST_PARAMS = dict(
    num_leaves=81,
    max_depth=5,
    learning_rate=0.0026908255930164736,
    n_estimators=91,
    subsample=0.6111336218174267,
    colsample_bytree=0.6121887020809305,
    reg_alpha=8.968161471295275,
    reg_lambda=1.5918532419259288,
    min_child_samples=21,
    random_state=42,
    verbose=-1,
)

FEATURE_COLS = [
    "oil_rate_tpd",
    "injection_rate_m3_day",
    "water_cut_pct",
    "pressure_wellhead_MPa",
    "pressure_bottomhole_MPa",
    "reservoir_pressure_MPa",
    "depth_m",
    "scaling_flag",
    "drawdown",
    "failure_pressure_index",
    "equipment_type_enc",
    "well_type_enc",
]


def main():
    df = pd.read_excel(DATA_PATH)

    df["failure"] = 1 - df["status"]

    df["drawdown"] = df["reservoir_pressure_MPa"] - df["pressure_bottomhole_MPa"]
    df["failure_pressure_index"] = df["drawdown"] * df["scaling_flag"]

    le_equip = LabelEncoder()
    le_well = LabelEncoder()
    df["equipment_type_enc"] = le_equip.fit_transform(df["equipment_type"])
    df["well_type_enc"] = le_well.fit_transform(df["well_type"])

    X = df[FEATURE_COLS]
    y = df["failure"]

    base = LGBMClassifier(**BEST_PARAMS)
    lgbm_cal = CalibratedClassifierCV(base, method="isotonic", cv=5)
    lgbm_cal.fit(X, y)

    train_probs = lgbm_cal.predict_proba(X)[:, 1]
    train_auc = roc_auc_score(y, train_probs)

    bundle = {
        "model": lgbm_cal,
        "le_equip": le_equip,
        "le_well": le_well,
        "feature_cols": FEATURE_COLS,
        "equip_classes": le_equip.classes_.tolist(),
        "well_classes": le_well.classes_.tolist(),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Training AUC (sanity check): {train_auc:.4f}")
    print(f"Equipment classes: {le_equip.classes_.tolist()}")
    print(f"Well classes: {le_well.classes_.tolist()}")
    print(f"Features: {FEATURE_COLS}")


if __name__ == "__main__":
    main()
