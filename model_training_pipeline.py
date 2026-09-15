"""
Model Training & Tuning Pipeline
RGB/HSV/LAB image features -> CIE L*a*b* prediction (spectrophotometer replacement)

Progression: Linear/PLS -> Random Forest -> XGBoost -> (optional) small ANN
Evaluation: RMSE, MAE, R^2 per channel + Delta E (CIE76 and CIEDE2000)

Install once:
    pip install scikit-learn xgboost pandas numpy scipy colormath --break-system-packages
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ---------------------------------------------------------------------
# 1. LOAD YOUR DATA
# ---------------------------------------------------------------------
# Expected CSV columns (from your feature-extraction step):
#   R_mean, G_mean, B_mean, H, S, V, L_img, a_img, b_img   -> INPUT features
#   L_star, a_star, b_star                                  -> TARGET (spectrophotometer ground truth)
#
parser = argparse.ArgumentParser(description="Train color prediction models from a CSV dataset.")
parser.add_argument(
    "dataset",
    nargs="?",
    type=Path,
    default=Path(__file__).with_name("fabric_color_dataset.csv"),
    help="Path to the training CSV (default: fabric_color_dataset.csv beside this script)",
)
args = parser.parse_args()

if not args.dataset.is_file():
    raise FileNotFoundError(
        f"Dataset not found: {args.dataset}\n"
        "Pass the CSV path when running the script, for example: "
        "python model_training_pipeline.py C:\\path\\to\\fabric_color_dataset.csv"
    )

df = pd.read_csv(args.dataset)

feature_cols = ["R_mean", "G_mean", "B_mean", "H", "S", "V", "L_img", "a_img", "b_img"]
target_cols = ["L_star", "a_star", "b_star"]

required_cols = feature_cols + target_cols
missing_cols = sorted(set(required_cols) - set(df.columns))
if missing_cols:
    raise ValueError(f"CSV is missing required columns: {', '.join(missing_cols)}")

if df[required_cols].isnull().any().any():
    raise ValueError("CSV contains missing values in the feature or target columns.")

X = df[feature_cols].values
y = df[target_cols].values  # shape: (n_samples, 3)

print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features -> {y.shape[1]} targets")

# ---------------------------------------------------------------------
# 2. CROSS-VALIDATION SETUP
# ---------------------------------------------------------------------
# With a small dataset (50-150 samples), k-fold CV gives a much more
# reliable performance estimate than one train/test split.
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)


def evaluate_model(model, X, y, name):
    """Run k-fold CV, return predictions and print metrics per target channel."""
    y_pred = cross_val_predict(model, X, y, cv=kf)

    print(f"\n=== {name} ===")
    for i, col in enumerate(target_cols):
        rmse = np.sqrt(mean_squared_error(y[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y[:, i], y_pred[:, i])
        r2 = r2_score(y[:, i], y_pred[:, i])
        print(f"  {col}: RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}")

    delta_e = compute_delta_e76(y, y_pred)
    print(f"  Delta E (CIE76): mean={delta_e.mean():.3f}  "
          f"%<=1: {np.mean(delta_e <= 1) * 100:.1f}%  "
          f"%<=2: {np.mean(delta_e <= 2) * 100:.1f}%  "
          f"%<=3: {np.mean(delta_e <= 3) * 100:.1f}%")
    return y_pred, delta_e


def compute_delta_e76(y_true, y_pred):
    """Simple Euclidean Delta E (CIE76) in L*a*b* space. Good enough for
    model comparison; swap in CIEDE2000 (colormath library) for your
    final thesis-quality reporting if you want the more perceptually
    accurate metric."""
    diff = y_true - y_pred
    return np.sqrt(np.sum(diff ** 2, axis=1))


# ---------------------------------------------------------------------
# 3. BASELINE MODELS
# ---------------------------------------------------------------------
# 3a. Linear Regression (simplest baseline)
lin_model = MultiOutputRegressor(LinearRegression())
evaluate_model(lin_model, X, y, "Linear Regression (baseline)")

# 3b. Partial Least Squares (handles multicollinear RGB/HSV/LAB features well)
pls_model = PLSRegression(n_components=min(5, X.shape[1]))
evaluate_model(pls_model, X, y, "PLS Regression (baseline)")

# ---------------------------------------------------------------------
# 4. INTERMEDIATE MODEL: RANDOM FOREST
# ---------------------------------------------------------------------
rf_model = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42)
)
evaluate_model(rf_model, X, y, "Random Forest")

# Feature importance (fit on full data just to inspect, not for evaluation)
rf_model.fit(X, y)
for i, col in enumerate(target_cols):
    importances = rf_model.estimators_[i].feature_importances_
    ranked = sorted(zip(feature_cols, importances), key=lambda t: -t[1])
    print(f"\nTop features for {col}: {ranked[:3]}")

# ---------------------------------------------------------------------
# 5. ADVANCED MODEL: XGBOOST (with hyperparameter tuning)
# ---------------------------------------------------------------------
xgb_base = XGBRegressor(random_state=42, objective="reg:squarederror")
xgb_model = MultiOutputRegressor(xgb_base)

# Small grid search example — expand this once you see which ranges look promising.
param_grid = {
    "estimator__n_estimators": [100, 300],
    "estimator__max_depth": [3, 5, 7],
    "estimator__learning_rate": [0.05, 0.1],
}

# NOTE: GridSearchCV needs a single scorer; for multi-output regression,
# a simple approach is to tune per-channel or use a custom scorer.
# Below: quick manual comparison instead of full GridSearchCV, to keep
# it simple for a first pass. Swap in GridSearchCV once your pipeline works.
best_params_list = [
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
    {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
]
for params in best_params_list:
    model = MultiOutputRegressor(XGBRegressor(random_state=42, **params))
    evaluate_model(model, X, y, f"XGBoost {params}")

# ---------------------------------------------------------------------
# 6. (OPTIONAL) SMALL ANN — only if you have 100+ samples
# ---------------------------------------------------------------------
# from sklearn.neural_network import MLPRegressor
# ann_model = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=42)
# evaluate_model(ann_model, X, y, "MLP (small ANN)")

# ---------------------------------------------------------------------
# 7. FINAL MODEL SELECTION
# ---------------------------------------------------------------------
# Pick whichever model gave the lowest mean Delta E and highest
# % of predictions within Delta E <= 2 across the k-fold CV above.
# Retrain that model on the FULL dataset for your final deployed model:
#
#   final_model = MultiOutputRegressor(XGBRegressor(random_state=42, **best_params))
#   final_model.fit(X, y)
#
# Save it for later use:
#   import joblib
#   joblib.dump(final_model, "final_color_model.pkl")
