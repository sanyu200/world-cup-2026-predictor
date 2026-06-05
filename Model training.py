"""
World Cup 2026 Match Predictor — Model Training
================================================
Input:  data/processed/match_features.csv
Output: models/logistic_regression.pkl
        models/random_forest.pkl
        models/xgboost.pkl
        models/scaler.pkl
        models/feature_columns.txt
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics         import (accuracy_score, log_loss,
                                     classification_report)
from xgboost import XGBClassifier

os.makedirs("models", exist_ok=True)

# ── Load features ──────────────────────────────────────────────────────────────

df = pd.read_csv("data/processed/match_features.csv", parse_dates=["date"])
print(f"Loaded {len(df):,} matches")

# ── Feature columns ────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "elo_diff",
    "home_elo",
    "away_elo",
    "rank_diff",
    "home_form",
    "away_form",
    "form_diff",
    "home_goals_scored_avg",
    "home_goals_conceded_avg",
    "away_goals_scored_avg",
    "away_goals_conceded_avg",
    "h2h_home_wins",
    "h2h_draws",
    "h2h_away_wins",
    "is_neutral",
]

TARGET = "result"   # 0=away win, 1=draw, 2=home win

# ── Train / test split — use last 2 years as hold-out ─────────────────────────
# Temporal split is critical for sports prediction — never random split,
# otherwise you "leak" future information into training.

cutoff = pd.Timestamp("2022-01-01")
train_df = df[df["date"] < cutoff].copy()
test_df  = df[df["date"] >= cutoff].copy()

X_train = train_df[FEATURE_COLS].fillna(0).values
y_train = train_df[TARGET].values
X_test  = test_df[FEATURE_COLS].fillna(0).values
y_test  = test_df[TARGET].values

print(f"Train: {len(X_train):,} matches  |  Test: {len(X_test):,} matches")
print(f"Class distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")

# ── Scale features (needed for Logistic Regression) ───────────────────────────

scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Cross-validation setup ─────────────────────────────────────────────────────

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_metrics = ["accuracy", "neg_log_loss"]

# ── Define models ──────────────────────────────────────────────────────────────

models = {
    "Logistic Regression": (
        LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        ),
        X_train_scaled,   # uses scaled features
        X_test_scaled,
    ),
    "Random Forest": (
        RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        X_train,          # tree models don't need scaling
        X_test,
    ),
    "XGBoost": (
        XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        ),
        X_train,
        X_test,
    ),
}

# ── Train, cross-validate, evaluate ───────────────────────────────────────────

results_summary = []

for name, (model, X_tr, X_te) in models.items():
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")

    # 5-fold cross-validation on training set
    cv_results = cross_validate(
        model, X_tr, y_train,
        cv=cv,
        scoring=cv_metrics,
        return_train_score=False,
    )
    cv_acc     = cv_results["test_accuracy"].mean()
    cv_acc_std = cv_results["test_accuracy"].std()
    cv_ll      = -cv_results["test_neg_log_loss"].mean()

    print(f"  CV accuracy : {cv_acc:.4f} ± {cv_acc_std:.4f}")
    print(f"  CV log loss : {cv_ll:.4f}")

    # Fit on full training set
    model.fit(X_tr, y_train)

    # Evaluate on hold-out test set
    y_pred      = model.predict(X_te)
    y_pred_prob = model.predict_proba(X_te)
    test_acc    = accuracy_score(y_test, y_pred)
    test_ll     = log_loss(y_test, y_pred_prob)

    print(f"\n  Test accuracy : {test_acc:.4f}")
    print(f"  Test log loss : {test_ll:.4f}")
    print(f"\n  Classification report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Away win", "Draw", "Home win"]
    ))

    results_summary.append({
        "Model":        name,
        "CV Accuracy":  f"{cv_acc:.4f} ± {cv_acc_std:.4f}",
        "CV Log Loss":  f"{cv_ll:.4f}",
        "Test Accuracy": f"{test_acc:.4f}",
        "Test Log Loss": f"{test_ll:.4f}",
    })

    # Save model
    safe_name = name.lower().replace(" ", "_")
    with open(f"models/{safe_name}.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"\n  ✓ Saved to models/{safe_name}.pkl")

# ── Save scaler & feature list (needed by Streamlit app later) ────────────────

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("models/feature_columns.txt", "w") as f:
    f.write("\n".join(FEATURE_COLS))

# ── Summary table ──────────────────────────────────────────────────────────────

print(f"\n{'='*55}")
print("  SUMMARY")
print(f"{'='*55}")
print(pd.DataFrame(results_summary).to_string(index=False))
print("\nNext step: run evaluation.py for SHAP feature importance")