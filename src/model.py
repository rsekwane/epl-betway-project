import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
import joblib

from .utils import parse_dates, standardize_teams, add_season, RESULT_MAP
from .features import build_basic_features, encode_odds_features

TARGET = "FTR"

def make_dataset(df: pd.DataFrame, rolling_windows, seed=42) -> Tuple[pd.DataFrame, pd.Series]:
    df = parse_dates(df)
    df = standardize_teams(df)
    df = df.sort_values("Date").reset_index(drop=True)
    df = add_season(df)

    # Build features
    X_team = build_basic_features(df, windows=rolling_windows)
    X_odds = encode_odds_features(df)

    # Categorical ID features
    X_cats = df[["HomeTeam","AwayTeam","Season"]].copy()

    # Merge
    X = pd.concat([X_team.reset_index(drop=True), X_odds.reset_index(drop=True), X_cats.reset_index(drop=True)], axis=1)

    # Target
    y = df[TARGET].map({"H":0, "D":1, "A":2})  # consistent ordering for columns

    return X, y, df

def build_pipeline(X: pd.DataFrame) -> Pipeline:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = ["HomeTeam","AwayTeam","Season"]

    pre = ColumnTransformer([
        ("num", StandardScaler(with_mean=False), [c for c in num_cols if c not in ["HomeAdvantage"]]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ], remainder="passthrough")

    base = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=None,
        max_iter=400,
        l2_regularization=0.01,
        random_state=42
    )

    # Calibrate for better probability estimates
    clf = CalibratedClassifierCV(base_estimator=base, method="isotonic", cv=3)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe

def evaluate_cv(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, n_splits=5, random_state=42) -> Dict[str, Any]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = []
    for train_idx, test_idx in tscv.split(X):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)
        pred = proba.argmax(axis=1)
        metrics.append({
            "log_loss": float(log_loss(yte, proba, labels=[0,1,2])),
            "brier": float(np.mean(np.sum((proba - pd.get_dummies(yte).reindex(columns=[0,1,2], fill_value=0).values)**2, axis=1))),
            "accuracy": float(accuracy_score(yte, pred)),
        })
    agg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0].keys()}
    agg["folds"] = metrics
    return agg

def fit_full(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipe.fit(X, y)
    return pipe

def predict_with_ev(pipe: Pipeline, X: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    proba = pipe.predict_proba(X)
    preds = pd.DataFrame(proba, columns=["p_H","p_D","p_A"], index=X.index)
    # Implied from Betway if available
    bet_cols = ["BWH","BWD","BWA"]
    implied = None
    if all(c in df_raw.columns for c in bet_cols):
        odds = df_raw[bet_cols].replace(0, np.nan)
        imp = 1.0 / odds
        imp = imp.div(imp.sum(axis=1), axis=0)
        implied = pd.DataFrame(imp.values, columns=["imp_H","imp_D","imp_A"], index=X.index)
        preds = pd.concat([preds, implied], axis=1)
        # EVs
        preds["EV_H"] = preds["p_H"] * df_raw["BWH"] - 1.0
        preds["EV_D"] = preds["p_D"] * df_raw["BWD"] - 1.0
        preds["EV_A"] = preds["p_A"] * df_raw["BWA"] - 1.0
        preds["Value_H"] = preds["EV_H"] > 0
        preds["Value_D"] = preds["EV_D"] > 0
        preds["Value_A"] = preds["EV_A"] > 0
    return preds

def save_pipeline(pipe: Pipeline, path: str):
    import joblib
    joblib.dump(pipe, path)
