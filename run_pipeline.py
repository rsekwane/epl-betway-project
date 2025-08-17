import argparse
import json
import os
import pandas as pd
import numpy as np
import yaml
from src.model import make_dataset, build_pipeline, evaluate_cv, predict_with_ev, save_pipeline
from sklearn.preprocessing import OrdinalEncoder

def main(config_path: str):
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_path = cfg.get("data_path", "data/matches.csv")
    n_splits = int(cfg.get("n_splits", 5))
    seed = int(cfg.get("seed", 42))
    rolling_windows = cfg.get("rolling_windows", [3,5,10])
    test_size_seasons = cfg.get("test_size_seasons", 1)

    # Load dataset
    df = pd.read_csv(data_path, sep=';', on_bad_lines='skip')
    X, y, df_proc = make_dataset(df, rolling_windows=rolling_windows, seed=seed)

    # Convert datetime columns to numeric
    datetime_cols = X.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns
    for col in datetime_cols:
        # Option 1: convert to ordinal
        X[col] = X[col].map(pd.Timestamp.toordinal)
        # Option 2: alternatively, extract year/month/day
        # X[col+'_year'] = X[col].dt.year
        # X[col+'_month'] = X[col].dt.month
        # X[col+'_day'] = X[col].dt.day
        # X.drop(columns=[col], inplace=True)

    # Split train/test by season or fallback to last 15%
    seasons = sorted(df_proc["Season"].unique())
    if len(seasons) > test_size_seasons:
        holdout_seasons = seasons[-test_size_seasons:]
        train_mask = ~df_proc["Season"].isin(holdout_seasons)
        test_mask = df_proc["Season"].isin(holdout_seasons)
    else:
        split_idx = int(len(df_proc) * 0.85)
        train_mask = np.zeros(len(df_proc), dtype=bool)
        train_mask[:split_idx] = True
        test_mask = ~train_mask

    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask], y[test_mask]
    df_te = df_proc.loc[test_mask].copy()

    # Build pipeline and evaluate CV
    pipe = build_pipeline(X)
    cv_metrics = evaluate_cv(pipe, Xtr, ytr, n_splits=n_splits)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/metrics.json", "w") as f:
        json.dump(cv_metrics, f, indent=2)

    # Fit full training set and evaluate holdout
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)
    yte_onehot = pd.get_dummies(yte).reindex(columns=[0,1,2], fill_value=0).values
    holdout_metrics = {
        "log_loss": float(log_loss(yte, proba, labels=[0,1,2])),
        "accuracy": float(accuracy_score(yte, proba.argmax(axis=1))),
        "brier": float(np.mean(np.sum((proba - yte_onehot)**2, axis=1)))
    }
    with open("outputs/metrics_holdout.json","w") as f:
        json.dump(holdout_metrics, f, indent=2)

    # Predictions + expected value
    preds = predict_with_ev(pipe, Xte, df_proc.loc[test_mask])
    out = pd.concat([
        df_te[["Game_Date","HomeTeam","AwayTeam","FTR"]].reset_index(drop=True),
        preds.reset_index(drop=True)
    ], axis=1)
    out.to_csv("outputs/predictions.csv", index=False)

    # Save pipeline
    save_pipeline(pipe, "outputs/pipeline.pkl")

    print("Done. See outputs/ for results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)