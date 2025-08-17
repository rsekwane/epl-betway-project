import argparse, json, os
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

from src.model import make_dataset, build_pipeline, evaluate_cv, fit_full, predict_with_ev, save_pipeline

def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_path = cfg.get("data_path", "data/epl_matches.csv")
    n_splits = int(cfg.get("n_splits", 5))
    seed = int(cfg.get("seed", 42))
    rolling_windows = cfg.get("rolling_windows", [3,5,10])
    test_size_seasons = cfg.get("test_size_seasons", 1)

    df = pd.read_csv(data_path)
    #df = pd.read_csv(data_path, on_bad_lines='skip')
    X, y, df_proc = make_dataset(df, rolling_windows=rolling_windows, seed=seed)

    # Split by season (hold out the most recent N seasons if possible)
    seasons = df_proc["Season"].unique().tolist()
    seasons.sort()
    if len(seasons) > test_size_seasons:
        holdout_seasons = seasons[-test_size_seasons:]
        train_mask = ~df_proc["Season"].isin(holdout_seasons)
        test_mask = df_proc["Season"].isin(holdout_seasons)
    else:
        # fallback: last 15%
        split_idx = int(len(df_proc) * 0.85)
        train_mask = np.zeros(len(df_proc), dtype=bool)
        train_mask[:split_idx] = True
        test_mask = ~train_mask

    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask], y[test_mask]
    df_te = df_proc.loc[test_mask].copy()

    pipe = build_pipeline(X)

    # CV on training set
    cv_metrics = evaluate_cv(pipe, Xtr, ytr, n_splits=n_splits)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/metrics.json", "w") as f:
        json.dump(cv_metrics, f, indent=2)

    # Fit on all training, evaluate on holdout
    pipe = build_pipeline(X)
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)
    holdout = {
        "log_loss": float(__import__("sklearn.metrics").metrics.log_loss(yte, proba, labels=[0,1,2])),
        "accuracy": float(__import__("sklearn.metrics").metrics.accuracy_score(yte, proba.argmax(axis=1))),
        "brier": float(np.mean(np.sum((proba - pd.get_dummies(yte).reindex(columns=[0,1,2], fill_value=0).values)**2, axis=1)))
    }
    with open("outputs/metrics_holdout.json","w") as f:
        json.dump(holdout, f, indent=2)

    # Predictions + EV
    preds = predict_with_ev(pipe, Xte, df_proc.loc[test_mask])
    out = pd.concat([
        df_te[["Date","HomeTeam","AwayTeam","FTR"]].reset_index(drop=True),
        preds.reset_index(drop=True)
    ], axis=1)
    out.to_csv("outputs/predictions.csv", index=False)

    # Save pipeline
    save_pipeline(pipe, "outputs/pipeline.pkl")

    print("Done. See outputs/ for results.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    main(args.config)
