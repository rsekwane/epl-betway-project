# run_pipeline.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# ------------------------------
# 1️⃣ Load historical results
# ------------------------------
def load_historical(file_path="data/matches.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Provide historical results CSV.")
    
    df = pd.read_csv(file_path)
    df["Game_Date"] = pd.to_datetime(df["Game_Date"], dayfirst=True, errors='coerce')
    
    # Ensure necessary columns exist
    for col in ["HomeTeam", "AwayTeam", "FTR", "B365H", "B365D", "B365A"]:
        if col not in df.columns:
            raise ValueError(f"{col} missing from historical CSV")
    
    return df

# ------------------------------
# 2️⃣ Feature engineering
# ------------------------------
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    
    # 1. Implied probabilities from odds
    odds_cols = ["B365H", "B365D", "B365A"]
    odds = df[odds_cols].replace(0, np.nan)
    imp = 1.0 / odds
    imp = imp.div(imp.sum(axis=1), axis=0)
    imp.columns = ["imp_H", "imp_D", "imp_A"]
    out = pd.concat([out, imp], axis=1)
    
    # 2. Recent form (rolling 3 matches)
    df_sorted = df.sort_values("Game_Date")
    for col, team_col in [("HomeForm", "HomeTeam"), ("AwayForm", "AwayTeam")]:
        out[col] = 0.0  # initialize
        for team in df[team_col].unique():
            team_mask = df[team_col] == team
            # encode result as 1=win, 0.5=draw, 0=loss
            result = df.loc[team_mask, "FTR"].map({"H":1, "D":0.5, "A":0}).values if team_col=="HomeTeam" else df.loc[team_mask, "FTR"].map({"A":1, "D":0.5, "H":0}).values
            out.loc[team_mask, col] = pd.Series(result).rolling(3, min_periods=1).mean().values
    
    # Fill any remaining NaNs
    out = out.fillna(0)
    return out

# ------------------------------
# 3️⃣ Prepare dataset
# ------------------------------
def prepare_dataset(df):
    X = encode_features(df)
    # Encode target
    y_map = {"H":0, "D":1, "A":2}
    y = df["FTR"].map(y_map)
    return X, y

# ------------------------------
# 4️⃣ Train model
# ------------------------------
def train_model(df):
    X, y = prepare_dataset(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=500, random_state=42)
    model.fit(X_train, y_train)
    
    acc = model.score(X_test, y_test)
    print(f"Model accuracy: {acc:.2f}")
    
    return model

# ------------------------------
# 5️⃣ Predict future
# ------------------------------
def predict_future(fixtures_file="data/matches.csv", model=None):
    if not os.path.exists(fixtures_file):
        raise FileNotFoundError(f"{fixtures_file} not found.")
    
    df_future = pd.read_csv(fixtures_file)
    df_future["Game_Date"] = pd.to_datetime(df_future["Game_Date"], dayfirst=True, errors='coerce')
    
    X_future = encode_features(df_future)
    
    if model is None:
        # fallback random predictions
        n = len(df_future)
        random_probs = np.random.dirichlet([1,1,1], n)
        df_future[["p_H","p_D","p_A"]] = random_probs
    else:
        probs = model.predict_proba(X_future)
        # RandomForest returns array of shape (n_samples, n_classes)
        if isinstance(probs, list):
            df_future[["p_H","p_D","p_A"]] = np.hstack(probs)
        else:
            df_future[["p_H","p_D","p_A"]] = probs
    
    # Calculate EV and Value bets
    odds_features = encode_features(df_future)
    df_future = pd.concat([df_future, odds_features[["imp_H","imp_D","imp_A"]]], axis=1)
    
    df_future["EV_H"] = df_future["p_H"] / df_future["imp_H"]
    df_future["EV_D"] = df_future["p_D"] / df_future["imp_D"]
    df_future["EV_A"] = df_future["p_A"] / df_future["imp_A"]
    
    df_future["Value_H"] = df_future["EV_H"] > 1
    df_future["Value_D"] = df_future["EV_D"] > 1
    df_future["Value_A"] = df_future["EV_A"] > 1
    
    # Save predictions
    df_future.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")

# ------------------------------
# 6️⃣ Main pipeline
# ------------------------------
def main(historical_file="data/matches.csv", fixtures_file="data/future_matches.csv"):
    df_hist = load_historical(historical_file)
    model = train_model(df_hist)
    predict_future(fixtures_file, model=model)

# ------------------------------
# 7️⃣ Run script
# ------------------------------
if __name__ == "__main__":
    main()
