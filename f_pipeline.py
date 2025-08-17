# run_pipeline.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # Example model
import os

# ------------------------------
# 1️⃣ Load future fixtures CSV
# ------------------------------
def load_fixtures(file_path="future_fixtures.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Please provide your fixtures CSV.")
    
    df = pd.read_csv(file_path)
    
    # Ensure date parsing
    df["Game_Date"] = pd.to_datetime(df["Game_Date"], dayfirst=True, errors='coerce')
    
    # Ensure odds columns exist
    for col in ["B365H","B365D","B365A"]:
        if col not in df.columns:
            raise ValueError(f"{col} missing from fixtures CSV")
    
    return df

# ------------------------------
# 2️⃣ Feature engineering
# ------------------------------
def encode_odds_features(df: pd.DataFrame, include_ev=False) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    
    # Implied probabilities
    odds_cols = ["B365H", "B365D", "B365A"]
    odds = df[odds_cols].replace(0, np.nan)
    imp = 1.0 / odds
    imp = imp.div(imp.sum(axis=1), axis=0)
    imp.columns = ["imp_H", "imp_D", "imp_A"]
    out = pd.concat([out, imp], axis=1)
    
    # Calculate EV/Value only if predicted probabilities exist
    if include_ev and all(col in df.columns for col in ["p_H","p_D","p_A"]):
        out["EV_H"] = df["p_H"] / out["imp_H"]
        out["EV_D"] = df["p_D"] / out["imp_D"]
        out["EV_A"] = df["p_A"] / out["imp_A"]

        out["Value_H"] = out["EV_H"] > 1
        out["Value_D"] = out["EV_D"] > 1
        out["Value_A"] = out["EV_A"] > 1
    
    return out

# ------------------------------
# 3️⃣ Prepare features for model
# ------------------------------
def make_dataset(df):
    # Currently using implied probabilities as features
    X = encode_odds_features(df, include_ev=False)
    X = X.fillna(0)  # handle any remaining NaNs
    return X

# ------------------------------
# 4️⃣ Predict probabilities
# ------------------------------
def predict_probabilities(df, model):
    X_future = make_dataset(df)
    
    # Example: placeholder model prediction
    # Replace with your trained model
    if not hasattr(model, "predict_proba"):
        # Random placeholder probabilities if no model
        n = len(df)
        random_probs = np.random.dirichlet([1,1,1], n)
        df[["p_H","p_D","p_A"]] = random_probs
    else:
        proba = model.predict_proba(X_future)
        if isinstance(proba, list):  # multi-class RF returns list for each class
            df[["p_H","p_D","p_A"]] = np.hstack(proba)
        else:
            df[["p_H","p_D","p_A"]] = proba
    
    return df

# ------------------------------
# 5️⃣ Main pipeline
# ------------------------------
def main(fixtures_file="future_fixtures.csv", output_file="predictions.csv"):
    df = load_fixtures(fixtures_file)
    
    # Initialize model (replace with your trained model)
    model = RandomForestClassifier()
    
    # Predict probabilities
    df = predict_probabilities(df, model)
    
    # Calculate EV and Value bets
    odds_features = encode_odds_features(df, include_ev=True)
    df = pd.concat([df, odds_features[["EV_H","EV_D","EV_A","Value_H","Value_D","Value_A"]]], axis=1)
    
    # Save predictions
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# ------------------------------
# 6️⃣ Run script
# ------------------------------
if __name__ == "__main__":
    main()
