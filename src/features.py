import numpy as np
import pandas as pd
from typing import List, Dict

# Simple result mapping
RESULT_MAP: Dict[str, int] = {"H": 2, "D": 1, "A": 0}

def outcome_to_numeric(x: str) -> int:
    """Convert match outcome H/D/A to numeric 2/1/0."""
    return RESULT_MAP.get(x, np.nan)

def _team_rolling(df: pd.DataFrame, team_col: str, value_cols: List[str], windows: List[int], prefix: str) -> pd.DataFrame:
    """Create rolling means per team using match order, shifted to avoid leakage."""
    df = df.sort_values("Game_Date").copy()
    out = df[["Game_Date", "HomeTeam", "AwayTeam"]].copy()
    team_series = df[team_col]
    for col in value_cols:
        for w in windows:
            name = f"{prefix}_{col}_roll{w}"
            out[name] = (
                df.groupby(team_series)[col]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            )
    return out

def build_basic_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Construct leakage-safe, per-team rolling features and match context features."""
    df = df.sort_values("Game_Date").copy()

    # Home and away stats mapping
    home_vals = {
        "goals_for":"FTHG","goals_against":"FTAG","shots_for":"HS","shots_against":"AS",
        "sot_for":"HST","sot_against":"AST","fouls_for":"HF","fouls_against":"AF",
        "corners_for":"HC","corners_against":"AC","yellows_for":"HY","yellows_against":"AY",
        "reds_for":"HR","reds_against":"AR"
    }
    away_vals = {
        "goals_for":"FTAG","goals_against":"FTHG","shots_for":"AS","shots_against":"HS",
        "sot_for":"AST","sot_against":"HST","fouls_for":"AF","fouls_against":"HF",
        "corners_for":"AC","corners_against":"HC","yellows_for":"AY","yellows_against":"HY",
        "reds_for":"AR","reds_against":"HR"
    }

    temp = df.copy()
    temp["Team_home"] = temp["HomeTeam"]
    temp["Team_away"] = temp["AwayTeam"]

    frames = []
    for w in windows:
        # Home rolling (using transform instead of apply)
        h = temp.copy()
        for k, v in home_vals.items():
            h[f"home_{k}_roll{w}"] = (
                h.groupby("HomeTeam")[v]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            )
        h = h[["Game_Date","HomeTeam","AwayTeam"] + [f"home_{k}_roll{w}" for k in home_vals.keys()]]

        # Away rolling (using transform instead of apply)
        a = temp.copy()
        for k, v in away_vals.items():
            a[f"away_{k}_roll{w}"] = (
                a.groupby("AwayTeam")[v]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            )
        a = a[["Game_Date","HomeTeam","AwayTeam"] + [f"away_{k}_roll{w}" for k in away_vals.keys()]]

        # Merge
        m = h.merge(a, left_index=True, right_index=True, suffixes=("_h", "_a"))
        frames.append(m)

    feat = pd.concat(frames, axis=1)
    feat = feat.loc[:, ~feat.columns.duplicated()]
    feat["HomeAdvantage"] = 1.0  # constant
    return feat

def encode_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create implied probabilities and raw odds features for multiple bookmakers."""
    out = pd.DataFrame(index=df.index)
    bookmakers = {
        "betway": ["BWH","BWD","BWA"],
        "avg": ["AvgH","AvgD","AvgA"],
        "b365": ["B365H","B365D","B365A"]
    }

    for bprefix, cols in bookmakers.items():
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"⚠️ Warning: Missing odds columns for {bprefix}: {missing}")
            continue

        odds = df[cols].replace(0, np.nan)
        imp = 1.0 / odds
        imp = imp.div(imp.sum(axis=1), axis=0)
        imp.columns = [f"{bprefix}_imp_H", f"{bprefix}_imp_D", f"{bprefix}_imp_A"]
        out = pd.concat([out, imp], axis=1)

        odds.columns = [f"{bprefix}_odds_H", f"{bprefix}_odds_D", f"{bprefix}_odds_A"]
        out = pd.concat([out, odds], axis=1)

    return out
