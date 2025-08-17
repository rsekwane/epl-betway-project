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
    out = pd.DataFrame(index=df.index)

    # if real odds exist
    odds_cols = {
        "betway": ["BWH","BWD","BWA"],
        "avg": ["AvgH","AvgD","AvgA"],
        "b365": ["B365H","B365D","B365A"]
    }
    added = False
    for prefix, cols in odds_cols.items():
        if all(c in df.columns for c in cols):
            odds = df[cols].replace(0, np.nan)
            imp = 1.0 / odds
            imp = imp.div(imp.sum(axis=1), axis=0)
            imp.columns = [f"imp_H", f"imp_D", f"imp_A"]
            out = pd.concat([out, imp], axis=1)
            added = True
    
    # fallback if no odds
    if not added:
        out["imp_H"] = 1/3
        out["imp_D"] = 1/3
        out["imp_A"] = 1/3

    # compute expected value + value bets
    out["EV_H"] = df["p_H"] / out["imp_H"]
    out["EV_D"] = df["p_D"] / out["imp_D"]
    out["EV_A"] = df["p_A"] / out["imp_A"]

    out["Value_H"] = out["EV_H"] > 1
    out["Value_D"] = out["EV_D"] > 1
    out["Value_A"] = out["EV_A"] > 1

    return out

