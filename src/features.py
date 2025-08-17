import numpy as np
import pandas as pd
from typing import List, Tuple
from .utils import RESULT_MAP

def outcome_to_numeric(x: str) -> int:
    # Map H/D/A to 2/1/0 (optional ordering)
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
                .apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )
    return out

def build_basic_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Construct leakage-safe, per-team rolling features and match context features."""
    df = df.sort_values("Game_Date").copy()
    # numeric base stats (Full time + first order)
    num_cols = [c for c in ["FTHG","FTAG","HS","AS","HST","AST","HF","AF","HC","AC","HY","AY","HR","AR"] if c in df.columns]

    # Create per-team "for" stats per match perspective
    # Home perspective: 'for' equals home stats; Away perspective: 'for' equals away stats
    temp = df.copy()
    temp["Team_home"] = temp["HomeTeam"]
    temp["Team_away"] = temp["AwayTeam"]

    # Build past-form features for home and away separately
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

    # Prepare aligned frames to merge
    frames = []
    for w in windows:
        # Home team rolling
        h = temp.groupby("HomeTeam").apply(
            lambda g: g.assign(
                **{f"home_{k}_roll{w}": g[v].shift(1).rolling(w, min_periods=1).mean() for k,v in home_vals.items()}
            )
        ).reset_index(level=0, drop=True)[[
            "Game_Date","HomeTeam","AwayTeam"] + [f"home_{k}_roll{w}" for k in home_vals.keys()]]
        # Away team rolling
        a = temp.groupby("AwayTeam").apply(
            lambda g: g.assign(
                **{f"away_{k}_roll{w}": g[v].shift(1).rolling(w, min_periods=1).mean() for k,v in away_vals.items()}
            )
        ).reset_index(level=0, drop=True)[[
            "Game_Date","HomeTeam","AwayTeam"] + [f"away_{k}_roll{w}" for k in away_vals.keys()]]
        # Merge on row index
        m = h.merge(a, left_index=True, right_index=True, suffixes=("",""))
        frames.append(m)

    feat = pd.concat(frames, axis=1)
    # Deduplicate key columns
    feat = feat.loc[:,~feat.columns.duplicated()]
    # Context features
    feat["HomeAdvantage"] = 1.0  # constant; allows model to learn base bias
    return feat

def encode_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create implied probs and price-based features for Betway and market averages."""
    out = pd.DataFrame(index=df.index)
    for bprefix, cols in {
        "betway": ["BWH","BWD","BWA"],
        "avg": ["AvgH","AvgD","AvgA"],
        "b365": ["B365H","B365D","B365A"]
    }.items():
        cols = [c for c in cols if c in df.columns]
        if len(cols)==3:
            odds = df[cols].replace(0, np.nan)
            imp = 1.0 / odds
            # normalize overround
            imp = imp.div(imp.sum(axis=1), axis=0)
            imp.columns = [f"{bprefix}_imp_H", f"{bprefix}_imp_D", f"{bprefix}_imp_A"]
            out = pd.concat([out, imp], axis=1)
            # raw odds too
            odds.columns = [f"{bprefix}_odds_H", f"{bprefix}_odds_D", f"{bprefix}_odds_A"]
            out = pd.concat([out, odds], axis=1)
    return out
