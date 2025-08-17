import numpy as np
import pandas as pd

RESULT_MAP = {"H": 2, "D": 1, "A": 0}  # for convenience (ordered strength)

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    if "Game_Date" in df.columns:
        df["Game_Date"] = pd.to_datetime(df["Game_Date"], dayfirst=True, errors="coerce")
    if "Time" in df.columns:
        # Some datasets have "HH:MM"; keep as string, but ensure no NaT issues
        df["Time"] = df["Time"].astype(str)
    return df

def standardize_teams(df: pd.DataFrame) -> pd.DataFrame:
    # Hook for harmonizing team names (e.g., "Man United" vs "Manchester United")
    # Here we keep as-is but strip spaces
    for col in ["HomeTeam", "AwayTeam"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

def season_from_date(d: pd.Timestamp) -> str:
    # EPL season usually spans Aug-May; define season by start year
    year = d.year
    month = d.month
    start_year = year if month >= 7 else year - 1
    return f"{start_year}/{(start_year + 1) % 100:02d}"

def add_season(df: pd.DataFrame) -> pd.DataFrame:
    df["Season"] = df["Game_Date"].apply(season_from_date)
    return df

def implied_probs_from_odds(odds: pd.DataFrame) -> pd.DataFrame:
    # Convert odds to implied probabilities and normalize overround per row
    probs = 1.0 / odds
    s = probs.sum(axis=1)
    return probs.div(s, axis=0)

def expected_value(p: float, odds: float) -> float:
    return p * odds - 1.0

def safe_div(a, b):
    return np.where(b != 0, a / b, np.nan)
