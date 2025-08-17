import numpy as np
import pandas as pd

RESULT_MAP = {"H": 2, "D": 1, "A": 0}  # for convenience (ordered strength)

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    # Convert Game_Date to datetime explicitly
    if "Game_Date" in df.columns:
        df["Game_Date"] = pd.to_datetime(df["Game_Date"], dayfirst=True, errors="coerce")
    # Ensure Time column is string type to avoid issues
    if "Time" in df.columns:
        df["Time"] = df["Time"].astype(str)
    return df

def standardize_teams(df: pd.DataFrame) -> pd.DataFrame:
    # Harmonize team names: strip spaces and ensure string type
    for col in ["HomeTeam", "AwayTeam"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

def season_from_date(d: pd.Timestamp) -> str:
    # EPL season: Aug-May; assign season by start year
    if pd.isna(d):
        return np.nan
    year = d.year
    month = d.month
    start_year = year if month >= 7 else year - 1
    return f"{start_year}/{(start_year + 1) % 100:02d}"

def add_season(df: pd.DataFrame) -> pd.DataFrame:
    # Apply season extraction safely
    if "Game_Date" in df.columns:
        df["Season"] = df["Game_Date"].apply(season_from_date)
    else:
        df["Season"] = np.nan
    return df

def implied_probs_from_odds(odds: pd.DataFrame) -> pd.DataFrame:
    # Convert odds to implied probabilities, normalize per row
    probs = 1.0 / odds.replace(0, np.nan)
    s = probs.sum(axis=1)
    return probs.div(s, axis=0)

def expected_value(p: float, odds: float) -> float:
    # EV formula: probability * odds - 1
    return p * odds - 1.0

def safe_div(a, b):
    # Safe division avoiding divide-by-zero
    return np.where(b != 0, a / b, np.nan)
