# EPL Betway Prediction — End‑to‑End Project

This project builds a **probability model** for English Premier League match outcomes (Home/Draw/Away) and identifies **value bets** versus **Betway** odds.  
It expects a CSV with (at least) the columns you listed (verbatim headers):

`Div, Date, Time, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HTHG, HTAG, HTR, Referee, HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR, B365H, B365D, B365A, BWH, BWD, BWA, BFH, BFD, BFA, PSH, PSD, PSA, WHH, WHD, WHA, 1XBH, 1XBD, 1XBA, MaxH, MaxD, MaxA, AvgH, AvgD, AvgA, BFEH, BFED, BFEA, B365>2.5, B365<2.5, P>2.5, P<2.5, Max>2.5, Max<2.5, Avg>2.5, Avg<2.5, BFE>2.5, BFE<2.5, AHh, B365AHH, B365AHA, PAHH, PAHA, MaxAHH, MaxAHA, AvgAHH, AvgAHA, BFEAHH, BFEAHA, B365CH, B365CD, B365CA, BWCH, BWCD, BWCA, BFCH, BFCD, BFCA, PSCH, PSCD, PSCA, WHCH, WHCD, WHCA, 1XBCH, 1XBCD, 1XBCA, MaxCH, MaxCD, MaxCA, AvgCH, AvgCD, AvgCA, BFECH, BFECD, BFECA, B365C>2.5, B365C<2.5, PC>2.5, PC<2.5, MaxC>2.5, MaxC<2.5, AvgC>2.5, AvgC<2.5, BFEC>2.5, BFEC<2.5, AHCh, B365CAHH, B365CAHA, PCAHH, PCAHA, MaxCAHH, MaxCAHA, AvgCAHH, AvgCAHA, BFECAHH, BFECAHA`

> Place your CSV at: `data/epl_matches.csv`

## What it does
- Cleans and validates columns, parses dates, standardizes team names
- Builds **leakage-safe rolling features** per team (last-N form, goals, shots, cards, etc.)
- Encodes bookmaker odds (Betway + market averages) into implied probabilities
- Splits data with **time-based cross‑validation** (no look-ahead)
- Trains a calibrated **HistGradientBoostingClassifier** (multiclass probabilities)
- Evaluates with **Log Loss, Brier Score, Accuracy**
- Exports predictions with **EV (expected value)** vs **Betway** per outcome
- Saves a production **pipeline.pkl** for future inference

## Quick start
1. Put your dataset here: `data/epl_matches.csv`  
2. (Optional) Adjust `config.yaml`
3. Create a virtual environment and install requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the full pipeline:
   ```bash
   python run_pipeline.py --config config.yaml
   ```
5. Outputs land in `outputs/`:
   - `metrics.json` — CV and holdout metrics
   - `predictions.csv` — probs, implied probs, EVs, value flags
   - `pipeline.pkl` — trained pipeline

## Value bets logic
We compute **implied probs** from Betway odds (`BWH, BWD, BWA`) with overround normalization.  
We flag **value** when **EV = p_model * odds - 1 > 0**.

## Notes
- Date parsing assumes **day‑first** formats commonly used in football datasets.
- Rolling features use only **past** matches per team (shifted) to avoid leakage.
- The target is `FTR` (H/D/A). You can switch to goal-based regression if needed.
