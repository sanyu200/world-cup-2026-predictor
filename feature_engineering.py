"""
World Cup 2026 Match Predictor — Feature Engineering
=====================================================
Input:  data/processed/results_2000_plus.csv
        data/processed/elo.csv
        data/processed/rankings.csv

Output: data/processed/match_features.csv

Features engineered:
  - elo_diff          : Elo rating difference at match time
  - home_elo / away_elo
  - home_form / away_form   : points from last 5 matches (W=3, D=1, L=0)
  - home_goals_scored_avg / away_goals_scored_avg  : rolling 5-match avg
  - home_goals_conceded_avg / away_goals_conceded_avg
  - h2h_home_wins / h2h_draws / h2h_away_wins : last 5 head-to-head
  - rank_diff         : FIFA rank difference (home - away)
  - is_neutral        : 1 if played on neutral ground
  - result            : target label  0=away win, 1=draw, 2=home win
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

# ── Load data ──────────────────────────────────────────────────────────────────

results  = pd.read_csv("data/processed/results_2000_plus.csv")
elo_df   = pd.read_csv("data/processed/elo.csv")
rank_df  = pd.read_csv("data/processed/rankings.csv")

results["date"]          = pd.to_datetime(results["date"])
elo_df["date"]           = pd.to_datetime(elo_df["date"], format="mixed", dayfirst=False)
rank_df["rank_date"]     = pd.to_datetime(rank_df["rank_date"])

results = results.sort_values("date").reset_index(drop=True)

# ── Target label ───────────────────────────────────────────────────────────────

def get_result(row):
    if row["home_score"] > row["away_score"]:
        return 2   # home win
    elif row["home_score"] == row["away_score"]:
        return 1   # draw
    else:
        return 0   # away win

results["result"] = results.apply(get_result, axis=1)

# ── Helper: get Elo rating for a team just before a given date ─────────────────

def get_elo(team, date):
    team_elo = elo_df[elo_df["team"] == team]
    past = team_elo[team_elo["date"] < date]
    if past.empty:
        return 1500.0   # default starting Elo
    return past.iloc[-1]["rating"]

# ── Helper: get FIFA rank for a team just before a given date ──────────────────

def get_rank(team, date):
    team_rank = rank_df[rank_df["country_full"] == team]
    past = team_rank[team_rank["rank_date"] < date]
    if past.empty:
        return 100   # default rank if unknown
    return past.iloc[-1]["rank"]

# ── Helper: rolling form & goal stats from last N matches ─────────────────────

def get_team_recent_stats(team, date, n=5):
    """
    Returns (form_points, avg_goals_scored, avg_goals_conceded)
    from the last n matches before `date`.
    """
    mask = (
        ((results["home_team"] == team) | (results["away_team"] == team)) &
        (results["date"] < date)
    )
    recent = results[mask].tail(n)

    if recent.empty:
        return 0.0, 0.0, 0.0

    points, scored, conceded = [], [], []

    for _, row in recent.iterrows():
        if row["home_team"] == team:
            s, c = row["home_score"], row["away_score"]
        else:
            s, c = row["away_score"], row["home_score"]

        scored.append(s)
        conceded.append(c)

        if s > c:
            points.append(3)
        elif s == c:
            points.append(1)
        else:
            points.append(0)

    form = sum(points) / (3 * len(points))   # normalised 0→1
    return form, np.mean(scored), np.mean(conceded)

# ── Helper: head-to-head record (last N meetings) ─────────────────────────────

def get_h2h(home, away, date, n=5):
    """
    Returns (home_wins, draws, away_wins) from last n h2h matches.
    Always from the perspective of `home` vs `away`.
    """
    mask = (
        (
            ((results["home_team"] == home) & (results["away_team"] == away)) |
            ((results["home_team"] == away) & (results["away_team"] == home))
        ) &
        (results["date"] < date)
    )
    h2h = results[mask].tail(n)

    if h2h.empty:
        return 0, 0, 0

    hw, d, aw = 0, 0, 0
    for _, row in h2h.iterrows():
        if row["home_team"] == home:
            gs, gc = row["home_score"], row["away_score"]
        else:
            gs, gc = row["away_score"], row["home_score"]

        if gs > gc:
            hw += 1
        elif gs == gc:
            d += 1
        else:
            aw += 1

    total = hw + d + aw
    return hw / total, d / total, aw / total   # normalised

# ── Build feature matrix ───────────────────────────────────────────────────────

rows = []

print(f"Engineering features for {len(results):,} matches...")

for _, match in tqdm(results.iterrows(), total=len(results)):
    date      = match["date"]
    home_team = match["home_team"]
    away_team = match["away_team"]

    # Elo
    home_elo = get_elo(home_team, date)
    away_elo = get_elo(away_team, date)

    # FIFA rank
    home_rank = get_rank(home_team, date)
    away_rank = get_rank(away_team, date)

    # Recent form
    h_form, h_scored, h_conceded = get_team_recent_stats(home_team, date)
    a_form, a_scored, a_conceded = get_team_recent_stats(away_team, date)

    # Head-to-head
    h2h_hw, h2h_d, h2h_aw = get_h2h(home_team, away_team, date)

    rows.append({
        "date":                    date,
        "home_team":               home_team,
        "away_team":               away_team,
        "tournament":              match["tournament"],
        "is_neutral":              int(match["neutral"]),

        # Elo features
        "home_elo":                home_elo,
        "away_elo":                away_elo,
        "elo_diff":                home_elo - away_elo,

        # FIFA rank features (lower rank = better, so away - home = positive = home is better)
        "home_rank":               home_rank,
        "away_rank":               away_rank,
        "rank_diff":               away_rank - home_rank,

        # Form features (normalised 0–1)
        "home_form":               h_form,
        "away_form":               a_form,
        "form_diff":               h_form - a_form,

        # Goal stats
        "home_goals_scored_avg":   h_scored,
        "home_goals_conceded_avg": h_conceded,
        "away_goals_scored_avg":   a_scored,
        "away_goals_conceded_avg": a_conceded,

        # Head-to-head (normalised 0–1)
        "h2h_home_wins":           h2h_hw,
        "h2h_draws":               h2h_d,
        "h2h_away_wins":           h2h_aw,

        # Target
        "result":                  match["result"],
    })

features = pd.DataFrame(rows)

# ── Sanity check ───────────────────────────────────────────────────────────────

print(f"\nFeature matrix shape: {features.shape}")
print(f"\nClass distribution:\n{features['result'].value_counts().rename({0:'Away win', 1:'Draw', 2:'Home win'}).to_string()}")
print(f"\nMissing values:\n{features.isnull().sum()[features.isnull().sum() > 0]}")
print(f"\nSample row:\n{features.iloc[100].to_string()}")

# ── Save ───────────────────────────────────────────────────────────────────────

features.to_csv("data/processed/match_features.csv", index=False)
print("\n✓ Saved to data/processed/match_features.csv")
print("\nNext step: run model_training.py")