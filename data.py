"""
World Cup 2026 Match Predictor — Data Download & Loading
=========================================================
Step 1: Install the Kaggle API
    pip install kaggle

Step 2: Get your API key
    Go to kaggle.com → Account → "Create New API Token"
    Save the downloaded kaggle.json to ~/.kaggle/kaggle.json
    On Linux/Mac: chmod 600 ~/.kaggle/kaggle.json

Step 3: Run this script
    python download_data.py
"""

import subprocess
import os
import pandas as pd 

# ── Download datasets ──────────────────────────────────────────────────────────

os.makedirs("data/raw", exist_ok=True)

datasets = {
    "results":  "martj42/international-football-results-from-1872-to-2017",
    "rankings": "cashncarry/fifaworldranking",
    "elo":      "saifalnimri/international-football-elo-ratings",
}

for name, slug in datasets.items():
    print(f"Downloading {name}...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug,
         "-p", f"data/raw/{name}", "--unzip"],
        check=True
    )
    print(f"  ✓ Saved to data/raw/{name}/")

# ── Load & preview ─────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("LOADING DATASETS")
print("="*60)

# 1. Match results
results = pd.read_csv("data/raw/results/results.csv", parse_dates=["date"])
print(f"\n[1] Match results: {results.shape[0]:,} rows × {results.shape[1]} cols")
print(results.head(3).to_string())

# 2. FIFA rankings
import glob
rankings_file = sorted(glob.glob("data/raw/rankings/fifa_ranking-*.csv"))[-1]
print(f"Using rankings file: {rankings_file}")
rankings = pd.read_csv(rankings_file, parse_dates=["rank_date"])
print(f"\n[2] FIFA rankings: {rankings.shape[0]:,} rows × {rankings.shape[1]} cols")
print(rankings.head(3).to_string())

# 3. Elo ratings
elo = pd.read_csv("data/raw/elo/eloratings.csv", parse_dates=["date"])
print(f"\n[3] Elo ratings: {elo.shape[0]:,} rows × {elo.shape[1]} cols")
print(elo.head(3).to_string())

# ── Quick quality check ────────────────────────────────────────────────────────

print("\n" + "="*60)
print("QUALITY CHECKS")
print("="*60)

print(f"\nResults span: {results['date'].min().date()} → {results['date'].max().date()}")
print(f"Unique teams:  {pd.concat([results['home_team'], results['away_team']]).nunique()}")
print(f"Tournaments:   {results['tournament'].nunique()} unique types")
print(f"\nTop tournaments:\n{results['tournament'].value_counts().head(8).to_string()}")
print(f"\nMissing values in results:\n{results.isnull().sum()[results.isnull().sum() > 0]}")

# ── Filter: keep only from 2000 onwards for recency ───────────────────────────

results_filtered = results[results["date"] >= "2000-01-01"].copy()
print(f"\nFiltered results (2000+): {results_filtered.shape[0]:,} matches")

# ── Save cleaned versions ──────────────────────────────────────────────────────

os.makedirs("data/processed", exist_ok=True)
results_filtered.to_csv("data/processed/results_2000_plus.csv", index=False)
rankings.to_csv("data/processed/rankings.csv", index=False)
elo.to_csv("data/processed/elo.csv", index=False)

print("\n✓ Processed files saved to data/processed/")
