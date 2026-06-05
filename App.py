"""
World Cup 2026 Match Predictor — Streamlit App
===============================================
Run with:
    streamlit run app.py
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="World Cup 2026 Predictor",
    page_icon="⚽",
    layout="centered",
)

# ── Load models & data ─────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    with open("models/xgboost.pkl",  "rb") as f: xgb    = pickle.load(f)
    with open("models/scaler.pkl",   "rb") as f: scaler = pickle.load(f)
    with open("models/feature_columns.txt") as f:
        features = [line.strip() for line in f.readlines()]
    return xgb, scaler, features

@st.cache_data
def load_data():
    df  = pd.read_csv("data/processed/match_features.csv", parse_dates=["date"])
    elo = pd.read_csv("data/processed/elo.csv")
    elo["date"] = pd.to_datetime(elo["date"], format="mixed", dayfirst=False)
    rk  = pd.read_csv("data/processed/rankings.csv")
    rk["rank_date"] = pd.to_datetime(rk["rank_date"])
    return df, elo, rk

xgb, scaler, FEATURE_COLS = load_models()
df, elo_df, rank_df = load_data()

# ── World Cup 2026 all 48 teams ────────────────────────────────────────────────

WC2026_TEAMS = sorted([
    "Mexico", "South Africa", "South Korea", "Czech Republic",
    "Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland",
    "Brazil", "Morocco", "Haiti", "Scotland",
    "United States", "Paraguay", "Australia", "Turkey",
    "Germany", "Curacao", "Ivory Coast", "Ecuador",
    "Netherlands", "Japan", "Sweden", "Tunisia",
    "Belgium", "Egypt", "Iran", "New Zealand",
    "Spain", "Cape Verde", "Saudi Arabia", "Uruguay",
    "France", "Senegal", "Iraq", "Norway",
    "Argentina", "Algeria", "Austria", "Jordan",
    "Portugal", "DR Congo", "Uzbekistan", "Colombia",
    "England", "Croatia", "Ghana", "Panama",
])

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_latest_elo(team):
    rows = elo_df[elo_df["team"] == team]
    return float(rows.iloc[-1]["rating"]) if not rows.empty else 1500.0

def get_latest_rank(team):
    rows = rank_df[rank_df["country_full"] == team]
    return int(rows.iloc[-1]["rank"]) if not rows.empty else 80

def get_recent_stats(team, n=5):
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    recent = df[mask].tail(n)
    if recent.empty:
        return 0.5, 1.2, 1.0
    points, scored, conceded = [], [], []
    for _, row in recent.iterrows():
        if row["home_team"] == team:
            s, c = row["home_goals_scored_avg"], row["home_goals_conceded_avg"]
            pts  = 3 if row["result"] == 2 else (1 if row["result"] == 1 else 0)
        else:
            s, c = row["away_goals_scored_avg"], row["away_goals_conceded_avg"]
            pts  = 3 if row["result"] == 0 else (1 if row["result"] == 1 else 0)
        points.append(pts); scored.append(s); conceded.append(c)
    form = sum(points) / (3 * len(points))
    return form, np.mean(scored), np.mean(conceded)

def get_h2h(team_a, team_b, n=5):
    mask = (
        ((df["home_team"] == team_a) & (df["away_team"] == team_b)) |
        ((df["home_team"] == team_b) & (df["away_team"] == team_a))
    )
    h2h = df[mask].tail(n)
    if h2h.empty:
        return 0.33, 0.34, 0.33
    aw = d = bw = 0
    for _, row in h2h.iterrows():
        if row["home_team"] == team_a:
            r = row["result"]
        else:
            r = {0: 2, 1: 1, 2: 0}[row["result"]]
        if r == 2: aw += 1
        elif r == 1: d += 1
        else: bw += 1
    total = aw + d + bw
    return aw / total, d / total, bw / total

def build_features(team_a, team_b):
    a_elo = get_latest_elo(team_a)
    b_elo = get_latest_elo(team_b)
    a_rk  = get_latest_rank(team_a)
    b_rk  = get_latest_rank(team_b)
    a_form, a_gs, a_gc = get_recent_stats(team_a)
    b_form, b_gs, b_gc = get_recent_stats(team_b)
    h2h_aw, h2h_d, h2h_bw = get_h2h(team_a, team_b)

    return {
        "elo_diff":                  a_elo - b_elo,
        "home_elo":                  a_elo,
        "away_elo":                  b_elo,
        "rank_diff":                 b_rk - a_rk,
        "home_form":                 a_form,
        "away_form":                 b_form,
        "form_diff":                 a_form - b_form,
        "home_goals_scored_avg":     a_gs,
        "home_goals_conceded_avg":   a_gc,
        "away_goals_scored_avg":     b_gs,
        "away_goals_conceded_avg":   b_gc,
        "h2h_home_wins":             h2h_aw,
        "h2h_draws":                 h2h_d,
        "h2h_away_wins":             h2h_bw,
        "is_neutral":                1,   # World Cup = always neutral
    }

# ── UI ─────────────────────────────────────────────────────────────────────────

st.title("⚽ World Cup 2026 Match Predictor")
st.caption("XGBoost model trained on 25,000+ international matches (2000–2021)")

st.divider()

col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox(" Team A", WC2026_TEAMS, index=WC2026_TEAMS.index("Brazil"))
with col2:
    options_b = [t for t in WC2026_TEAMS if t != team_a]
    team_b = st.selectbox(" Team B", options_b,
                          index=options_b.index("Argentina") if "Argentina" in options_b else 0)

predict_btn = st.button("Predict outcome", type="primary", use_container_width=True)

# ── Prediction ─────────────────────────────────────────────────────────────────

if predict_btn:
    feat_dict = build_features(team_a, team_b)
    X = np.array([[feat_dict[c] for c in FEATURE_COLS]])
    probs = xgb.predict_proba(X)[0]   # [team_b wins, draw, team_a wins]

    prob_b, draw_prob, prob_a = probs

    st.divider()
    st.subheader(f"📊 {team_a} vs {team_b}")

    # ── Probability bars ──

    def prob_bar(label, prob, color):
        st.markdown(f"**{label}**")
        st.progress(float(prob))
        st.markdown(
            f"<p style='margin-top:-12px; font-size:22px; font-weight:600; color:{color}'>{prob*100:.1f}%</p>",
            unsafe_allow_html=True,
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        prob_bar(f"🏆 {team_a}", prob_a, "#2d6a4f")
    with c2:
        prob_bar("🤝 Draw", draw_prob, "#888888")
    with c3:
        prob_bar(f"🏆 {team_b}", prob_b, "#c1440e")

    # ── Verdict ──

    if max(probs) < 0.45:
        verdict = "Very evenly matched — anything can happen"
    elif prob_a > prob_b:
        verdict = f"{team_a} are favourites"
    elif prob_b > prob_a:
        verdict = f"{team_b} are favourites"
    else:
        verdict = "Too close to call — likely a draw"

    st.info(f"**Verdict:** {verdict}", icon="🔍")

    # ── Stats comparison ──

    with st.expander("📋 Team stats comparison"):
        a_elo = get_latest_elo(team_a)
        b_elo = get_latest_elo(team_b)
        stats = {
            "Metric": ["Elo rating", "FIFA rank", "Recent form (0–1)",
                       "Avg goals scored", "Avg goals conceded"],
            team_a: [
                f"{feat_dict['home_elo']:.0f}",
                f"#{get_latest_rank(team_a)}",
                f"{feat_dict['home_form']:.2f}",
                f"{feat_dict['home_goals_scored_avg']:.2f}",
                f"{feat_dict['home_goals_conceded_avg']:.2f}",
            ],
            team_b: [
                f"{feat_dict['away_elo']:.0f}",
                f"#{get_latest_rank(team_b)}",
                f"{feat_dict['away_form']:.2f}",
                f"{feat_dict['away_goals_scored_avg']:.2f}",
                f"{feat_dict['away_goals_conceded_avg']:.2f}",
            ],
        }
        st.dataframe(pd.DataFrame(stats).set_index("Metric"), use_container_width=True)

        h2h_aw, h2h_d, h2h_bw = get_h2h(team_a, team_b)
        st.markdown(
            f"**Head-to-head (last 5 meetings):** "
            f"{team_a} {h2h_aw*5:.0f}W — {h2h_d*5:.0f}D — {h2h_bw*5:.0f}W {team_b}"
        )

    # ── SHAP explanation ──

    with st.expander("🔬 Why did the model predict this?"):
        st.caption("Green = favours team A · Red = favours team B · Longer bar = bigger influence")

        FRIENDLY = {
            "elo_diff":                f"Elo gap ({team_a} vs {team_b})",
            "home_elo":                f"{team_a} — Elo rating",
            "away_elo":                f"{team_b} — Elo rating",
            "rank_diff":               f"FIFA rank gap",
            "home_form":               f"{team_a} — recent form",
            "away_form":               f"{team_b} — recent form",
            "form_diff":               f"Form gap ({team_a} vs {team_b})",
            "home_goals_scored_avg":   f"{team_a} — avg goals scored",
            "home_goals_conceded_avg": f"{team_a} — avg goals conceded",
            "away_goals_scored_avg":   f"{team_b} — avg goals scored",
            "away_goals_conceded_avg": f"{team_b} — avg goals conceded",
            "h2h_home_wins":           f"Head-to-head: {team_a} wins",
            "h2h_draws":               "Head-to-head: draws",
            "h2h_away_wins":           f"Head-to-head: {team_b} wins",
            "is_neutral":              "Neutral ground",
        }

        explainer   = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(X)
        sv          = shap_values[0, :, 2]   # impact toward team_a winning

        shap_df = pd.DataFrame({
            "Feature":    [FRIENDLY.get(c, c) for c in FEATURE_COLS],
            "Impact":     sv,
        }).sort_values("Impact", key=abs, ascending=True).tail(8)

        colors = ["#c1440e" if v < 0 else "#2d6a4f" for v in shap_df["Impact"]]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(shap_df["Feature"], shap_df["Impact"], color=colors, height=0.55)
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_xlabel(f"← Favours {team_b}          Favours {team_a} →", fontsize=9)
        ax.set_title(f"{team_a} vs {team_b} — prediction breakdown", fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

        green_patch = mpatches.Patch(color="#2d6a4f", label=f"Favours {team_a}")
        red_patch   = mpatches.Patch(color="#c1440e", label=f"Favours {team_b}")
        ax.legend(handles=[green_patch, red_patch], fontsize=8)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── Footer ─────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "XGBoost model · trained on 25,000+ international matches (2000–2021) · "
    "Elo ratings, FIFA rankings, recent form, head-to-head · ~62% test accuracy"
)