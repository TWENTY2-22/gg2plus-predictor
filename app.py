import glob
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

st.set_page_config(page_title="GG2+ Predictor", page_icon="⚽", layout="centered")

st.title("⚽ GG2+ Predictor")
st.markdown("**Both teams score at least 2 goals each**")
st.markdown("---")

LEAGUE_NAMES = {
    "E0": "England - Premier League", "E1": "England - Championship",
    "E2": "England - League One", "E3": "England - League Two",
    "SC0": "Scotland - Premiership", "D1": "Germany - Bundesliga",
    "D2": "Germany - 2. Bundesliga", "I1": "Italy - Serie A",
    "I2": "Italy - Serie B", "SP1": "Spain - La Liga",
    "SP2": "Spain - Segunda", "F1": "France - Ligue 1",
    "F2": "France - Ligue 2", "N1": "Netherlands - Eredivisie",
    "B1": "Belgium - Pro League", "P1": "Portugal - Primeira Liga",
    "T1": "Turkey - Super Lig", "G1": "Greece - Super League",
    "BRA": "Brazil - Serie A", "USA": "USA - MLS",
}

@st.cache_data
def load_data():
    frames = []
    for f in glob.glob("*.xlsx"):
        sheets = pd.read_excel(f, sheet_name=None)
        for sheet, d in sheets.items():
            d = d.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam",
                                  "HG": "FTHG", "AG": "FTAG"})
            need = ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]
            if not all(c in d.columns for c in need):
                continue
            d = d[need].copy()
            d["FTHG"] = pd.to_numeric(d["FTHG"], errors="coerce")
            d["FTAG"] = pd.to_numeric(d["FTAG"], errors="coerce")
            d = d.dropna()
            if d.empty:
                continue
            d["League"] = LEAGUE_NAMES.get(sheet, sheet)
            frames.append(d)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)

@st.cache_data
def build_strengths(df, league):
    d = df[df["League"] == league]
    avg_h = d["FTHG"].mean()
    avg_a = d["FTAG"].mean()
    avg_all = (avg_h + avg_a) / 2
    k = 4  # smoothing so teams with few games are not extreme
    att, dfn = {}, {}
    for t in set(d["HomeTeam"]) | set(d["AwayTeam"]):
        h = d[d["HomeTeam"] == t]
        a = d[d["AwayTeam"] == t]
        games = len(h) + len(a)
        scored = h["FTHG"].sum() + a["FTAG"].sum()
        conceded = h["FTAG"].sum() + a["FTHG"].sum()
        att[t] = (scored + k * avg_all) / (games + k) / avg_all
        dfn[t] = (conceded + k * avg_all) / (games + k) / avg_all
    return att, dfn, avg_h, avg_a

df = load_data()
if df is None:
    st.error("No data files found. Upload the .xlsx files to the repo.")
    st.stop()

league = st.selectbox("League", sorted(df["League"].unique()))
attack, defence, avg_h, avg_a = build_strengths(df, league)
teams = sorted(attack.keys())

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home Team", teams, index=0)
with col2:
    away_team = st.selectbox("Away Team", teams, index=min(1, len(teams) - 1))

def predict(home, away):
    lam_h = float(np.clip(attack[home] * defence[away] * avg_h, 0.35, 3.2))
    lam_a = float(np.clip(attack[away] * defence[home] * avg_a, 0.25, 2.8))
    p_h = 1 - poisson.pmf(0, lam_h) - poisson.pmf(1, lam_h)
    p_a = 1 - poisson.pmf(0, lam_a) - poisson.pmf(1, lam_a)
    return p_h * p_a, lam_h, lam_a

if st.button("Predict GG2+", type="primary", use_container_width=True):
    if home_team == away_team:
        st.warning("Please pick two different teams.")
    else:
        prob, lh, la = predict(home_team, away_team)
        st.markdown("### Result")
        st.metric("Probability both teams score ≥ 2 goals", f"{prob*100:.1f}%")
        c1, c2 = st.columns(2)
        c1.metric("Expected Home Goals", f"{lh:.2f}")
        c2.metric("Expected Away Goals", f"{la:.2f}")
        if prob >= 0.18:
            st.success("🔥 Relatively high chance for this rare market")
        elif prob >= 0.13:
            st.info("Average / slightly above average chance")
        else:
            st.warning("Below average chance")

st.markdown("---")
st.markdown("""
**What is GG2+?**  
Both teams score **at least 2 goals each** (examples: 2-2, 3-2, 2-3, 4-2, etc.)

**How the model works**  
It uses match results from the data files to estimate each team's attack and
defence strength inside its league, then calculates the probability with a Poisson model.
""")
