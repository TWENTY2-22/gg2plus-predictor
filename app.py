import glob
import difflib
import unicodedata
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import poisson

st.set_page_config(page_title="GG2+ Daily Picks", page_icon="⚽", layout="centered")
st.title("⚽ GG2+ Daily Picks")
st.markdown("**Top matches where both teams may score at least 2 goals each**")
st.markdown("---")

LEAGUE_NAMES = {
    "E0": "England - Premier League", "E1": "England - Championship",
    "E2": "England - League One", "E3": "England - League Two",
    "EC": "England - Conference", "SC0": "Scotland - Premiership",
    "SC1": "Scotland - Championship", "D1": "Germany - Bundesliga",
    "D2": "Germany - 2. Bundesliga", "I1": "Italy - Serie A",
    "I2": "Italy - Serie B", "SP1": "Spain - La Liga",
    "SP2": "Spain - Segunda", "F1": "France - Ligue 1",
    "F2": "France - Ligue 2", "N1": "Netherlands - Eredivisie",
    "B1": "Belgium - Pro League", "P1": "Portugal - Primeira Liga",
    "T1": "Turkey - Super Lig", "G1": "Greece - Super League",
    "BRA": "Brazil - Serie A", "USA": "USA - MLS",
}

# ESPN schedule codes for leagues missing from the football-data fixtures file
ESPN_LEAGUES = {"bra.1": "BRA", "usa.1": "USA"}

def clean(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return s.lower().strip()

@st.cache_data
def load_results():
    frames = []
    for f in glob.glob("*.xlsx"):
        for sheet, d in pd.read_excel(f, sheet_name=None).items():
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
            d["Div"] = sheet
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else None

@st.cache_data
def build_strengths(df):
    out = {}
    k = 4
    for div, d in df.groupby("Div"):
        avg_h, avg_a = d["FTHG"].mean(), d["FTAG"].mean()
        avg_all = (avg_h + avg_a) / 2
        if avg_all == 0:
            continue
        att, dfn = {}, {}
        for t in set(d["HomeTeam"]) | set(d["AwayTeam"]):
            h = d[d["HomeTeam"] == t]
            a = d[d["AwayTeam"] == t]
            games = len(h) + len(a)
            scored = h["FTHG"].sum() + a["FTAG"].sum()
            conceded = h["FTAG"].sum() + a["FTHG"].sum()
            att[t] = (scored + k * avg_all) / (games + k) / avg_all
            dfn[t] = (conceded + k * avg_all) / (games + k) / avg_all
        out[div] = (att, dfn, avg_h, avg_a)
    return out

def match_team(name, teams):
    """Match an ESPN team name to the closest name in our results data."""
    if name in teams:
        return name
    lookup = {clean(t): t for t in teams}
    c = clean(name)
    if c in lookup:
        return lookup[c]
    best = difflib.get_close_matches(c, list(lookup), n=1, cutoff=0.6)
    return lookup[best[0]] if best else None

@st.cache_data(ttl=3600)
def load_euro_fixtures():
    fx = pd.read_csv("https://football-data.co.uk/fixtures.csv", encoding="latin-1")
    fx = fx.dropna(subset=["HomeTeam", "AwayTeam"])
    fx["Date"] = pd.to_datetime(fx["Date"], dayfirst=True, errors="coerce")
    return fx.dropna(subset=["Date"])[["Div", "Date", "HomeTeam", "AwayTeam"]]

@st.cache_data(ttl=3600)
def load_espn_fixtures(dates):
    rows = []
    for code, div in ESPN_LEAGUES.items():
        for d in dates:
            url = ("https://site.api.espn.com/apis/site/v2/sports/soccer/"
                   f"{code}/scoreboard?dates={d}")
            try:
                data = requests.get(url, timeout=10).json()
            except Exception:
                continue
            for e in data.get("events", []):
                try:
                    if e["status"]["type"]["state"] != "pre":
                        continue
                    comp = e["competitions"][0]["competitors"]
                    home = next(c for c in comp if c["homeAway"] == "home")
                    away = next(c for c in comp if c["homeAway"] == "away")
                    rows.append({
                        "Div": div,
                        "Date": pd.Timestamp(d),
                        "HomeTeam": home["team"]["displayName"],
                        "AwayTeam": away["team"]["displayName"],
                    })
                except Exception:
                    continue
    return pd.DataFrame(rows, columns=["Div", "Date", "HomeTeam", "AwayTeam"])

def gg2_prob(home, away, s):
    att, dfn, avg_h, avg_a = s
    lam_h = float(np.clip(att[home] * dfn[away] * avg_h, 0.35, 3.2))
    lam_a = float(np.clip(att[away] * dfn[home] * avg_a, 0.25, 2.8))
    p_h = 1 - poisson.pmf(0, lam_h) - poisson.pmf(1, lam_h)
    p_a = 1 - poisson.pmf(0, lam_a) - poisson.pmf(1, lam_a)
    return p_h * p_a, lam_h, lam_a

results = load_results()
if results is None:
    st.error("No data files found. Upload the .xlsx files to the repo.")
    st.stop()
strengths = build_strengths(results)

choice = st.radio("Show picks for", ["Today", "Tomorrow", "Next 3 days"], horizontal=True)
today = pd.Timestamp.today().normalize()
if choice == "Today":
    start, end = today, today
elif choice == "Tomorrow":
    start = end = today + pd.Timedelta(days=1)
else:
    start, end = today, today + pd.Timedelta(days=3)

dates = tuple(d.strftime("%Y%m%d") for d in pd.date_range(start, end))

parts = []
try:
    euro = load_euro_fixtures()
    parts.append(euro[(euro["Date"] >= start) & (euro["Date"] <= end)])
except Exception:
    st.caption("Could not load the European fixtures right now.")
parts.append(load_espn_fixtures(dates))
fixtures = pd.concat(parts, ignore_index=True)

rows = []
for _, r in fixtures.iterrows():
    div = r["Div"]
    if div not in strengths:
        continue
    att = strengths[div][0]
    h = match_team(r["HomeTeam"], att.keys())
    a = match_team(r["AwayTeam"], att.keys())
    if h is None or a is None:
        continue
    prob, lh, la = gg2_prob(h, a, strengths[div])
    rows.append({
        "Date": r["Date"].strftime("%d %b"),
        "League": LEAGUE_NAMES.get(div, div),
        "Home": r["HomeTeam"],
        "Away": r["AwayTeam"],
        "GG2+ chance %": round(prob * 100, 1),
        "Exp. goals": f"{lh:.1f} - {la:.1f}",
    })

if not rows:
    st.warning("No matches found for this period. Try 'Next 3 days'.")
else:
    top = pd.DataFrame(rows).sort_values("GG2+ chance %", ascending=False).head(10)
    st.subheader(f"Top {len(top)} picks")
    st.dataframe(top, hide_index=True, use_container_width=True)
    st.caption(f"{len(rows)} matches checked.")

st.markdown("---")
st.markdown("""
**What is GG2+?**  
Both teams score **at least 2 goals each** (examples: 2-2, 3-2, 2-3, 4-2).

**How the picks are made**  
The app learns each team's attack and defence strength from past results,
estimates the goals for each upcoming match with a Poisson model, and ranks
matches by the chance of GG2+.
""")
