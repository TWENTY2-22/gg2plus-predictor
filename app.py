import glob
import io
import re
import difflib
import unicodedata
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from scipy.stats import poisson

try:
    import pytesseract
except Exception:
    pytesseract = None

st.set_page_config(page_title="GG2+ Daily Picks", page_icon="⚽", layout="centered")
st.title("⚽ GG2+ Daily Picks")
st.markdown("**Matches where both teams may score at least 2 goals each**")
st.markdown("---")

LEAGUE_NAMES = {
    "E0": "England - Premier League", "E1": "England - Championship",
    "E2": "England - League One", "E3": "England - League Two",
    "EC": "England - Conference", "SC0": "Scotland - Premiership",
    "SC1": "Scotland - Championship", "SC2": "Scotland - League One",
    "SC3": "Scotland - League Two", "D1": "Germany - Bundesliga",
    "D2": "Germany - 2. Bundesliga", "I1": "Italy - Serie A",
    "I2": "Italy - Serie B", "SP1": "Spain - La Liga",
    "SP2": "Spain - Segunda", "F1": "France - Ligue 1",
    "F2": "France - Ligue 2", "N1": "Netherlands - Eredivisie",
    "B1": "Belgium - Pro League", "P1": "Portugal - Primeira Liga",
    "T1": "Turkey - Super Lig", "G1": "Greece - Super League",
    "Brazil": "Brazil - Serie A", "USA": "USA - MLS",
}

ESPN_LEAGUES = {
    "bra.1": "Brazil", "usa.1": "USA", "swe.1": "Sweden", "nor.1": "Norway",
    "den.1": "Denmark", "sui.1": "Switzerland", "aut.1": "Austria",
    "pol.1": "Poland", "rou.1": "Romania", "rus.1": "Russia",
    "fin.1": "Finland", "irl.1": "Ireland", "arg.1": "Argentina",
    "mex.1": "Mexico", "jpn.1": "Japan", "chn.1": "China",
}
ESPN_CUPS = {
    "eng.fa": "FA Cup", "eng.league_cup": "League Cup",
    "esp.copa_del_rey": "Copa del Rey", "ger.dfb_pokal": "DFB-Pokal",
    "ita.coppa_italia": "Coppa Italia", "fra.coupe_de_france": "Coupe de France",
    "uefa.champions": "Champions League", "uefa.europa": "Europa League",
    "uefa.europa.conf": "Conference League",
}


def clean(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


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
            keep = need + [c for c in ("Season", "Country") if c in d.columns]
            d = d[keep].copy()
            d["FTHG"] = pd.to_numeric(d["FTHG"], errors="coerce")
            d["FTAG"] = pd.to_numeric(d["FTAG"], errors="coerce")
            d = d.dropna(subset=need)
            if d.empty:
                continue
            d["Div"] = d["Country"].astype(str) if "Country" in d.columns else sheet
            if "Season" in d.columns:
                d["Season"] = d["Season"].astype(str)
                d = d[d.groupby("Div")["Season"].transform(
                    lambda s: s.isin(sorted(s.unique())[-2:]))]
            frames.append(d[need + ["Div"]])
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


results = load_results()
if results is None:
    st.error("No data files found. Upload the .xlsx files to the repo.")
    st.stop()
strengths = build_strengths(results)

team_index = {}
div_index = {}
for _div, (_att, _dfn, _h, _a) in strengths.items():
    div_index[_div] = {}
    for _t in _att:
        div_index[_div][clean(_t)] = (_div, _t)
        team_index.setdefault(clean(_t), (_div, _t))


@lru_cache(maxsize=None)
def find_team(name, div=None, cutoff=0.75):
    c = clean(name)
    if not c:
        return None
    pool = team_index if div is None else div_index.get(div, {})
    if c in pool:
        return pool[c]
    best = difflib.get_close_matches(c, list(pool), n=1, cutoff=cutoff)
    return pool[best[0]] if best else None


def pair_prob(h, a):
    hd, ht = h
    ad, at = a
    att_h = strengths[hd][0].get(ht, 1.0)
    def_h = strengths[hd][1].get(ht, 1.0)
    att_a = strengths[ad][0].get(at, 1.0)
    def_a = strengths[ad][1].get(at, 1.0)
    if hd == ad:
        avg_h, avg_a = strengths[hd][2], strengths[hd][3]
    else:
        avg_h = (strengths[hd][2] + strengths[ad][2]) / 2
        avg_a = (strengths[hd][3] + strengths[ad][3]) / 2
    lam_h = float(np.clip(att_h * def_a * avg_h, 0.35, 3.2))
    lam_a = float(np.clip(att_a * def_h * avg_a, 0.25, 2.8))
    p_h = 1 - poisson.pmf(0, lam_h) - poisson.pmf(1, lam_h)
    p_a = 1 - poisson.pmf(0, lam_a) - poisson.pmf(1, lam_a)
    return p_h * p_a, lam_h, lam_a


@st.cache_data(ttl=3600)
def load_euro_fixtures():
    fx = pd.read_csv("https://football-data.co.uk/fixtures.csv", encoding="latin-1")
    fx = fx.dropna(subset=["HomeTeam", "AwayTeam"])
    fx["Date"] = pd.to_datetime(fx["Date"], dayfirst=True, errors="coerce")
    return fx.dropna(subset=["Date"])[["Div", "Date", "HomeTeam", "AwayTeam"]]


@st.cache_data(ttl=1800)
def load_espn(start, end):
    rng = start if start == end else f"{start}-{end}"
    jobs = [(c, n, False) for c, n in ESPN_LEAGUES.items()] + \
           [(c, n, True) for c, n in ESPN_CUPS.items()]

    def fetch(job):
        code, label, is_cup = job
        url = ("https://site.api.espn.com/apis/site/v2/sports/soccer/"
               f"{code}/scoreboard?dates={rng}&limit=300")
        rows = []
        try:
            data = requests.get(url, timeout=8).json()
        except Exception:
            return rows
        for e in data.get("events", []):
            try:
                if e["status"]["type"]["state"] != "pre":
                    continue
                comp = e["competitions"][0]["competitors"]
                home = next(c for c in comp if c["homeAway"] == "home")
                away = next(c for c in comp if c["homeAway"] == "away")
                date = pd.to_datetime(e["date"], utc=True).tz_convert(None).normalize()
                rows.append({
                    "Div": label, "Comp": label, "Cup": is_cup, "Date": date,
                    "HomeTeam": home["team"]["displayName"],
                    "AwayTeam": away["team"]["displayName"],
                })
            except Exception:
                continue
        return rows

    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(fetch, jobs))
    return [r for part in out for r in part]


# ---------------- Daily picks ----------------
st.header("📅 Daily picks")
choice = st.radio("Show picks for", ["Today", "Tomorrow", "Next 3 days", "Next 7 days"],
                  horizontal=True)
min_prob = st.slider("Minimum GG2+ chance (%)", 8, 30, 14)

today = pd.Timestamp.today().normalize()
if choice == "Today":
    start, end = today, today
elif choice == "Tomorrow":
    start = end = today + pd.Timedelta(days=1)
elif choice == "Next 3 days":
    start, end = today, today + pd.Timedelta(days=3)
else:
    start, end = today, today + pd.Timedelta(days=7)

fx_rows = []
try:
    euro = load_euro_fixtures()
    euro = euro[(euro["Date"] >= start) & (euro["Date"] <= end)]
    for _, r in euro.iterrows():
        fx_rows.append({"Div": r["Div"], "Comp": LEAGUE_NAMES.get(r["Div"], r["Div"]),
                        "Cup": False, "Date": r["Date"],
                        "HomeTeam": r["HomeTeam"], "AwayTeam": r["AwayTeam"]})
except Exception:
    st.caption("Could not load the European fixtures right now.")
fx_rows += load_espn(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))

scored, skipped, counts = [], [], {}
for r in fx_rows:
    comp = LEAGUE_NAMES.get(r["Comp"], r["Comp"])
    counts.setdefault(comp, [0, 0])
    counts[comp][0] += 1
    note = ""
    if r["Cup"]:
        h = find_team(r["HomeTeam"], None, 0.85)
        a = find_team(r["AwayTeam"], None, 0.85)
    elif r["Div"] in strengths:
        h = find_team(r["HomeTeam"], r["Div"], 0.7)
        a = find_team(r["AwayTeam"], r["Div"], 0.7)
        if h or a:
            if not h:
                h, note = (r["Div"], None), "home team estimated"
            if not a:
                a, note = (r["Div"], None), "away team estimated"
    else:
        h = a = None
    if not h or not a or h == a:
        skipped.append(f"{comp}: {r['HomeTeam']} v {r['AwayTeam']}")
        continue
    prob, lh, la = pair_prob(h, a)
    counts[comp][1] += 1
    scored.append({
        "Date": r["Date"].strftime("%d %b"),
        "Competition": comp,
        "Home": r["HomeTeam"], "Away": r["AwayTeam"],
        "GG2+ chance %": round(prob * 100, 1),
        "Exp. goals": f"{lh:.1f} - {la:.1f}",
        "Note": note,
    })

if not scored:
    st.warning("No matches could be rated for this period. Try 'Next 7 days'.")
else:
    df_s = pd.DataFrame(scored).sort_values("GG2+ chance %", ascending=False)
    strong = df_s[df_s["GG2+ chance %"] >= min_prob].head(10)
    if strong.empty:
        st.info(f"No strong GG2+ matches for this period. None reach {min_prob}%. "
                f"The highest is {df_s['GG2+ chance %'].iloc[0]}%. "
                "Better to skip than force a pick.")
    else:
        st.subheader(f"Top {len(strong)} picks")
        st.dataframe(strong, hide_index=True, use_container_width=True)
    st.caption(f"{len(scored)} matches rated, {len(skipped)} skipped.")

with st.expander("🔍 Diagnostics: which leagues were found"):
    if counts:
        diag = pd.DataFrame(
            [{"Competition": k, "Matches found": v[0], "Matches rated": v[1]}
             for k, v in sorted(counts.items())])
        st.dataframe(diag, hide_index=True, use_container_width=True)
    else:
        st.write("No fixtures were returned by any source for these dates.")
    if skipped:
        st.write("Skipped matches:")
        st.write("\n".join(f"- {s}" for s in skipped[:40]))
    st.caption("Leagues not on this list have no matches in the period "
               "(international break or off-season), or the source has no feed for them.")

# ---------------- Pictures / pasted list ----------------
st.markdown("---")
st.header("📷 Check your own list")
st.markdown("Upload one or more screenshots of matches, or paste the list. "
            "The app finds the matches in all of them and ranks them together.")

NOISE = {"fc", "wfc", "women", "womens", "ladies", "lionesses", "w", "afc",
         "cf", "sc", "club", "ac", "fk", "sk", "bk", "if", "the"}
WOMEN_RE = re.compile(r"\b(women|womens|wfc|ladies|lionesses|feminin\w*|frauen)\b", re.I)


def is_women(name):
    n = str(name).strip()
    return bool(WOMEN_RE.search(n)) or bool(re.search(r"\bW$", n))


def loose_find(name):
    toks = [t for t in clean(name).split() if t not in NOISE]
    c2 = " ".join(toks)
    if len(c2) < 3:
        return None
    t = find_team(c2, None, 0.8)
    if t:
        return t
    padded = f" {c2} "
    best = None
    for key, val in team_index.items():
        if len(key) >= 4 and f" {key} " in padded:
            if best is None or len(key) > len(best[0]):
                best = (key, val)
    return best[1] if best else None


@st.cache_data
def ocr_image(data: bytes):
    im = Image.open(io.BytesIO(data)).convert("L")
    im = im.resize((im.width * 2, im.height * 2))
    return pytesseract.image_to_string(im, config="--psm 6")


VS = re.compile(r"\s+(?:vs\.?|v)\s+|\s+[-–—]\s+", re.I)


def parse_matches(txt):
    out = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = VS.split(line, maxsplit=1)
        if len(parts) != 2:
            continue
        h, a = parts
        h = re.sub(r"^[^A-Za-z]*(?:yes|no)\b[\s:.\-]*", "", h, flags=re.I)
        h = re.sub(r"[^\w&' .\-]+", " ", h).strip()
        a = re.sub(r"\b(?:going|gg|goals?)\b.*$", "", a, flags=re.I)
        a = re.sub(r"\d+[.,]\d+\s*$", "", a)
        a = re.sub(r"[^\w&' .\-]+", " ", a).strip()
        if len(h) >= 3 and len(a) >= 3:
            out.append((h, a))
    return out


ups = st.file_uploader("Upload pictures of the matches (you can pick several)",
                       type=["png", "jpg", "jpeg"], accept_multiple_files=True)
pasted = st.text_area("Or paste the list here (one match per line, like: Arsenal v Chelsea)",
                      height=120)

texts = [pasted] if pasted.strip() else []
if ups:
    if pytesseract is None:
        st.warning("Picture reading is not installed yet. Paste the list instead.")
    else:
        read_ok, read_fail = 0, []
        for f in ups:
            try:
                texts.append(ocr_image(f.getvalue()))
                read_ok += 1
            except Exception as e:
                read_fail.append(f"{f.name} ({type(e).__name__}: {e})")
        st.caption(f"{read_ok} of {len(ups)} pictures read.")
        if read_fail:
            st.warning("Could not read: " + ", ".join(read_fail))

if texts:
    with st.expander("What the app read from the pictures"):
        st.text("\n\n-----\n\n".join(texts))

    parsed, seen = [], set()
    for t in texts:
        for pair in parse_matches(t):
            if pair not in seen:
                seen.add(pair)
                parsed.append(pair)
    st.caption(f"{len(parsed)} matches found. Fix any mistakes or add rows below.")
    base = pd.DataFrame(parsed, columns=["Home", "Away"])
    edited = st.data_editor(base, num_rows="dynamic", use_container_width=True,
                            key=f"ed_{len(parsed)}_{len(ups) if ups else 0}")

    res, unrated = [], []
    for _, r in edited.dropna().iterrows():
        hn, an = str(r["Home"]), str(r["Away"])
        if is_women(hn) or is_women(an):
            unrated.append(f"{hn} v {an}: women's match, no data")
            continue
        h, a = loose_find(hn), loose_find(an)
        note = ""
        if h and not a:
            a, note = (h[0], None), "away team estimated"
        elif a and not h:
            h, note = (a[0], None), "home team estimated"
        if not h or not a or h == a:
            unrated.append(f"{hn} v {an}: teams not in the data files")
            continue
        prob, lh, la = pair_prob(h, a)
        res.append({"Home": hn, "Away": an,
                    "GG2+ chance %": round(prob * 100, 1),
                    "Exp. goals": f"{lh:.1f} - {la:.1f}",
                    "Note": note})
    if res:
        df_r = pd.DataFrame(res).sort_values("GG2+ chance %", ascending=False)
        best = df_r[df_r["GG2+ chance %"] >= min_prob].head(8)
        if best.empty:
            st.info(f"None of these matches reach {min_prob}%. "
                    "No strong GG2+ pick in this list.")
        else:
            st.subheader(f"Best {len(best)} from your list")
            st.dataframe(best, hide_index=True, use_container_width=True)
        with st.expander("All rated matches"):
            st.dataframe(df_r, hide_index=True, use_container_width=True)
    if unrated:
        with st.expander(f"Not rated ({len(unrated)})"):
            st.write("\n".join(f"- {u}" for u in unrated))

st.markdown("---")
st.markdown("""
**What is GG2+?**  
Both teams score **at least 2 goals each** (examples: 2-2, 3-2, 2-3, 4-2).

**How the picks are made**  
The app learns each team's attack and defence strength from past results,
estimates the goals for each match with a Poisson model, and shows only
matches above your minimum chance.
""")
