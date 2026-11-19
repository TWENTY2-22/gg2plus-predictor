import glob
import io
import os
import re
import json
import time
import tempfile
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

st.set_page_config(page_title="GG Predictor", page_icon="⚽", layout="centered")

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
HEADERS = {"User-Agent": "Mozilla/5.0"}
BUCKET = int(time.time() // 21600)
BASE_RATE = 1.35
HIST_PATH = os.path.join(tempfile.gettempdir(), "gg2plus_history.json")


# ---------------- Helpers ----------------
def clean(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


NOISE = {"fc", "wfc", "women", "womens", "ladies", "lionesses", "w", "afc",
         "cf", "sc", "club", "ac", "fk", "sk", "bk", "if", "the"}
WOMEN_RE = re.compile(r"\b(women|womens|wfc|ladies|lionesses|feminin\w*|frauen)\b", re.I)


def is_women(name):
    n = str(name).strip()
    return bool(WOMEN_RE.search(n)) or bool(re.search(r"\bW$", n))


def prob_k(lh, la, k):
    """Chance both teams score at least k goals (k=1 is GG, k=2 is GG2+)."""
    return float((1 - poisson.cdf(k - 1, lh)) * (1 - poisson.cdf(k - 1, la)))


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

team_index, div_index = {}, {}
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


def file_lams(h, a):
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
    return (float(np.clip(att_h * def_a * avg_h, 0.35, 3.2)),
            float(np.clip(att_a * def_h * avg_a, 0.25, 2.8)))


# ---------------- Live team lookup ----------------
def _walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)


def _score(x):
    if isinstance(x, dict):
        x = x.get("value", x.get("displayValue"))
    try:
        return float(x)
    except Exception:
        return None


def search_query(name):
    q = clean(name)
    q = re.sub(r"\b(wfc|women|womens|ladies|lionesses|w)\b", " ", q)
    q = re.sub(r"\b(fc|afc|sc|cf)\b", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def _women_name(n):
    n = str(n).lower()
    return "women" in n or bool(re.search(r"\bw$", n.strip()))


@lru_cache(maxsize=None)
def espn_team_id(name, women, bucket):
    q = search_query(name)
    if not q:
        return None
    tries = [q + (" women" if women else "")]
    if len(q.split()) > 2:
        tries.append(" ".join(q.split()[:2]) + (" women" if women else ""))
    best = (0.0, None)
    for query in tries:
        url = ("https://site.web.api.espn.com/apis/common/v3/search"
               f"?region=us&lang=en&limit=15&query={requests.utils.quote(query)}")
        try:
            data = requests.get(url, timeout=8, headers=HEADERS).json()
        except Exception:
            continue
        seen = set()
        for d in _walk(data):
            m = re.search(r"s:600~t:(\d+)", str(d.get("uid", "")))
            nm = d.get("displayName") or d.get("name")
            if not m or not nm or m.group(1) in seen:
                continue
            seen.add(m.group(1))
            if women != _women_name(nm):
                continue
            score = difflib.SequenceMatcher(None, search_query(nm), q).ratio()
            if score > best[0]:
                best = (score, m.group(1))
        if best[0] >= 0.6:
            break
    return best[1] if best[0] >= 0.6 else None


@lru_cache(maxsize=None)
def espn_recent(team_id, bucket):
    url = ("https://site.api.espn.com/apis/site/v2/sports/soccer/all/"
           f"teams/{team_id}/schedule")
    try:
        data = requests.get(url, timeout=8, headers=HEADERS).json()
    except Exception:
        return []
    games = []
    for e in data.get("events", []):
        try:
            comp = e["competitions"][0]
            stt = comp.get("status") or e.get("status") or {}
            t = stt.get("type", {})
            if not (t.get("completed") or t.get("state") == "post"):
                continue
            mine = opp = None
            for c in comp["competitors"]:
                s = _score(c.get("score"))
                if str(c["team"]["id"]) == str(team_id):
                    mine = s
                else:
                    opp = s
            if mine is None or opp is None:
                continue
            games.append((e.get("date", ""), mine, opp))
        except Exception:
            continue
    games.sort()
    return games[-12:]


@lru_cache(maxsize=None)
def live_stats(key, bucket):
    name, women = key
    tid = espn_team_id(name, women, bucket)
    if not tid:
        return None
    g = espn_recent(tid, bucket)
    if len(g) < 4:
        return None
    n, k = len(g), 3
    sc = sum(x[1] for x in g)
    co = sum(x[2] for x in g)
    return ((sc + k * BASE_RATE) / (n + k), (co + k * BASE_RATE) / (n + k), n)


def live_lams(sh, sa):
    return (float(np.clip((sh[0] + sa[1]) / 2 * 1.08, 0.35, 3.2)),
            float(np.clip((sa[0] + sh[1]) / 2 * 0.95, 0.25, 2.8)))


# ---------------- Fixtures ----------------
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


# ---------------- History ----------------
def load_history():
    try:
        with open(HIST_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def save_history(h):
    try:
        with open(HIST_PATH, "w") as f:
            json.dump(h, f)
    except Exception:
        pass


def add_history(entries):
    hist = load_history()
    keys = {e["key"] for e in hist}
    new = [e for e in entries if e["key"] not in keys]
    if new:
        save_history(hist + new)


def outcome(e):
    s = e.get("score")
    if not s:
        return "⏳"
    try:
        h, a = (int(x) for x in s.split("-"))
    except Exception:
        return "⏳"
    k = 2 if e["market"] == "GG2+" else 1
    return "✅" if h >= k and a >= k else "❌"


def check_results(hist):
    changed = 0
    today = pd.Timestamp.today().normalize()
    fresh = int(time.time())
    for e in hist:
        if e.get("score"):
            continue
        md = pd.Timestamp(e["match_date"])
        if md > today:
            continue
        women = is_women(e["home"]) or is_women(e["away"])
        tid = espn_team_id(e["home"], women, BUCKET)
        if not tid:
            continue
        for date, mine, opp in espn_recent(tid, fresh):
            try:
                gd = pd.to_datetime(date, utc=True).tz_convert(None).normalize()
            except Exception:
                continue
            if md <= gd <= md + pd.Timedelta(days=4):
                e["score"] = f"{int(mine)}-{int(opp)}"
                changed += 1
                break
    return changed


# ---------------- Slip reading ----------------
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


# ---------------- Sidebar settings ----------------
with st.sidebar:
    st.header("Settings")
    min2 = st.slider("GG2+ counts as strong from (%)", 8, 30, 14)
    min1 = st.slider("GG counts as strong from (%)", 30, 80, 50)
MARKETS = {"GG2+": (2, min2), "GG": (1, min1)}
CHANCE_CFG = {"Chance %": st.column_config.ProgressColumn(
    "Chance", min_value=0, max_value=100, format="%.1f%%")}


def market_table(df, mkt):
    k, mn = MARKETS[mkt]
    out = df.copy()
    out["Chance %"] = [round(prob_k(a, b, k) * 100, 1)
                       for a, b in zip(out["lh"], out["la"])]
    out["Verdict"] = np.where(out["Chance %"] >= mn, "Strong", "Weak")
    out["Exp. goals"] = [f"{a:.1f} - {b:.1f}" for a, b in zip(out["lh"], out["la"])]
    return out.sort_values("Chance %", ascending=False)


# ---------------- Page ----------------
st.title("⚽ GG Predictor")
tab_slip, tab_daily, tab_hist = st.tabs(["📷 My slip", "📅 Daily picks", "🕘 History"])

# ======== My slip ========
with tab_slip:
    ups = st.file_uploader("Upload slip pictures", type=["png", "jpg", "jpeg"],
                           accept_multiple_files=True)
    pasted = st.text_area("Or paste matches (Arsenal v Chelsea, one per line)", height=90)

    texts = [pasted] if pasted.strip() else []
    if ups:
        if pytesseract is None:
            st.warning("Picture reading is not installed. Paste the list instead.")
        else:
            fails = []
            for f in ups:
                try:
                    texts.append(ocr_image(f.getvalue()))
                except Exception as e:
                    fails.append(f"{f.name} ({type(e).__name__})")
            if fails:
                st.warning("Could not read: " + ", ".join(fails))

    if texts:
        parsed, seen_pairs = [], set()
        for t in texts:
            for pair in parse_matches(t):
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    parsed.append(pair)

        with st.expander(f"{len(parsed)} matches found (fix mistakes here)"):
            base = pd.DataFrame(parsed, columns=["Home", "Away"])
            edited = st.data_editor(base, num_rows="dynamic", use_container_width=True,
                                    key=f"ed_{len(parsed)}_{len(ups) if ups else 0}")
            if st.checkbox("Show raw text read from the pictures"):
                st.text("\n\n-----\n\n".join(texts))

        pairs = [(str(r["Home"]), str(r["Away"])) for _, r in edited.dropna().iterrows()]
        keys = sorted({(n, is_women(h) or is_women(a)) for h, a in pairs for n in (h, a)})
        with st.spinner("Looking up each team..."):
            with ThreadPoolExecutor(8) as ex:
                stats = dict(zip(keys, ex.map(lambda k: live_stats(k, BUCKET), keys)))

        rows, unrated = [], []
        for hn, an in pairs:
            women = is_women(hn) or is_women(an)
            sh, sa = stats.get((hn, women)), stats.get((an, women))
            if sh and sa:
                lh, la = live_lams(sh, sa)
                src = "Live results"
            else:
                h = None if women else loose_find(hn)
                a = None if women else loose_find(an)
                if h and a and h != a:
                    lh, la = file_lams(h, a)
                    src = "League files"
                else:
                    unrated.append(f"{hn} v {an}")
                    continue
            rows.append({"Home": hn, "Away": an, "lh": lh, "la": la, "Data": src})

        if not rows:
            st.warning("None of the matches could be rated. Check the spelling above.")
        else:
            df_slip = pd.DataFrame(rows)
            t2, t1 = st.tabs(["🎯 GG2+ · best 7", "⚽ GG · best 7"])
            today_iso = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
            entries = []
            for tab, mkt in ((t2, "GG2+"), (t1, "GG")):
                with tab:
                    top = market_table(df_slip, mkt).head(7)
                    st.dataframe(top[["Home", "Away", "Chance %", "Exp. goals",
                                      "Verdict", "Data"]],
                                 hide_index=True, use_container_width=True,
                                 column_config=CHANCE_CFG)
                    if (top["Verdict"] == "Strong").sum() == 0:
                        st.info("None reach your minimum chance. Skipping is reasonable.")
                for _, r in top.iterrows():
                    entries.append({
                        "key": f"slip|{mkt}|{today_iso}|{r['Home']}|{r['Away']}",
                        "saved": today_iso, "match_date": today_iso, "source": "Slip",
                        "market": mkt, "home": r["Home"], "away": r["Away"],
                        "chance": float(r["Chance %"]), "score": None})
            add_history(entries)
            st.caption(f"{len(rows)} of {len(pairs)} matches rated. Saved to History.")
        if unrated:
            with st.expander(f"{len(unrated)} not rated"):
                st.write("No recent results were found for one of the teams.")
                st.write("\n".join(f"- {u}" for u in unrated))

# ======== Daily picks ========
with tab_daily:
    choice = st.radio("Period", ["Today", "Tomorrow", "Next 3 days", "Next 7 days"],
                      horizontal=True, label_visibility="collapsed")
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
        st.caption("European fixtures could not be loaded right now.")
    fx_rows += load_espn(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))

    drows, skipped, counts = [], [], {}
    for r in fx_rows:
        comp = LEAGUE_NAMES.get(r["Comp"], r["Comp"])
        counts.setdefault(comp, [0, 0])
        counts[comp][0] += 1
        if r["Cup"]:
            h = find_team(r["HomeTeam"], None, 0.85)
            a = find_team(r["AwayTeam"], None, 0.85)
        elif r["Div"] in strengths:
            h = find_team(r["HomeTeam"], r["Div"], 0.7)
            a = find_team(r["AwayTeam"], r["Div"], 0.7)
            if h or a:
                h = h or (r["Div"], None)
                a = a or (r["Div"], None)
        else:
            h = a = None
        if not h or not a or h == a:
            skipped.append(f"{comp}: {r['HomeTeam']} v {r['AwayTeam']}")
            continue
        lh, la = file_lams(h, a)
        counts[comp][1] += 1
        drows.append({"Date": r["Date"].strftime("%d %b"),
                      "iso": r["Date"].strftime("%Y-%m-%d"),
                      "Competition": comp, "Home": r["HomeTeam"], "Away": r["AwayTeam"],
                      "lh": lh, "la": la})

    if not drows:
        st.info("No rated matches for this period. Try a longer period.")
    else:
        df_d = pd.DataFrame(drows)
        d2, d1 = st.tabs(["🎯 GG2+ · top 10", "⚽ GG · top 10"])
        entries = []
        for tab, mkt in ((d2, "GG2+"), (d1, "GG")):
            _, mn = MARKETS[mkt]
            with tab:
                full = market_table(df_d, mkt)
                strong = full[full["Chance %"] >= mn].head(10)
                if strong.empty:
                    st.info(f"No strong {mkt} matches for this period. "
                            f"The highest is {full['Chance %'].iloc[0]}%. "
                            "Better to skip than force a pick.")
                else:
                    st.dataframe(strong[["Date", "Competition", "Home", "Away",
                                         "Chance %", "Exp. goals"]],
                                 hide_index=True, use_container_width=True,
                                 column_config=CHANCE_CFG)
                for _, r in strong.iterrows():
                    entries.append({
                        "key": f"daily|{mkt}|{r['iso']}|{r['Home']}|{r['Away']}",
                        "saved": today.strftime("%Y-%m-%d"), "match_date": r["iso"],
                        "source": "Daily", "market": mkt, "home": r["Home"],
                        "away": r["Away"], "chance": float(r["Chance %"]),
                        "score": None})
        add_history(entries)
        st.caption(f"{len(drows)} matches rated.")

    with st.expander("Leagues checked"):
        if counts:
            st.dataframe(pd.DataFrame(
                [{"Competition": k, "Found": v[0], "Rated": v[1]}
                 for k, v in sorted(counts.items())]),
                hide_index=True, use_container_width=True)
        else:
            st.write("No fixtures returned for these dates.")
        if skipped:
            st.write("Not rated:")
            st.write("\n".join(f"- {s}" for s in skipped[:40]))

# ======== History ========
with tab_hist:
    hist = load_history()
    c1, c2 = st.columns(2)
    if c1.button("🔄 Check results", use_container_width=True):
        with st.spinner("Looking up final scores..."):
            n = check_results(hist)
            save_history(hist)
        st.success(f"{n} results found.")
        hist = load_history()
    if hist:
        hdf = pd.DataFrame(hist)
        hdf["Result"] = hdf["score"].fillna("")
        hdf["Outcome"] = [outcome(e) for e in hist]
        hdf["Match"] = hdf["home"] + " v " + hdf["away"]
        hdf = hdf.sort_values(["match_date", "chance"], ascending=[False, False])

        m1, m2 = st.columns(2)
        for col, mkt in ((m1, "GG2+"), (m2, "GG")):
            sub = hdf[(hdf["market"] == mkt) & (hdf["Outcome"] != "⏳")]
            hits = int((sub["Outcome"] == "✅").sum())
            col.metric(f"{mkt} hit rate",
                       f"{hits}/{len(sub)}" if len(sub) else "–",
                       f"{hits / len(sub) * 100:.0f}%" if len(sub) else None,
                       delta_color="off")

        f_mkt = st.radio("Show", ["All", "GG2+", "GG"], horizontal=True)
        view = hdf if f_mkt == "All" else hdf[hdf["market"] == f_mkt]
        st.dataframe(view.rename(columns={"match_date": "Date", "market": "Market",
                                          "source": "From", "chance": "Chance %"})[
            ["Date", "Market", "From", "Match", "Chance %", "Result", "Outcome"]],
            hide_index=True, use_container_width=True, column_config=CHANCE_CFG)
        c2.download_button("⬇️ Download", hdf.to_csv(index=False).encode(),
                           "gg_history.csv", "text/csv", use_container_width=True)
    else:
        st.info("Nothing saved yet. Picks and slip results appear here automatically.")

    with st.expander("Restore saved history"):
        up_h = st.file_uploader("Upload a downloaded history file (.csv)", type=["csv"],
                                key="hist_up")
        if up_h is not None:
            try:
                old = pd.read_csv(up_h)
                recs = []
                for _, r in old.iterrows():
                    recs.append({
                        "key": str(r["key"]), "saved": str(r["saved"]),
                        "match_date": str(r["match_date"]), "source": str(r["source"]),
                        "market": str(r["market"]), "home": str(r["home"]),
                        "away": str(r["away"]), "chance": float(r["chance"]),
                        "score": None if pd.isna(r.get("score")) else str(r["score"])})
                add_history(recs)
                st.success(f"{len(recs)} rows restored. Reload the page to see them.")
            except Exception:
                st.error("That file could not be read.")
    st.caption("History is kept on the app server. It is cleared when the app "
               "restarts or when you change the code. Download it regularly.")

with st.expander("How it works"):
    st.markdown("""
**GG2+** means both teams score at least 2 goals each. **GG** means both teams
score at least 1. The app estimates each team's goals from recent results with a
Poisson model, then works out the chance of each result. These are chances, not
guarantees.
""")
