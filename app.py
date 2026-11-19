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

# code: (competition name, results-file division, is_cup)
ESPN_COMPS = {
    "eng.1": ("England - Premier League", "E0", False),
    "eng.2": ("England - Championship", "E1", False),
    "eng.3": ("England - League One", "E2", False),
    "eng.4": ("England - League Two", "E3", False),
    "eng.5": ("England - National League", "EC", False),
    "sco.1": ("Scotland - Premiership", "SC0", False),
    "sco.2": ("Scotland - Championship", "SC1", False),
    "ger.1": ("Germany - Bundesliga", "D1", False),
    "ger.2": ("Germany - 2. Bundesliga", "D2", False),
    "ita.1": ("Italy - Serie A", "I1", False),
    "ita.2": ("Italy - Serie B", "I2", False),
    "esp.1": ("Spain - La Liga", "SP1", False),
    "esp.2": ("Spain - Segunda", "SP2", False),
    "fra.1": ("France - Ligue 1", "F1", False),
    "fra.2": ("France - Ligue 2", "F2", False),
    "ned.1": ("Netherlands - Eredivisie", "N1", False),
    "bel.1": ("Belgium - Pro League", "B1", False),
    "por.1": ("Portugal - Primeira Liga", "P1", False),
    "tur.1": ("Turkey - Super Lig", "T1", False),
    "gre.1": ("Greece - Super League", "G1", False),
    "bra.1": ("Brazil - Serie A", "Brazil", False),
    "usa.1": ("USA - MLS", "USA", False),
    "swe.1": ("Sweden - Allsvenskan", "Sweden", False),
    "nor.1": ("Norway - Eliteserien", "Norway", False),
    "den.1": ("Denmark - Superliga", "Denmark", False),
    "sui.1": ("Switzerland - Super League", "Switzerland", False),
    "aut.1": ("Austria - Bundesliga", "Austria", False),
    "pol.1": ("Poland - Ekstraklasa", "Poland", False),
    "rou.1": ("Romania - Liga 1", "Romania", False),
    "rus.1": ("Russia - Premier League", "Russia", False),
    "fin.1": ("Finland - Veikkausliiga", "Finland", False),
    "irl.1": ("Ireland - Premier Division", "Ireland", False),
    "arg.1": ("Argentina - Liga", "Argentina", False),
    "mex.1": ("Mexico - Liga MX", "Mexico", False),
    "jpn.1": ("Japan - J1 League", "Japan", False),
    "chn.1": ("China - Super League", "China", False),
    "eng.fa": ("FA Cup", None, True),
    "eng.league_cup": ("League Cup", None, True),
    "esp.copa_del_rey": ("Copa del Rey", None, True),
    "ger.dfb_pokal": ("DFB-Pokal", None, True),
    "ita.coppa_italia": ("Coppa Italia", None, True),
    "fra.coupe_de_france": ("Coupe de France", None, True),
    "uefa.champions": ("Champions League", None, True),
    "uefa.europa": ("Europa League", None, True),
    "uefa.europa.conf": ("Conference League", None, True),
}
# Extra team lists used only to find teams from your slips. A wrong code returns nothing.
EXTRA_DIR_CODES = [
    "eng.w.1", "esp.w.1", "ger.w.1", "fra.w.1", "ita.w.1", "usa.w.1",
    "uefa.w.champions", "ned.2", "ger.3", "por.2", "tur.2", "bel.2", "sco.3",
    "qat.1", "ksa.1", "uae.1", "idn.1", "phi.1", "ind.1", "aus.1", "kor.1",
    "egy.1", "rsa.1", "col.1", "chi.1", "uru.1", "per.1", "ecu.1", "par.1",
    "bol.1", "ven.1", "cro.1", "cze.1", "hun.1", "ukr.1", "isr.1", "ser.1",
    "bul.1", "svk.1", "slo.1", "nir.1", "wal.1", "isl.1", "cyp.1", "kaz.1",
    "tha.1", "mys.1", "sgp.1", "vnm.1", "hkg.1", "nzl.1", "crc.1", "gua.1",
    "hon.1", "pan.1", "jam.1", "mar.1", "tun.1", "alg.1", "nga.1", "gha.1",
    "ger.4", "esp.3", "ita.3", "fra.3", "eng.6",
]
HEADERS = {"User-Agent": "Mozilla/5.0"}
BUCKET = int(time.time() // 21600)   # team data: refreshed every 6 hours
RB = int(time.time() // 600)         # results: refreshed every 10 minutes
BASE_RATE = 1.35
HIST_PATH = os.path.join(tempfile.gettempdir(), "gg_history_v2.json")
ONE_DAY = pd.Timedelta(days=1)


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


def result_icon(score, k):
    if not score:
        return "–"
    try:
        h, a = (int(x) for x in str(score).split("-"))
    except Exception:
        return "–"
    return "✅" if h >= k and a >= k else "❌"


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


# ---------------- Live team lookup (ESPN) ----------------
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


def name_score(a, b):
    n, q = search_query(a), search_query(b)
    if not n or not q:
        return 0.0
    s = difflib.SequenceMatcher(None, n, q).ratio()
    if min(len(q), len(n)) >= 4 and (q in n or n in q):
        s = max(s, 0.85)
    return s


def best_match(q, cands):
    best = (0.0, None)
    for tid, nm in cands:
        s = name_score(nm, q)
        if s > best[0]:
            best = (s, tid)
    return best


@lru_cache(maxsize=None)
def directory(bucket):
    """All teams from ESPN league team lists: list of (id, name, is_women)."""
    codes = list(ESPN_COMPS.keys()) + EXTRA_DIR_CODES

    def fetch(code):
        url = ("https://site.api.espn.com/apis/site/v2/sports/soccer/"
               f"{code}/teams?limit=1000")
        try:
            data = requests.get(url, timeout=8, headers=HEADERS).json()
        except Exception:
            return []
        w = ".w." in code
        out = []
        for d in _walk(data):
            if d.get("id") and d.get("displayName") and d.get("slug"):
                out.append((str(d["id"]), str(d["displayName"]), w))
        return out

    with ThreadPoolExecutor(8) as ex:
        parts = list(ex.map(fetch, codes))
    seen, res = set(), []
    for part in parts:
        for t in part:
            if (t[0], t[2]) not in seen:
                seen.add((t[0], t[2]))
                res.append(t)
    return res


@lru_cache(maxsize=None)
def espn_team_id(name, women, bucket):
    """Backup: ESPN site search."""
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
        cands, seen = [], set()
        for d in _walk(data):
            m = re.search(r"s:600~t:(\d+)", str(d.get("uid", "")))
            nm = d.get("displayName") or d.get("name")
            if m and nm and m.group(1) not in seen and women == _women_name(nm):
                seen.add(m.group(1))
                cands.append((m.group(1), str(nm)))
        s, tid = best_match(q, cands)
        if s > best[0]:
            best = (s, tid)
        if best[0] >= 0.6:
            break
    return best[1] if best[0] >= 0.6 else None


@lru_cache(maxsize=None)
def espn_recent(team_id, bucket):
    """Completed games of a team: (date, goals_for, goals_against, opp_id, opp_name)."""
    url = ("https://site.api.espn.com/apis/site/v2/sports/soccer/all/"
           f"teams/{team_id}/schedule")
    try:
        data = requests.get(url, timeout=8, headers=HEADERS).json()
    except Exception:
        return []
    now = pd.Timestamp.utcnow().tz_localize(None)
    games = []
    for e in data.get("events", []):
        try:
            comp = e["competitions"][0]
            stt = comp.get("status") or e.get("status") or {}
            t = stt.get("type", {})
            mine = opp = None
            oid, oname = "", ""
            for c in comp["competitors"]:
                s = _score(c.get("score"))
                if str(c["team"]["id"]) == str(team_id):
                    mine = s
                else:
                    opp = s
                    oid = str(c["team"].get("id", ""))
                    oname = str(c["team"].get("displayName", ""))
            if mine is None or opp is None:
                continue
            when = pd.to_datetime(e.get("date", ""), utc=True).tz_convert(None)
            done = bool(t.get("completed")) or t.get("state") == "post" or when < now
            if not done:
                continue
            games.append((e.get("date", ""), mine, opp, oid, oname))
        except Exception:
            continue
    games.sort()
    return games[-30:]


def find_result(tid, opp_name, since, until, bucket):
    """Final score 'home-away' of the game vs opp_name between two dates, or None."""
    if not tid:
        return None
    for date, mine, opp, oid, oname in espn_recent(tid, bucket):
        try:
            gd = pd.to_datetime(date, utc=True).tz_convert(None).normalize()
        except Exception:
            continue
        if gd < since or gd > until:
            continue
        if name_score(oname, opp_name) >= 0.6:
            return f"{int(mine)}-{int(opp)}"
    return None


@lru_cache(maxsize=None)
def id_stats(tid, bucket):
    if not tid:
        return None
    g = espn_recent(tid, bucket)[-12:]
    if len(g) < 4:
        return None
    n, k = len(g), 3
    sc = sum(x[1] for x in g)
    co = sum(x[2] for x in g)
    return ((sc + k * BASE_RATE) / (n + k), (co + k * BASE_RATE) / (n + k), n)


@lru_cache(maxsize=None)
def lookup(key, bucket):
    """Returns (stats or None, info dict explaining what happened)."""
    name, women = key
    info = {"id": None, "why": ""}
    q = search_query(name)
    if not q:
        info["why"] = "name too short to search"
        return None, info
    cands = [(i, n) for i, n, w in directory(bucket) if w == women]
    s, tid = best_match(q, cands)
    src = "team list"
    if s < 0.6:
        tid = None
    if not tid:
        tid = espn_team_id(name, women, bucket)
        src = "search"
    if not tid:
        info["why"] = "team not found on ESPN"
        return None, info
    info["id"] = tid
    g = espn_recent(tid, bucket)[-12:]
    if len(g) < 4:
        info["why"] = f"found via {src}, but only {len(g)} recent games read"
        return None, info
    n, k = len(g), 3
    sc = sum(x[1] for x in g)
    co = sum(x[2] for x in g)
    info["why"] = "ok"
    return ((sc + k * BASE_RATE) / (n + k), (co + k * BASE_RATE) / (n + k), n), info


def live_lams(sh, sa):
    return (float(np.clip((sh[0] + sa[1]) / 2 * 1.08, 0.35, 3.2)),
            float(np.clip((sa[0] + sh[1]) / 2 * 0.95, 0.25, 2.8)))


# ---------------- Fixtures and results (ESPN) ----------------
@st.cache_data(ttl=1200)
def load_espn(start, end):
    rng = start if start == end else f"{start}-{end}"

    def fetch(item):
        code, (label, div, is_cup) = item
        url = ("https://site.api.espn.com/apis/site/v2/sports/soccer/"
               f"{code}/scoreboard?dates={rng}&limit=300")
        rows = []
        try:
            data = requests.get(url, timeout=8).json()
        except Exception:
            return rows
        for e in data.get("events", []):
            try:
                state = e["status"]["type"]["state"]
                if state not in ("pre", "post"):
                    continue
                comp = e["competitions"][0]["competitors"]
                home = next(c for c in comp if c["homeAway"] == "home")
                away = next(c for c in comp if c["homeAway"] == "away")
                date = pd.to_datetime(e["date"], utc=True).tz_convert(None).normalize()
                score = None
                if state == "post":
                    hs, as_ = _score(home.get("score")), _score(away.get("score"))
                    if hs is not None and as_ is not None:
                        score = f"{int(hs)}-{int(as_)}"
                rows.append({
                    "Comp": label, "Div": div, "Cup": is_cup, "Date": date,
                    "HomeTeam": home["team"]["displayName"],
                    "AwayTeam": away["team"]["displayName"],
                    "hid": str(home["team"].get("id", "")),
                    "aid": str(away["team"].get("id", "")),
                    "state": state, "score": score,
                })
            except Exception:
                continue
        return rows

    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(fetch, ESPN_COMPS.items()))
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
    index = {e["key"]: e for e in hist}
    changed = False
    for e in entries:
        old = index.get(e["key"])
        if old is None:
            hist.append(e)
            changed = True
        elif e.get("score") and not old.get("score"):
            old["score"] = e["score"]
            changed = True
    if changed:
        save_history(hist)


def check_results(hist):
    changed = 0
    today = pd.Timestamp.today().normalize()
    for e in hist:
        if e.get("score"):
            continue
        md = pd.Timestamp(e["match_date"])
        if md > today:
            continue
        women = is_women(e["home"]) or is_women(e["away"])
        tid = e.get("hid") or lookup((e["home"], women), BUCKET)[1]["id"]
        sc = find_result(tid, e["away"], md - ONE_DAY, md + 10 * ONE_DAY, RB)
        if sc:
            e["score"] = sc
            changed += 1
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


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    min2 = st.slider("GG2+ is strong from (%)", 8, 30, 14)
    min1 = st.slider("GG is strong from (%)", 30, 80, 50)

CFG2 = st.column_config.ProgressColumn("GG2+ %", min_value=0, max_value=100,
                                       format="%.1f%%")
CFG1 = st.column_config.ProgressColumn("GG %", min_value=0, max_value=100,
                                       format="%.1f%%")


def add_chances(df):
    df = df.copy()
    df["GG2+ %"] = [round(prob_k(a, b, 2) * 100, 1) for a, b in zip(df["lh"], df["la"])]
    df["GG %"] = [round(prob_k(a, b, 1) * 100, 1) for a, b in zip(df["lh"], df["la"])]
    df["Exp. goals"] = [f"{a:.1f} - {b:.1f}" for a, b in zip(df["lh"], df["la"])]
    return df


TODAY = pd.Timestamp.today().normalize()

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
        with st.spinner("Looking up each team (the first time takes a while)..."):
            directory(BUCKET)
            with ThreadPoolExecutor(8) as ex:
                looked = dict(zip(keys, ex.map(lambda k: lookup(k, BUCKET), keys)))

        rows, unrated = [], []
        for hn, an in pairs:
            women = is_women(hn) or is_women(an)
            sh, ih = looked[(hn, women)]
            sa, ia = looked[(an, women)]
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
                    unrated.append({"Match": f"{hn} v {an}",
                                    "Home team": ih["why"], "Away team": ia["why"]})
                    continue
            score = find_result(ih["id"], an, TODAY - 7 * ONE_DAY, TODAY + ONE_DAY, RB)
            rows.append({"Home": hn, "Away": an, "Match": f"{hn} v {an}",
                         "lh": lh, "la": la, "Data": src, "hid": ih["id"] or "",
                         "Score": score or ""})

        if not rows:
            st.warning("None of the matches could be rated. See the reasons below.")
        else:
            df_slip = add_chances(pd.DataFrame(rows))
            df_slip["GG2+ result"] = [result_icon(s, 2) for s in df_slip["Score"]]
            df_slip["GG result"] = [result_icon(s, 1) for s in df_slip["Score"]]

            rank = st.radio("Rank my best 7 by", ["GG2+", "GG"], horizontal=True)
            col = "GG2+ %" if rank == "GG2+" else "GG %"
            mn = min2 if rank == "GG2+" else min1
            top = df_slip.sort_values(col, ascending=False).head(7).copy()
            top["Verdict"] = np.where(top[col] >= mn, "Strong", "Weak")
            st.dataframe(top[["Match", "GG2+ %", "GG %", "Score", "GG2+ result",
                              "GG result", "Verdict"]],
                         hide_index=True, use_container_width=True,
                         column_config={"GG2+ %": CFG2, "GG %": CFG1})
            if (top["Verdict"] == "Strong").sum() == 0:
                st.info(f"None reach {mn}% for {rank}. Skipping is reasonable.")
            played = int((df_slip["Score"] != "").sum())
            st.caption(f"{len(rows)} of {len(pairs)} matches rated. "
                       f"{played} already played, with scores shown. "
                       "Upload the same slip again after the matches to see results.")

            today_iso = TODAY.strftime("%Y-%m-%d")
            picked = {}
            for _, r in df_slip.sort_values("GG2+ %", ascending=False).head(7).iterrows():
                picked.setdefault(r["Match"], set()).add("GG2+")
            for _, r in df_slip.sort_values("GG %", ascending=False).head(7).iterrows():
                picked.setdefault(r["Match"], set()).add("GG")
            entries = []
            for _, r in df_slip.iterrows():
                if r["Match"] in picked:
                    entries.append({
                        "key": f"Slip|{today_iso}|{r['Match']}", "saved": today_iso,
                        "match_date": today_iso, "source": "Slip", "home": r["Home"],
                        "away": r["Away"], "gg2": float(r["GG2+ %"]),
                        "gg1": float(r["GG %"]),
                        "picked": ", ".join(sorted(picked[r["Match"]])),
                        "hid": r["hid"], "score": r["Score"] or None})
            add_history(entries)
        if unrated:
            with st.expander(f"{len(unrated)} not rated: why", expanded=True):
                st.dataframe(pd.DataFrame(unrated), hide_index=True,
                             use_container_width=True)

# ======== Daily picks ========
with tab_daily:
    period = st.selectbox("Period", ["Today", "Tomorrow", "Next 3 days", "Next 7 days",
                                     "Yesterday", "Last 3 days"])
    start, end = {
        "Today": (TODAY, TODAY), "Tomorrow": (TODAY + ONE_DAY, TODAY + ONE_DAY),
        "Next 3 days": (TODAY, TODAY + 3 * ONE_DAY),
        "Next 7 days": (TODAY, TODAY + 7 * ONE_DAY),
        "Yesterday": (TODAY - ONE_DAY, TODAY - ONE_DAY),
        "Last 3 days": (TODAY - 3 * ONE_DAY, TODAY - ONE_DAY),
    }[period]

    with st.spinner("Loading matches..."):
        fx = load_espn(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))

    rated, pending, counts = [], [], {}
    for r in fx:
        counts.setdefault(r["Comp"], [0, 0])
        counts[r["Comp"]][0] += 1
        lams = None
        if r["Div"] and r["Div"] in strengths:
            h = find_team(r["HomeTeam"], r["Div"], 0.7)
            a = find_team(r["AwayTeam"], r["Div"], 0.7)
            if h and a and h != a:
                lams = (*file_lams(h, a), "League files")
        elif r["Cup"]:
            h = find_team(r["HomeTeam"], None, 0.85)
            a = find_team(r["AwayTeam"], None, 0.85)
            if h and a and h != a:
                lams = (*file_lams(h, a), "League files")
        if lams:
            rated.append((r, lams))
        else:
            pending.append(r)

    if pending:
        ids = sorted({i for r in pending for i in (r["hid"], r["aid"]) if i})
        with st.spinner("Fetching recent results for teams..."):
            with ThreadPoolExecutor(8) as ex:
                list(ex.map(lambda t: id_stats(t, BUCKET), ids))
        for r in pending:
            sh, sa = id_stats(r["hid"], BUCKET), id_stats(r["aid"], BUCKET)
            if sh and sa:
                rated.append((r, (*live_lams(sh, sa), "Live results")))

    drows = []
    for r, (lh, la, src) in rated:
        counts[r["Comp"]][1] += 1
        drows.append({
            "Match": f"{r['HomeTeam']} v {r['AwayTeam']}",
            "Home": r["HomeTeam"], "Away": r["AwayTeam"],
            "Date": r["Date"].strftime("%d %b"), "iso": r["Date"].strftime("%Y-%m-%d"),
            "Competition": r["Comp"], "lh": lh, "la": la, "Data": src,
            "Score": r["score"] or "", "state": r["state"], "hid": r["hid"]})

    if not drows:
        st.info("No rated matches for this period. Try a different period.")
    else:
        df_d = add_chances(pd.DataFrame(drows))
        df_d["GG2+ result"] = [result_icon(s, 2) for s in df_d["Score"]]
        df_d["GG result"] = [result_icon(s, 1) for s in df_d["Score"]]

        c1, c2, c3 = st.columns(3)
        sort_by = c1.selectbox("Sort by", ["GG2+ %", "GG %", "Date"])
        top_n = c2.selectbox("Show", ["10", "20", "50", "All"])
        only_strong = c3.checkbox("Strong only", value=True)

        view = df_d.copy()
        if only_strong:
            view = view[(view["GG2+ %"] >= min2) | (view["GG %"] >= min1)]
        view = view.sort_values("iso" if sort_by == "Date" else sort_by,
                                ascending=(sort_by == "Date"))
        if top_n != "All":
            view = view.head(int(top_n))

        if view.empty:
            st.info("No strong matches for this period. Untick 'Strong only' "
                    "to see everything. Skipping is reasonable.")
        else:
            st.dataframe(view[["Match", "GG2+ %", "GG %", "Score", "GG2+ result",
                               "GG result", "Date", "Competition", "Exp. goals"]],
                         hide_index=True, use_container_width=True,
                         column_config={"GG2+ %": CFG2, "GG %": CFG1})
        st.caption(f"{len(df_d)} matches rated. For finished matches the chances are "
                   "worked out now, so they are a rough check and not the original "
                   "pre-match numbers. Saved picks in History keep the original.")

        entries = []
        today_iso = TODAY.strftime("%Y-%m-%d")
        up = df_d[(df_d["state"] == "pre")]
        for _, r in up.iterrows():
            tags = []
            if r["GG2+ %"] >= min2:
                tags.append("GG2+")
            if r["GG %"] >= min1:
                tags.append("GG")
            if tags:
                entries.append({
                    "key": f"Daily|{r['iso']}|{r['Match']}", "saved": today_iso,
                    "match_date": r["iso"], "source": "Daily", "home": r["Home"],
                    "away": r["Away"], "gg2": float(r["GG2+ %"]),
                    "gg1": float(r["GG %"]), "picked": ", ".join(tags),
                    "hid": r["hid"], "score": None})
        add_history(entries)

    with st.expander("Leagues checked"):
        if counts:
            st.dataframe(pd.DataFrame(
                [{"Competition": k, "Found": v[0], "Rated": v[1]}
                 for k, v in sorted(counts.items())]),
                hide_index=True, use_container_width=True)
        else:
            st.write("No matches returned for these dates (international break "
                     "or off-season).")

# ======== History ========
with tab_hist:
    hist = load_history()
    pending_h = [e for e in hist if not e.get("score")
                 and pd.Timestamp(e["match_date"]) <= TODAY]
    b1, b2 = st.columns(2)
    manual = b1.button("🔄 Re-check results", use_container_width=True)
    if pending_h and (manual or time.time() - st.session_state.get("last_auto", 0) > 600):
        with st.spinner("Checking results..."):
            n = check_results(hist)
            save_history(hist)
        st.session_state["last_auto"] = time.time()
        if manual:
            st.success(f"{n} new results found.")
        hist = load_history()

    if hist:
        hdf = pd.DataFrame(hist)
        hdf["Match"] = hdf["home"] + " v " + hdf["away"]
        hdf["Score"] = hdf["score"].fillna("")
        hdf["GG2+ result"] = [result_icon(s, 2) for s in hdf["Score"]]
        hdf["GG result"] = [result_icon(s, 1) for s in hdf["Score"]]
        hdf = hdf.sort_values(["match_date", "gg2"], ascending=[False, False])

        m1, m2 = st.columns(2)
        for col, mkt, rc in ((m1, "GG2+", "GG2+ result"), (m2, "GG", "GG result")):
            sub = hdf[hdf["picked"].str.contains(mkt, regex=False) & (hdf[rc] != "–")]
            hits = int((sub[rc] == "✅").sum())
            col.metric(f"{mkt} hit rate", f"{hits}/{len(sub)}" if len(sub) else "–",
                       f"{hits / len(sub) * 100:.0f}%" if len(sub) else None,
                       delta_color="off")

        st.dataframe(
            hdf.rename(columns={"gg2": "GG2+ %", "gg1": "GG %", "match_date": "Date",
                                "source": "From", "picked": "Picked for"})[
                ["Match", "GG2+ %", "GG %", "Score", "GG2+ result", "GG result",
                 "Date", "From", "Picked for"]],
            hide_index=True, use_container_width=True,
            column_config={"GG2+ %": CFG2, "GG %": CFG1})
        b2.download_button("⬇️ Download", hdf.to_csv(index=False).encode(),
                           "gg_history.csv", "text/csv", use_container_width=True)
    else:
        st.info("Nothing saved yet. Slip analyses and strong daily picks are "
                "saved here automatically.")

    with st.expander("Restore saved history"):
        up_h = st.file_uploader("Upload a downloaded history file (.csv)",
                                type=["csv"], key="hist_up")
        if up_h is not None:
            try:
                old = pd.read_csv(up_h)
                recs = []
                for _, r in old.iterrows():
                    recs.append({
                        "key": str(r["key"]), "saved": str(r["saved"]),
                        "match_date": str(r["match_date"]), "source": str(r["source"]),
                        "home": str(r["home"]), "away": str(r["away"]),
                        "gg2": float(r["gg2"]), "gg1": float(r["gg1"]),
                        "picked": str(r["picked"]),
                        "hid": "" if pd.isna(r.get("hid")) else str(r["hid"]),
                        "score": None if pd.isna(r.get("score")) else str(r["score"])})
                add_history(recs)
                st.success(f"{len(recs)} rows restored. Reload the page to see them.")
            except Exception:
                st.error("That file could not be read.")
    st.caption("History is kept on the app server. It is cleared when the app "
               "restarts or when you change the code, so download it regularly. "
               "You can always upload an old slip in 'My slip' to see its results.")

with st.expander("How it works"):
    st.markdown("""
**GG2+** means both teams score at least 2 goals each. **GG** means both teams
score at least 1. The app estimates each team's goals from past results with a
Poisson model, then works out the chance of each result. These are chances, not
guarantees.
""")
