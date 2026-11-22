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
SNAP_PATH = os.path.join(tempfile.gettempdir(), "gg_daily_snap.json")
ONE_DAY = pd.Timedelta(days=1)


def _secret(name, default=""):
    try:
        return str(st.secrets[name])
    except Exception:
        return default


SDB_KEY = _secret("SPORTSDB_KEY", "3") or "3"
AF_KEY = _secret("APIFOOTBALL_KEY", "")


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
    except E
