# ⚽ GG2+ Football Predictor

**Predict the probability that both teams score at least 2 goals each** in a football match.

This is a simple web tool based on a statistical model trained on Brazilian Série A data.

---

## How to use this project (for complete beginners)

### Option 1 – Easiest: Run it online with Streamlit Cloud (Recommended)

1. Create a free account on [GitHub](https://github.com) if you don’t have one.
2. Create a free account on [Streamlit Community Cloud](https://share.streamlit.io).
3. Follow the steps below to put this project on GitHub.
4. Then connect it to Streamlit Cloud (instructions at the bottom).

---

### Step-by-step: Put this project on GitHub

#### 1. Create a new repository on GitHub
- Go to https://github.com/new
- Repository name: `gg2plus-predictor` (or any name you like)
- Make it **Public**
- Do **not** add README, .gitignore or license (we already have them)
- Click **Create repository**

#### 2. Upload the files
You have two easy ways:

**Method A – Upload with the website (easiest)**
1. On your new repository page, click **Add file** → **Upload files**
2. Drag and drop these 3 files:
   - `app.py`
   - `gg2plus_model_v2.pkl`
   - `requirements.txt`
3. Click **Commit changes**

**Method B – Using GitHub Desktop** (if you prefer a program)
1. Download [GitHub Desktop](https://desktop.github.com)
2. Clone your new repository
3. Copy the 3 files into the folder
4. Commit and push

---

### Step 3 – Make it run live on the internet (Streamlit Cloud)

1. Go to https://share.streamlit.io and log in with your GitHub account
2. Click **New app**
3. Select your repository (`gg2plus-predictor`)
4. Main file path: `app.py`
5. Click **Deploy**

After 1–2 minutes you will get a public link like:
`https://gg2plus-predictor.streamlit.app`

Anyone can open that link and use the predictor — no coding needed.

---

## Files in this project

| File | What it does |
|------|--------------|
| `app.py` | The web interface (Streamlit app) |
| `gg2plus_model_v2.pkl` | The trained prediction model |
| `requirements.txt` | List of Python packages needed |

---

## What GG2+ means

Both teams score **at least 2 goals each**.

Examples of qualifying scores: 2-2, 3-2, 2-3, 3-3, 4-2, 2-4, etc.

This is a rare event (happens roughly 11–13% of the time in Brazilian Série A).

---

Made with a simple Poisson statistical model trained on Brazilian league data.
