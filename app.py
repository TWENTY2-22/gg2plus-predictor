import streamlit as st
import pickle
import numpy as np
from scipy.stats import poisson
import os

st.set_page_config(
    page_title="GG2+ Predictor",
    page_icon="⚽",
    layout="centered"
)

st.title("⚽ GG2+ Predictor")
st.markdown("**Both teams score at least 2 goals each**")
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    model_path = "gg2plus_model_v2.pkl"
    if not os.path.exists(model_path):
        st.error("Model file not found!")
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

if model is None:
    st.stop()

attack = model["attack"]
defence = model["defence"]
avg_h = model["avg_home_goals"]
avg_a = model["avg_away_goals"]
home_adv = model["home_advantage"]

# Name aliases for easier input
name_aliases = {
    "Flamengo": "Flamengo RJ",
    "Botafogo": "Botafogo RJ",
    "Chapecoense": "Chapecoense-SC",
    "Grêmio": "Gremio",
    "Gremio": "Gremio",
    "São Paulo": "Sao Paulo",
    "Sao Paulo": "Sao Paulo",
    "Vasco da Gama": "Vasco",
    "Vasco": "Vasco",
    "Athletico PR": "Athletico-PR",
    "Atletico-PR": "Athletico-PR",
    "Athletico-PR": "Athletico-PR",
}

def normalize(name):
    name = name.strip()
    return name_aliases.get(name, name)

def predict(home, away):
    home = normalize(home)
    away = normalize(away)
    
    att_h = attack.get(home, 1.0)
    def_a = defence.get(away, 1.0)
    att_a = attack.get(away, 1.0)
    def_h = defence.get(home, 1.0)
    
    lambda_home = float(np.clip(att_h * def_a * avg_h * home_adv, 0.35, 3.2))
    lambda_away = float(np.clip(att_a * def_h * avg_a, 0.25, 2.8))
    
    p_home_ge2 = 1 - poisson.pmf(0, lambda_home) - poisson.pmf(1, lambda_home)
    p_away_ge2 = 1 - poisson.pmf(0, lambda_away) - poisson.pmf(1, lambda_away)
    
    prob = p_home_ge2 * p_away_ge2
    return prob, lambda_home, lambda_away, home, away

# Sidebar with team list
with st.sidebar:
    st.header("Available Teams")
    st.caption("Click to copy a name")
    teams = sorted(attack.keys())
    for t in teams:
        st.code(t, language=None)

# Main input
col1, col2 = st.columns(2)
with col1:
    home_team = st.text_input("Home Team", value="Coritiba")
with col2:
    away_team = st.text_input("Away Team", value="Santos")

if st.button("Predict GG2+", type="primary", use_container_width=True):
    if not home_team or not away_team:
        st.warning("Please enter both team names.")
    else:
        prob, lh, la, h_norm, a_norm = predict(home_team, away_team)
        
        st.markdown("### Result")
        
        # Big probability
        st.metric(
            label="Probability both teams score ≥ 2 goals",
            value=f"{prob*100:.1f}%"
        )
        
        # Expected goals
        c1, c2 = st.columns(2)
        c1.metric("Expected Home Goals", f"{lh:.2f}")
        c2.metric("Expected Away Goals", f"{la:.2f}")
        
        # Interpretation
        if prob >= 0.18:
            st.success("🔥 Relatively high chance for this rare market")
        elif prob >= 0.13:
            st.info("Average / slightly above average chance")
        else:
            st.warning("Below average chance")
        
        st.caption(f"Model used normalized names: **{h_norm}** vs **{a_norm}**")

st.markdown("---")
st.markdown("""
**What is GG2+?**  
Both teams score **at least 2 goals each** (examples: 2-2, 3-2, 2-3, 4-2, etc.)

**How the model works**  
It uses historical Brazilian Série A data to estimate each team's attacking and defensive strength, then calculates the probability using a Poisson model.
""")
