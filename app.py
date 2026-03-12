import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 House Price ML App",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Global background */
  .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

  /* Sidebar */
  [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
      border-right: 1px solid #e94560;
  }
  [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
  [data-testid="stSidebar"] .stSelectbox label { color: #00d4ff !important; }

  /* Hero card */
  .hero-card {
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
      border: 1px solid #e94560;
      border-radius: 20px;
      padding: 3rem 2rem;
      text-align: center;
      box-shadow: 0 0 40px rgba(233,69,96,0.3);
      margin-bottom: 2rem;
  }
  .hero-title {
      font-size: 3.5rem;
      font-weight: 900;
      background: linear-gradient(90deg, #e94560, #00d4ff, #a855f7);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin-bottom: 0.5rem;
  }
  .hero-sub {
      color: #a0a0b0;
      font-size: 1.2rem;
      margin-top: 0.5rem;
  }

  /* Feature cards */
  .feat-grid { display: flex; gap: 1.5rem; flex-wrap: wrap; margin-top: 2rem; }
  .feat-card {
      flex: 1; min-width: 220px;
      border-radius: 16px;
      padding: 1.5rem 1.2rem;
      text-align: center;
      transition: transform .2s;
  }
  .feat-card:hover { transform: translateY(-6px); }
  .feat-1 { background: linear-gradient(135deg, #e94560 0%, #c62a47 100%); }
  .feat-2 { background: linear-gradient(135deg, #0f3460 0%, #533483 100%); }
  .feat-3 { background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%); }
  .feat-icon { font-size: 2.8rem; }
  .feat-title { color: #fff; font-size: 1.2rem; font-weight: 700; margin: .5rem 0 .3rem; }
  .feat-desc  { color: rgba(255,255,255,.75); font-size: .9rem; }

  /* Stat pill */
  .stat-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 1.5rem; justify-content: center; }
  .stat-pill {
      background: rgba(255,255,255,.07);
      border: 1px solid rgba(255,255,255,.15);
      border-radius: 50px;
      padding: .4rem 1.2rem;
      color: #00d4ff;
      font-size: .9rem;
      font-weight: 600;
  }

  /* Section title */
  .section-title {
      color: #e94560;
      font-size: 1.5rem;
      font-weight: 800;
      margin: 2rem 0 1rem;
      border-left: 4px solid #e94560;
      padding-left: .8rem;
  }

  /* Info box */
  .info-box {
      background: rgba(0, 212, 255, .08);
      border: 1px solid #00d4ff;
      border-radius: 12px;
      padding: 1rem 1.4rem;
      color: #cce7ff;
      font-size: .95rem;
      line-height: 1.7;
  }

  /* Streamlit default overrides */
  h1,h2,h3,h4 { color: #ffffff !important; }
  p, li { color: #c0c0d0 !important; }
  .stMarkdown { color: #c0c0d0; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 Navigation")
    st.markdown("---")
    st.markdown("📊 **Page 1** — Données")
    st.markdown("🤖 **Page 2** — Entraînement")
    st.markdown("🔮 **Page 3** — Prédiction")
    st.markdown("---")
    st.markdown("### 📁 Dataset")
    st.info("Ames Housing Dataset\n\n• Train : 1 460 lignes\n• Test  : 1 459 lignes\n• Cible : SalePrice")
    st.markdown("---")
    st.markdown(
        "<div style='color:#666;font-size:.8rem;text-align:center'>Made with ❤️ & Streamlit</div>",
        unsafe_allow_html=True,
    )

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-card'>
  <div class='hero-title'>🏠 House Price Predictor</div>
  <div style='color:#00d4ff;font-size:1.5rem;font-weight:700;margin:.3rem 0'>
      ML Application — Ames Housing
  </div>
  <div class='hero-sub'>
      Explorez les données · Entraînez un modèle · Faites des prédictions
  </div>
  <div class='stat-row'>
      <span class='stat-pill'>🏡 1 460 maisons</span>
      <span class='stat-pill'>📐 79 features</span>
      <span class='stat-pill'>🎯 Régression</span>
      <span class='stat-pill'>🤖 Random Forest & XGBoost</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Feature cards ──────────────────────────────────────────────────────────────
st.markdown("""
<div class='feat-grid'>
  <div class='feat-card feat-1'>
      <div class='feat-icon'>📊</div>
      <div class='feat-title'>Page 1 · Données</div>
      <div class='feat-desc'>Upload CSV, statistiques descriptives, distributions, corrélations et valeurs manquantes</div>
  </div>
  <div class='feat-card feat-2'>
      <div class='feat-icon'>🤖</div>
      <div class='feat-title'>Page 2 · Entraînement</div>
      <div class='feat-desc'>Choix du modèle, hyperparamètres, métriques RMSE/R², courbes d'apprentissage</div>
  </div>
  <div class='feat-card feat-3'>
      <div class='feat-icon'>🔮</div>
      <div class='feat-title'>Page 3 · Prédiction</div>
      <div class='feat-desc'>Saisie manuelle ou upload CSV, prédiction instantanée avec intervalle de confiance</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── How it works ───────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>🚀 Comment utiliser l'application</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
steps = [
    ("1️⃣", "#e94560", "Chargez vos données", "Allez sur **Page 1 · Données** et uploadez `train.csv`. Explorez les statistiques et visualisations automatiques."),
    ("2️⃣", "#a855f7", "Entraînez le modèle", "Sur **Page 2 · Entraînement**, choisissez l'algorithme, réglez les hyperparamètres et lancez l'entraînement."),
    ("3️⃣", "#00d4ff", "Faites des prédictions", "Sur **Page 3 · Prédiction**, entrez les caractéristiques d'une maison ou uploadez `test.csv` pour des prédictions batch."),
]
for col, (num, color, title, desc) in zip([col1, col2, col3], steps):
    with col:
        st.markdown(f"""
        <div style='background:rgba(255,255,255,.04);border:1px solid {color};border-radius:14px;padding:1.3rem;height:180px'>
            <div style='font-size:2rem;text-align:center'>{num}</div>
            <div style='color:{color};font-weight:700;font-size:1rem;text-align:center;margin:.4rem 0'>{title}</div>
            <div style='color:#aaa;font-size:.88rem;text-align:center'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Dataset info ───────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>📚 À propos du dataset</div>", unsafe_allow_html=True)
st.markdown("""
<div class='info-box'>
  Le <b>Ames Housing Dataset</b> contient des informations sur les ventes de maisons à Ames, Iowa (2006–2010).<br>
  Il comprend <b>79 variables explicatives</b> décrivant presque chaque aspect des maisons résidentielles.<br><br>
  <b>Objectif :</b> Prédire le prix de vente final de chaque maison (<code>SalePrice</code>).<br>
  <b>Évaluation :</b> RMSE sur le log des prix prédit vs réel.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;color:#555;font-size:.85rem'>© 2024 House Price ML App · Exercice 3 — Application Multi-Pages</div>",
    unsafe_allow_html=True,
)

