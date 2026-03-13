import streamlit as st
from auth import require_auth, show_user_info
from logger import logger

st.set_page_config(
    page_title="🏠 House Price ML App",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 🔐 Auth
require_auth()

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&display=swap');
  * { font-family: 'Outfit', sans-serif; }
  .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
  [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #1a1a2e, #16213e, #0f3460);
      border-right: 1px solid #e94560;
  }
  [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
  .hero-card {
      background: linear-gradient(135deg, #1a1a2e, #16213e);
      border: 1px solid #e94560; border-radius: 20px;
      padding: 3rem 2rem; text-align: center;
      box-shadow: 0 0 40px rgba(233,69,96,0.3); margin-bottom: 2rem;
  }
  .hero-title {
      font-size: 3.5rem; font-weight: 900;
      background: linear-gradient(90deg, #e94560, #00d4ff, #a855f7);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .feat-grid { display:flex; gap:1.5rem; flex-wrap:wrap; margin-top:2rem; }
  .feat-card { flex:1; min-width:220px; border-radius:16px; padding:1.5rem; text-align:center; }
  .feat-1 { background: linear-gradient(135deg,#e94560,#c62a47); }
  .feat-2 { background: linear-gradient(135deg,#0f3460,#533483); }
  .feat-3 { background: linear-gradient(135deg,#00b4d8,#0077b6); }
  .feat-icon { font-size:2.8rem; }
  .feat-title { color:#fff; font-size:1.2rem; font-weight:700; margin:.5rem 0 .3rem; }
  .feat-desc  { color:rgba(255,255,255,.75); font-size:.9rem; }
  h1,h2,h3,h4 { color:#fff !important; }
  p,li { color:#c0c0d0 !important; }
</style>
""", unsafe_allow_html=True)

logger.info(f"PAGE_VIEW | user={st.session_state.get('username')} | page=Home")

with st.sidebar:
    st.markdown("## 🏠 Navigation")
    st.markdown("---")
    st.markdown("📊 **Page 1** — Données")
    st.markdown("🤖 **Page 2** — Entraînement")
    st.markdown("🔮 **Page 3** — Prédiction")
    st.markdown("---")
    st.info("Ames Housing Dataset\n\n• Train : 1 460 lignes\n• Test  : 1 459 lignes\n• Cible : SalePrice")
    show_user_info()

user_name = st.session_state.get("user_name", "")
st.markdown(f"""
<div class='hero-card'>
  <div class='hero-title'>🏠 House Price Predictor</div>
  <div style='color:#00d4ff;font-size:1.3rem;font-weight:700'>ML Application — Ames Housing</div>
  <div style='color:#aaa;margin-top:.5rem'>Bienvenue, <b style='color:#e94560'>{user_name}</b> 👋</div>
</div>
<div class='feat-grid'>
  <div class='feat-card feat-1'>
      <div class='feat-icon'>📊</div>
      <div class='feat-title'>Page 1 · Données</div>
      <div class='feat-desc'>Upload CSV, statistiques, distributions, corrélations</div>
  </div>
  <div class='feat-card feat-2'>
      <div class='feat-icon'>🤖</div>
      <div class='feat-title'>Page 2 · Entraînement</div>
      <div class='feat-desc'>Random Forest, Gradient Boosting, Ridge, Lasso</div>
  </div>
  <div class='feat-card feat-3'>
      <div class='feat-icon'>🔮</div>
      <div class='feat-title'>Page 3 · Prédiction</div>
      <div class='feat-desc'>Saisie manuelle ou batch CSV</div>
  </div>
</div>
<div style='background:rgba(16,185,129,.07);border:1px solid #10b981;
            border-radius:10px;padding:.7rem 1.2rem;margin-top:2rem'>
  🔒 <span style='color:#10b981;font-size:.9rem'>
      Connexion sécurisée · HTTPS · Sessions protégées · Logs actifs
  </span>
</div>
""", unsafe_allow_html=True)
