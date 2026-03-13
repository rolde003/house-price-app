import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="📊 Données", page_icon="📊", layout="wide")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
  [data-testid="stSidebar"] {
      background: linear-gradient(180deg,#1a1a2e,#16213e,#0f3460);
      border-right: 1px solid #e94560;
  }
  [data-testid="stSidebar"] * { color: #e0e0e0 !important; }

  .page-header {
      background: linear-gradient(135deg,#1a1a2e,#16213e);
      border: 1px solid #e94560; border-radius:16px;
      padding:1.8rem 2rem; margin-bottom:1.5rem;
      box-shadow: 0 0 30px rgba(233,69,96,.25);
  }
  .page-title { font-size:2.2rem; font-weight:900;
      background:linear-gradient(90deg,#e94560,#ff6b6b);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent; }

  .kpi-card {
      background:rgba(255,255,255,.05); border-radius:14px;
      padding:1.2rem; text-align:center;
      border: 1px solid rgba(255,255,255,.1);
      transition: transform .2s;
  }
  .kpi-card:hover { transform:translateY(-4px); }
  .kpi-value { font-size:2rem; font-weight:900; }
  .kpi-label { color:#888; font-size:.85rem; margin-top:.3rem; }

  .section-title {
      color:#00d4ff; font-size:1.3rem; font-weight:800;
      border-left:4px solid #00d4ff; padding-left:.7rem;
      margin:1.8rem 0 1rem;
  }
  .stDataFrame { border-radius:12px !important; }
  h1,h2,h3,h4 { color:#fff !important; }
  p,li { color:#c0c0d0 !important; }
  .stTabs [data-baseweb="tab"] { color: #888 !important; }
  .stTabs [aria-selected="true"] { color: #00d4ff !important; border-bottom:2px solid #00d4ff !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Page 1 · Données")
    st.markdown("---")
    st.markdown("### Filtres")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
  <div class='page-title'>📊 Exploration des Données</div>
  <div style='color:#aaa;margin-top:.4rem'>Upload, statistiques, distributions & corrélations</div>
</div>
""", unsafe_allow_html=True)

# ── Upload ─────────────────────────────────────────────────────────────────────
col_up1, col_up2 = st.columns(2)
with col_up1:
    train_file = st.file_uploader("📂 Charger train.csv", type="csv", key="train")
with col_up2:
    test_file = st.file_uploader("📂 Charger test.csv (optionnel)", type="csv", key="test")

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_df(file):
    return pd.read_csv(file)

if train_file is None:
    st.markdown("""
    <div style='background:rgba(0,212,255,.07);border:1px solid #00d4ff;
                border-radius:12px;padding:1.5rem;text-align:center;margin-top:1rem'>
        <div style='font-size:3rem'>📂</div>
        <div style='color:#00d4ff;font-size:1.2rem;font-weight:700;margin:.5rem 0'>
            Uploadez train.csv pour commencer</div>
        <div style='color:#888'>Glissez-déposez ou cliquez sur le bouton ci-dessus</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

df = load_df(train_file)
st.session_state["train_df"] = df

if test_file:
    df_test = load_df(test_file)
    st.session_state["test_df"] = df_test

# ── KPIs ───────────────────────────────────────────────────────────────────────
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()
missing_pct = (df.isnull().sum().sum() / df.size * 100)

kpis = [
    ("#e94560", "🏡", f"{len(df):,}", "Observations"),
    ("#a855f7", "📐", f"{df.shape[1]}", "Features"),
    ("#00d4ff", "🔢", f"{len(num_cols)}", "Numériques"),
    ("#f59e0b", "🏷️", f"{len(cat_cols)}", "Catégorielles"),
    ("#10b981", "❓", f"{missing_pct:.1f}%", "Valeurs manquantes"),
    ("#06b6d4", "💰", f"${df['SalePrice'].median()/1000:.0f}K", "Prix médian"),
]
cols = st.columns(6)
for col, (color, icon, val, label) in zip(cols, kpis):
    with col:
        st.markdown(f"""
        <div class='kpi-card'>
            <div style='font-size:1.6rem'>{icon}</div>
            <div class='kpi-value' style='color:{color}'>{val}</div>
            <div class='kpi-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Aperçu", "📈 Distributions", "🔗 Corrélations", "❓ Manquants", "🔍 Analyse cible"
])

# ── Tab 1 : Preview ────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<div class='section-title'>Aperçu des données</div>", unsafe_allow_html=True)
    n_rows = st.slider("Nombre de lignes à afficher", 5, 50, 10)
    st.dataframe(df.head(n_rows), use_container_width=True, height=300)

    st.markdown("<div class='section-title'>Statistiques descriptives</div>", unsafe_allow_html=True)
    col_sel = st.multiselect("Sélectionner des colonnes (vide = toutes numériques)",
                             num_cols, default=[], key="desc_cols")
    show_cols = col_sel if col_sel else num_cols[:15]
    st.dataframe(df[show_cols].describe().T.style.background_gradient(cmap="RdYlGn"), use_container_width=True)

# ── Tab 2 : Distributions ─────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-title'>Distribution des variables numériques</div>", unsafe_allow_html=True)
    feat = st.selectbox("Choisir une variable", num_cols, key="dist_feat")
    log_scale = st.checkbox("Échelle logarithmique", value=(feat == "SalePrice"))

    data_plot = np.log1p(df[feat].dropna()) if log_scale else df[feat].dropna()
    label = f"log({feat})" if log_scale else feat

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Histogramme + KDE", "Box Plot"])
    fig.add_trace(go.Histogram(x=data_plot, nbinsx=50,
                               marker_color="#e94560", opacity=.75, name="Distribution"), row=1, col=1)
    fig.add_trace(go.Box(y=data_plot, marker_color="#00d4ff",
                         boxmean="sd", name="Box"), row=1, col=2)
    fig.update_layout(template="plotly_dark", height=400,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      showlegend=False, title_text=f"Distribution de {label}")
    st.plotly_chart(fig, use_container_width=True)

    # Top numerical correlations with SalePrice
    st.markdown("<div class='section-title'>Top features corrélées à SalePrice</div>", unsafe_allow_html=True)
    corr_sp = df[num_cols].corr()["SalePrice"].abs().sort_values(ascending=False).drop("SalePrice").head(15)
    fig2 = px.bar(x=corr_sp.values, y=corr_sp.index, orientation="h",
                  color=corr_sp.values, color_continuous_scale="RdYlGn",
                  labels={"x": "Corrélation (|r|)", "y": "Feature"})
    fig2.update_layout(template="plotly_dark", height=400,
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

# ── Tab 3 : Correlations ───────────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-title'>Matrice de corrélation</div>", unsafe_allow_html=True)
    top_n = st.slider("Top N features par corrélation avec SalePrice", 5, 30, 15)
    top_feats = df[num_cols].corr()["SalePrice"].abs().nlargest(top_n).index.tolist()
    corr_mat = df[top_feats].corr()

    fig3 = px.imshow(corr_mat, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     text_auto=".2f", aspect="auto")
    fig3.update_layout(template="plotly_dark", height=600,
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<div class='section-title'>Scatter plot</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        x_feat = st.selectbox("Variable X", num_cols, index=num_cols.index("GrLivArea") if "GrLivArea" in num_cols else 0)
    with c2:
        color_feat = st.selectbox("Couleur par", ["OverallQual", "Neighborhood", "YearBuilt"] + cat_cols[:5])
    fig4 = px.scatter(df, x=x_feat, y="SalePrice", color=color_feat,
                      opacity=.6, trendline="ols",
                      color_continuous_scale="Turbo" if df[color_feat].dtype != object else None)
    fig4.update_layout(template="plotly_dark", height=450,
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig4, use_container_width=True)

# ── Tab 4 : Missing values ─────────────────────────────────────────────────────
with tab4:
    st.markdown("<div class='section-title'>Valeurs manquantes</div>", unsafe_allow_html=True)
    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)

    if miss.empty:
        st.success("✅ Aucune valeur manquante !")
    else:
        miss_pct = (miss / len(df) * 100).round(2)
        miss_df = pd.DataFrame({"Manquants": miss, "Pourcentage (%)": miss_pct, "Type": df[miss.index].dtypes})
        st.dataframe(miss_df.style.background_gradient(subset=["Pourcentage (%)"], cmap="Reds"),
                     use_container_width=True)

        fig5 = px.bar(x=miss_pct.values[:20], y=miss_pct.index[:20], orientation="h",
                      color=miss_pct.values[:20], color_continuous_scale="Reds",
                      labels={"x": "% manquants", "y": "Feature"}, title="Top 20 features avec valeurs manquantes")
        fig5.update_layout(template="plotly_dark", height=450,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig5, use_container_width=True)

# ── Tab 5 : Target analysis ────────────────────────────────────────────────────
with tab5:
    st.markdown("<div class='section-title'>Analyse de la variable cible : SalePrice</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig6 = go.Figure()
        fig6.add_trace(go.Histogram(x=df["SalePrice"], nbinsx=60,
                                    marker_color="#e94560", opacity=.8, name="SalePrice"))
        fig6.update_layout(template="plotly_dark", title="Distribution SalePrice",
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320)
        st.plotly_chart(fig6, use_container_width=True)
    with c2:
        fig7 = go.Figure()
        fig7.add_trace(go.Histogram(x=np.log1p(df["SalePrice"]), nbinsx=60,
                                    marker_color="#00d4ff", opacity=.8, name="log(SalePrice)"))
        fig7.update_layout(template="plotly_dark", title="Distribution log(SalePrice)",
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320)
        st.plotly_chart(fig7, use_container_width=True)

    # Price by neighborhood
    st.markdown("<div class='section-title'>Prix moyen par quartier</div>", unsafe_allow_html=True)
    neigh = df.groupby("Neighborhood")["SalePrice"].median().sort_values(ascending=False).reset_index()
    fig8 = px.bar(neigh, x="Neighborhood", y="SalePrice", color="SalePrice",
                  color_continuous_scale="RdYlGn",
                  labels={"SalePrice": "Prix médian ($)"},
                  title="Prix médian par quartier")
    fig8.update_layout(template="plotly_dark", height=400,
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       xaxis_tickangle=-45)
    st.plotly_chart(fig8, use_container_width=True)

    # Price by year
    yr = df.groupby("YearBuilt")["SalePrice"].median().reset_index()
    fig9 = px.line(yr, x="YearBuilt", y="SalePrice", title="Évolution du prix médian selon l'année de construction",
                   color_discrete_sequence=["#a855f7"])
    fig9.update_traces(line_width=2.5)
    fig9.update_layout(template="plotly_dark", height=350,
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig9, use_container_width=True)

st.markdown("""
<div style='text-align:center;color:#555;font-size:.8rem;margin-top:2rem'>
  📊 Page Données — Exercice 3
</div>""", unsafe_allow_html=True)

