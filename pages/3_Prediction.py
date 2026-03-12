import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🔮 Prédiction", page_icon="🔮", layout="wide")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
  [data-testid="stSidebar"] {
      background: linear-gradient(180deg,#1a1a2e,#16213e,#0f3460);
      border-right: 1px solid #00d4ff;
  }
  [data-testid="stSidebar"] * { color: #e0e0e0 !important; }

  .page-header {
      background: linear-gradient(135deg,#1a1a2e,#16213e);
      border:1px solid #00d4ff; border-radius:16px;
      padding:1.8rem 2rem; margin-bottom:1.5rem;
      box-shadow: 0 0 30px rgba(0,212,255,.25);
  }
  .page-title { font-size:2.2rem; font-weight:900;
      background: linear-gradient(90deg,#00d4ff,#00b4d8);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent; }

  .predict-result {
      background: linear-gradient(135deg,#0f3460,#16213e);
      border: 2px solid #00d4ff; border-radius:20px;
      padding:2rem; text-align:center;
      box-shadow: 0 0 40px rgba(0,212,255,.3);
      margin:1.5rem 0;
  }
  .predict-price {
      font-size:3.5rem; font-weight:900;
      background: linear-gradient(90deg,#00d4ff,#a855f7);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  }
  .predict-range { color:#888; font-size:1rem; margin-top:.5rem; }

  .section-title {
      color:#00d4ff; font-size:1.3rem; font-weight:800;
      border-left:4px solid #00d4ff; padding-left:.7rem;
      margin:1.8rem 0 1rem;
  }
  .input-group {
      background:rgba(255,255,255,.04);
      border:1px solid rgba(255,255,255,.1);
      border-radius:12px; padding:1.2rem; margin-bottom:.8rem;
  }
  h1,h2,h3,h4 { color:#fff !important; }
  p,li { color:#c0c0d0 !important; }
  .stTabs [data-baseweb="tab"] { color:#888 !important; }
  .stTabs [aria-selected="true"] { color:#00d4ff !important; border-bottom:2px solid #00d4ff !important; }
  div[data-testid="stNumberInput"] label,
  div[data-testid="stSelectbox"] label { color:#ccc !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
  <div class='page-title'>🔮 Interface de Prédiction</div>
  <div style='color:#aaa;margin-top:.4rem'>Prédiction manuelle · Batch · Comparaison</div>
</div>
""", unsafe_allow_html=True)

# ── Check model ────────────────────────────────────────────────────────────────
if "trained_model" not in st.session_state:
    st.markdown("""
    <div style='background:rgba(0,212,255,.08);border:1px solid #00d4ff;
                border-radius:12px;padding:2rem;text-align:center;margin-top:2rem'>
        <div style='font-size:3rem'>⚠️</div>
        <div style='color:#00d4ff;font-size:1.2rem;font-weight:700'>
            Aucun modèle entraîné</div>
        <div style='color:#888;margin-top:.5rem'>
            Veuillez d'abord entraîner un modèle sur la <b>Page 2 · Entraînement</b></div>
    </div>""", unsafe_allow_html=True)
    st.stop()

model = st.session_state["trained_model"]
feat_names = st.session_state["feature_names"]
use_log = st.session_state.get("use_log", True)
model_name = st.session_state.get("model_name", "Modèle")
r2 = st.session_state.get("r2", 0)
rmse_real = st.session_state.get("rmse_real", 0)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔮 Prédiction")
    st.markdown("---")
    st.markdown(f"**Modèle actif :** {model_name}")
    st.markdown(f"**R² :** `{r2:.4f}`")
    st.markdown(f"**RMSE :** `${rmse_real/1000:.1f}K`")
    st.markdown("---")
    pred_mode = st.radio("Mode de prédiction", ["✏️ Manuelle", "📁 Batch CSV"])

# ── Encode helper ──────────────────────────────────────────────────────────────
CAT_OPTS = {
    "MSZoning":    ["RL", "RM", "C (all)", "FV", "RH"],
    "Street":      ["Pave", "Grvl"],
    "LotShape":    ["Reg", "IR1", "IR2", "IR3"],
    "LandContour": ["Lvl", "Bnk", "HLS", "Low"],
    "Utilities":   ["AllPub", "NoSewr", "NoSeWa"],
    "LotConfig":   ["Inside", "Corner", "CulDSac", "FR2", "FR3"],
    "LandSlope":   ["Gtl", "Mod", "Sev"],
    "Neighborhood":["CollgCr","Veenker","Crawfor","NoRidge","Mitchel",
                    "Somerst","NWAmes","OldTown","BrkSide","Sawyer","NridgHt",
                    "NAmes","SawyerW","IDOTRR","MeadowV","Edwards","Timber",
                    "Gilbert","StoneBr","ClearCr","NPkVill","Blmngtn","BrDale",
                    "SWISU","Blueste"],
    "BldgType":    ["1Fam","2fmCon","Duplx","TwnhsE","TwnhsI"],
    "HouseStyle":  ["2Story","1Story","1.5Fin","1.5Unf","SFoyer","SLvl","2.5Unf","2.5Fin"],
    "ExterQual":   ["Ex","Gd","TA","Fa","Po"],
    "ExterCond":   ["Ex","Gd","TA","Fa","Po"],
    "Foundation":  ["PConc","CBlock","BrkTil","Wood","Slab","Stone"],
    "Heating":     ["GasA","GasW","Grav","Wall","OthW","Floor"],
    "HeatingQC":   ["Ex","Gd","TA","Fa","Po"],
    "CentralAir":  ["Y","N"],
    "KitchenQual": ["Ex","Gd","TA","Fa","Po"],
    "Functional":  ["Typ","Min1","Min2","Mod","Maj1","Maj2","Sev","Sal"],
    "PavedDrive":  ["Y","P","N"],
    "SaleType":    ["WD","CWD","VWD","COD","Con","ConLw","ConLI","ConLD","Oth"],
    "SaleCondition":["Normal","Abnorml","AdjLand","Alloca","Family","Partial"],
    "GarageType":  ["Attchd","Detchd","BuiltIn","CarPort","NA","Basment","2Types"],
    "GarageFinish":["Fin","RFn","Unf","NA"],
    "GarageQual":  ["Ex","Gd","TA","Fa","Po","NA"],
    "GarageCond":  ["Ex","Gd","TA","Fa","Po","NA"],
}

def encode_row(row_dict, feat_names):
    """Encode a single input dict to a feature array."""
    df_row = pd.DataFrame([row_dict])
    for col in df_row.select_dtypes(include="object").columns:
        df_row[col] = LabelEncoder().fit_transform(df_row[col].astype(str))
    df_row = df_row.reindex(columns=feat_names, fill_value=0)
    imp = SimpleImputer(strategy="constant", fill_value=0)
    return imp.fit_transform(df_row)

# ── Mode 1: Manual ─────────────────────────────────────────────────────────────
if "✏️" in pred_mode:
    st.markdown("<div class='section-title'>✏️ Saisie manuelle des caractéristiques</div>", unsafe_allow_html=True)

    # Form in tabs for clarity
    t1, t2, t3 = st.tabs(["🏗️ Structure", "🔧 Équipements", "📍 Localisation"])

    with t1:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("<div class='input-group'>", unsafe_allow_html=True)
            overall_qual   = st.slider("Qualité globale (1–10)", 1, 10, 7)
            overall_cond   = st.slider("Condition globale (1–10)", 1, 10, 5)
            year_built     = st.number_input("Année de construction", 1872, 2010, 2000)
            year_remod     = st.number_input("Année rénovation", 1950, 2010, 2000)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='input-group'>", unsafe_allow_html=True)
            house_style    = st.selectbox("Style de maison", CAT_OPTS["HouseStyle"])
            bldg_type      = st.selectbox("Type de bâtiment", CAT_OPTS["BldgType"])
            lot_area       = st.number_input("Surface terrain (pi²)", 1000, 200000, 8000)
            lot_frontage   = st.number_input("Façade terrain (pi)", 0, 300, 65)
            st.markdown("</div>", unsafe_allow_html=True)
        with c3:
            st.markdown("<div class='input-group'>", unsafe_allow_html=True)
            gr_liv_area    = st.number_input("Surface habitable (pi²)", 300, 6000, 1500)
            total_bsmt_sf  = st.number_input("Surface sous-sol (pi²)", 0, 5000, 800)
            first_flr_sf   = st.number_input("1er étage (pi²)", 300, 5000, 900)
            second_flr_sf  = st.number_input("2ème étage (pi²)", 0, 3000, 600)
            st.markdown("</div>", unsafe_allow_html=True)

    with t2:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("<div class='input-group'>", unsafe_allow_html=True)
            full_bath      = st.number_input("Salles de bain complètes", 0, 4, 2)
            half_bath      = st.number_input("Demi-salles de bain", 0, 2, 0)
            bedroom        = st.number_input("Chambres", 0, 10, 3)
            kitchen        = st.number_input("Cuisines", 1, 3, 1)
            kitchen_qual   = st.selectbox("Qualité cuisine", CAT_OPTS["KitchenQual"])
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='input-group'>", unsafe_allow_html=True)
            garage_cars    = st.number_input("Capacité garage (voitures)", 0, 4, 2)
            garage_area    = st.number_input("Surface garage (pi²)", 0, 1500, 480)
            garage_type    = st.selectbox("Type garage", CAT_OPTS["GarageType"])
            garage_finish  = st.selectbox("Finition garage", CAT_OPTS["GarageFinish"])
            garage_yr      = st.number_input("Année construction garage", 1900, 2010, 2000)
            st.markdown("</div>", unsafe_allow_html=True)
        with c3:
            st.markdown("<div class='input-group'>", unsafe_allow_html=True)
            fireplaces     = st.number_input("Cheminées", 0, 4, 0)
            wood_deck      = st.number_input("Terrasse bois (pi²)", 0, 1500, 0)
            open_porch     = st.number_input("Porche ouvert (pi²)", 0, 600, 0)
            pool_area      = st.number_input("Piscine (pi²)", 0, 800, 0)
            central_air    = st.selectbox("Climatisation centrale", CAT_OPTS["CentralAir"])
            st.markdown("</div>", unsafe_allow_html=True)

    with t3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='input-group'>", unsafe_allow_html=True)
            neighborhood   = st.selectbox("Quartier", CAT_OPTS["Neighborhood"])
            ms_zoning      = st.selectbox("Zonage", CAT_OPTS["MSZoning"])
            sale_type      = st.selectbox("Type de vente", CAT_OPTS["SaleType"])
            sale_cond      = st.selectbox("Condition de vente", CAT_OPTS["SaleCondition"])
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='input-group'>", unsafe_allow_html=True)
            foundation     = st.selectbox("Fondation", CAT_OPTS["Foundation"])
            exterior_qual  = st.selectbox("Qualité extérieur", CAT_OPTS["ExterQual"])
            heating_qc     = st.selectbox("Qualité chauffage", CAT_OPTS["HeatingQC"])
            mo_sold        = st.slider("Mois de vente", 1, 12, 6)
            yr_sold        = st.selectbox("Année de vente", [2006,2007,2008,2009,2010])
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Build prediction row ───────────────────────────────────────────────────
    input_dict = {
        "MSSubClass": 60, "MSZoning": ms_zoning, "LotFrontage": lot_frontage,
        "LotArea": lot_area, "Street": "Pave", "Alley": "NA",
        "LotShape": "Reg", "LandContour": "Lvl", "Utilities": "AllPub",
        "LotConfig": "Inside", "LandSlope": "Gtl",
        "Neighborhood": neighborhood, "Condition1": "Norm", "Condition2": "Norm",
        "BldgType": bldg_type, "HouseStyle": house_style,
        "OverallQual": overall_qual, "OverallCond": overall_cond,
        "YearBuilt": year_built, "YearRemodAdd": year_remod,
        "RoofStyle": "Gable", "RoofMatl": "CompShg",
        "Exterior1st": "VinylSd", "Exterior2nd": "VinylSd",
        "MasVnrType": "None", "MasVnrArea": 0,
        "ExterQual": exterior_qual, "ExterCond": "TA",
        "Foundation": foundation,
        "BsmtQual": "Gd", "BsmtCond": "TA", "BsmtExposure": "No",
        "BsmtFinType1": "GLQ", "BsmtFinSF1": 500,
        "BsmtFinType2": "Unf", "BsmtFinSF2": 0,
        "BsmtUnfSF": 300, "TotalBsmtSF": total_bsmt_sf,
        "Heating": "GasA", "HeatingQC": heating_qc, "CentralAir": central_air,
        "Electrical": "SBrkr",
        "1stFlrSF": first_flr_sf, "2ndFlrSF": second_flr_sf, "LowQualFinSF": 0,
        "GrLivArea": gr_liv_area,
        "BsmtFullBath": 1, "BsmtHalfBath": 0,
        "FullBath": full_bath, "HalfBath": half_bath,
        "BedroomAbvGr": bedroom, "KitchenAbvGr": kitchen, "KitchenQual": kitchen_qual,
        "TotRmsAbvGrd": bedroom + 2, "Functional": "Typ",
        "Fireplaces": fireplaces, "FireplaceQu": "NA",
        "GarageType": garage_type, "GarageYrBlt": garage_yr,
        "GarageFinish": garage_finish,
        "GarageCars": garage_cars, "GarageArea": garage_area,
        "GarageQual": "TA", "GarageCond": "TA", "PavedDrive": "Y",
        "WoodDeckSF": wood_deck, "OpenPorchSF": open_porch,
        "EnclosedPorch": 0, "3SsnPorch": 0, "ScreenPorch": 0,
        "PoolArea": pool_area, "PoolQC": "NA", "Fence": "NA",
        "MiscFeature": "NA", "MiscVal": 0,
        "MoSold": mo_sold, "YrSold": yr_sold,
        "SaleType": sale_type, "SaleCondition": sale_cond,
    }

    if st.button("🔮 Prédire le prix", use_container_width=True, type="primary"):
        X_input = encode_row(input_dict, feat_names)
        pred = model.predict(X_input)[0]
        price = np.expm1(pred) if use_log else pred
        low, high = price * 0.90, price * 1.10

        st.markdown(f"""
        <div class='predict-result'>
            <div style='color:#00d4ff;font-size:1rem;font-weight:600;letter-spacing:2px'>
                PRIX ESTIMÉ</div>
            <div class='predict-price'>${price:,.0f}</div>
            <div class='predict-range'>
                Intervalle de confiance estimé : ${low:,.0f} — ${high:,.0f}
            </div>
            <div style='color:#888;font-size:.85rem;margin-top:.8rem'>
                Modèle : {model_name} · R² : {r2:.4f}
            </div>
        </div>""", unsafe_allow_html=True)

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=price,
            delta={"reference": 180000, "valueformat": ",.0f"},
            number={"prefix": "$", "valueformat": ",.0f"},
            gauge={
                "axis": {"range": [50000, 500000],
                         "tickformat": ",.0f",
                         "tickprefix": "$"},
                "bar": {"color": "#00d4ff"},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [50000, 150000], "color": "rgba(233,69,96,.2)"},
                    {"range": [150000, 250000], "color": "rgba(168,85,247,.2)"},
                    {"range": [250000, 500000], "color": "rgba(16,185,129,.2)"},
                ],
                "threshold": {"line": {"color": "#e94560", "width": 3}, "value": price},
            },
            title={"text": "Positionnement du prix", "font": {"color": "#ccc"}},
        ))
        fig.update_layout(template="plotly_dark", height=300,
                          paper_bgcolor="rgba(0,0,0,0)", font={"color": "#fff"})
        st.plotly_chart(fig, use_container_width=True)

# ── Mode 2: Batch CSV ──────────────────────────────────────────────────────────
else:
    st.markdown("<div class='section-title'>📁 Prédiction Batch (CSV)</div>", unsafe_allow_html=True)
    batch_file = st.file_uploader("Uploader test.csv", type="csv")

    if batch_file is None and "test_df" in st.session_state:
        st.info("💡 Utilisation du test.csv chargé en Page 1")
        df_batch = st.session_state["test_df"]
    elif batch_file:
        df_batch = pd.read_csv(batch_file)
    else:
        st.warning("Uploadez test.csv pour les prédictions batch.")
        st.stop()

    ids = df_batch["Id"] if "Id" in df_batch.columns else pd.RangeIndex(len(df_batch))
    df_proc = df_batch.drop(columns=["Id"], errors="ignore")

    for col in df_proc.select_dtypes(include="object").columns:
        df_proc[col] = LabelEncoder().fit_transform(df_proc[col].astype(str))

    df_proc = df_proc.reindex(columns=feat_names, fill_value=0)
    imp2 = SimpleImputer(strategy="median")
    X_batch = imp2.fit_transform(df_proc)

    preds = model.predict(X_batch)
    prices = np.expm1(preds) if use_log else preds

    results = pd.DataFrame({"Id": ids, "SalePrice_Predicted": prices.astype(int)})
    st.success(f"✅ {len(results):,} prédictions générées !")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div style='background:rgba(0,212,255,.08);border:1px solid #00d4ff;
            border-radius:12px;padding:1rem;text-align:center'>
            <div style='color:#00d4ff;font-size:1.8rem;font-weight:900'>${prices.mean():,.0f}</div>
            <div style='color:#888;font-size:.85rem'>Prix moyen prédit</div></div>""",
            unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div style='background:rgba(168,85,247,.08);border:1px solid #a855f7;
            border-radius:12px;padding:1rem;text-align:center'>
            <div style='color:#a855f7;font-size:1.8rem;font-weight:900'>${prices.median():,.0f}</div>
            <div style='color:#888;font-size:.85rem'>Prix médian prédit</div></div>""",
            unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div style='background:rgba(16,185,129,.08);border:1px solid #10b981;
            border-radius:12px;padding:1rem;text-align:center'>
            <div style='color:#10b981;font-size:1.8rem;font-weight:900'>${prices.std():,.0f}</div>
            <div style='color:#888;font-size:.85rem'>Écart-type</div></div>""",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    # Histogram of predictions
    fig = px.histogram(x=prices, nbins=60, color_discrete_sequence=["#00d4ff"],
                       labels={"x": "Prix prédit ($)", "count": "Fréquence"},
                       title="Distribution des prix prédits")
    fig.update_layout(template="plotly_dark", height=380,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>📋 Résultats (Top 50)</div>", unsafe_allow_html=True)
    st.dataframe(results.head(50), use_container_width=True, height=300)

    csv_out = results.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Télécharger les prédictions (CSV)",
                        data=csv_out,
                        file_name="predictions.csv",
                        mime="text/csv",
                        use_container_width=True)

st.markdown("""
<div style='text-align:center;color:#555;font-size:.8rem;margin-top:2rem'>
  🔮 Page Prédiction — Exercice 3
</div>""", unsafe_allow_html=True)
