import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import time, pickle

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🤖 Entraînement", page_icon="🤖", layout="wide")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
  [data-testid="stSidebar"] {
      background: linear-gradient(180deg,#1a1a2e,#16213e,#0f3460);
      border-right: 1px solid #a855f7;
  }
  [data-testid="stSidebar"] * { color: #e0e0e0 !important; }

  .page-header {
      background: linear-gradient(135deg,#1a1a2e,#16213e);
      border:1px solid #a855f7; border-radius:16px;
      padding:1.8rem 2rem; margin-bottom:1.5rem;
      box-shadow: 0 0 30px rgba(168,85,247,.25);
  }
  .page-title { font-size:2.2rem; font-weight:900;
      background: linear-gradient(90deg,#a855f7,#c084fc);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent; }

  .metric-card {
      background: rgba(255,255,255,.05);
      border-radius:14px; padding:1.2rem;
      text-align:center; border:1px solid rgba(255,255,255,.1);
  }
  .metric-value { font-size:2rem; font-weight:900; }
  .metric-label { color:#888; font-size:.85rem; margin-top:.2rem; }

  .section-title {
      color:#a855f7; font-size:1.3rem; font-weight:800;
      border-left:4px solid #a855f7; padding-left:.7rem;
      margin:1.8rem 0 1rem;
  }
  .model-badge {
      display:inline-block; padding:.3rem .9rem; border-radius:50px;
      font-weight:700; font-size:.85rem; margin:.2rem;
  }
  h1,h2,h3,h4 { color:#fff !important; }
  p,li { color:#c0c0d0 !important; }
  .stTabs [data-baseweb="tab"] { color: #888 !important; }
  .stTabs [aria-selected="true"] { color:#a855f7 !important; border-bottom:2px solid #a855f7 !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
  <div class='page-title'>🤖 Entraînement du Modèle</div>
  <div style='color:#aaa;margin-top:.4rem'>Prétraitement · Sélection du modèle · Métriques · Visualisations</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar config ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    model_choice = st.selectbox("🤖 Algorithme", [
        "Random Forest", "Gradient Boosting", "Ridge Regression", "Lasso Regression"
    ])
    test_size = st.slider("📊 Taille du jeu de test (%)", 10, 40, 20) / 100
    use_log = st.checkbox("📈 Log-transform SalePrice", value=True)
    st.markdown("---")

    if model_choice == "Random Forest":
        st.markdown("### 🌲 Hyperparamètres RF")
        n_estimators = st.slider("n_estimators", 50, 500, 200, 50)
        max_depth = st.select_slider("max_depth", [None, 5, 10, 15, 20, 30], value=None)
        min_samples_split = st.slider("min_samples_split", 2, 20, 2)
        model_params = dict(n_estimators=n_estimators, max_depth=max_depth,
                            min_samples_split=min_samples_split, random_state=42, n_jobs=-1)

    elif model_choice == "Gradient Boosting":
        st.markdown("### 🚀 Hyperparamètres GB")
        n_estimators = st.slider("n_estimators", 50, 500, 200, 50)
        learning_rate = st.select_slider("learning_rate", [0.01, 0.05, 0.1, 0.2], value=0.1)
        max_depth = st.slider("max_depth", 2, 8, 4)
        model_params = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                            max_depth=max_depth, random_state=42)

    elif model_choice == "Ridge Regression":
        st.markdown("### 📐 Hyperparamètres Ridge")
        alpha = st.select_slider("alpha", [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], value=1.0)
        model_params = dict(alpha=alpha)

    else:
        st.markdown("### 🔗 Hyperparamètres Lasso")
        alpha = st.select_slider("alpha", [0.0001, 0.001, 0.01, 0.1, 1.0], value=0.001)
        model_params = dict(alpha=alpha, max_iter=5000)

# ── Data preprocessing ─────────────────────────────────────────────────────────
@st.cache_data
def preprocess(df, use_log=True):
    df = df.copy()
    # Target
    y = np.log1p(df["SalePrice"]) if use_log else df["SalePrice"]
    df = df.drop(columns=["Id", "SalePrice"], errors="ignore")

    # Encode categoricals
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Impute numerics
    imp = SimpleImputer(strategy="median")
    X = pd.DataFrame(imp.fit_transform(df), columns=df.columns)
    return X, y, imp

# ── Check session state for data ───────────────────────────────────────────────
if "train_df" not in st.session_state:
    st.markdown("""
    <div style='background:rgba(168,85,247,.1);border:1px solid #a855f7;border-radius:12px;
                padding:2rem;text-align:center;margin-top:2rem'>
        <div style='font-size:3rem'>⚠️</div>
        <div style='color:#a855f7;font-size:1.2rem;font-weight:700'>
            Données non chargées</div>
        <div style='color:#888;margin-top:.5rem'>
            Veuillez d'abord charger les données sur la <b>Page 1 · Données</b></div>
    </div>""", unsafe_allow_html=True)
    # Allow file upload here too
    uploaded = st.file_uploader("Ou chargez train.csv directement ici", type="csv")
    if uploaded:
        st.session_state["train_df"] = pd.read_csv(uploaded)
        st.rerun()
    st.stop()

df = st.session_state["train_df"]
X, y, imputer = preprocess(df, use_log)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

st.markdown(f"""
<div style='background:rgba(16,185,129,.08);border:1px solid #10b981;
            border-radius:10px;padding:.8rem 1.2rem;margin-bottom:1rem'>
  ✅ Dataset chargé — <b>{len(df):,}</b> observations · <b>{X.shape[1]}</b> features
  · Train: <b>{len(X_train):,}</b> · Test: <b>{len(X_test):,}</b>
</div>""", unsafe_allow_html=True)

# ── Train button ───────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Lancement de l'entraînement</div>", unsafe_allow_html=True)
train_btn = st.button(f"🚀 Entraîner {model_choice}", use_container_width=True,
                       type="primary")

# ── Build & train ──────────────────────────────────────────────────────────────
def build_model(name, params):
    if name == "Random Forest":
        return RandomForestRegressor(**params)
    elif name == "Gradient Boosting":
        return GradientBoostingRegressor(**params)
    elif name == "Ridge Regression":
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(**params))])
    else:
        return Pipeline([("scaler", StandardScaler()), ("model", Lasso(**params))])

if train_btn or "trained_model" in st.session_state:
    if train_btn:
        with st.spinner(f"⏳ Entraînement de {model_choice} en cours..."):
            t0 = time.time()
            model = build_model(model_choice, model_params)
            model.fit(X_train, y_train)
            train_time = time.time() - t0

            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)

            if use_log:
                rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                rmse_log = rmse
                rmse_real = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred_test)))
            else:
                rmse_real = np.sqrt(mean_squared_error(y_test, y_pred_test))
                rmse_log = None

            r2 = r2_score(y_test, y_pred_test)
            mae = mean_absolute_error(y_test, y_pred_test)
            r2_train = r2_score(y_train, y_pred_train)

            st.session_state.update({
                "trained_model": model,
                "model_name": model_choice,
                "model_params": model_params,
                "X_train": X_train, "X_test": X_test,
                "y_train": y_train, "y_test": y_test,
                "y_pred_test": y_pred_test, "y_pred_train": y_pred_train,
                "rmse_real": rmse_real, "rmse_log": rmse_log,
                "r2": r2, "r2_train": r2_train, "mae": mae,
                "train_time": train_time,
                "feature_names": X.columns.tolist(),
                "use_log": use_log, "imputer": imputer,
            })
        st.success(f"✅ Modèle entraîné en {train_time:.2f}s !")

    # ── Retrieve results ────────────────────────────────────────────────────────
    r = st.session_state
    y_pred_test = r["y_pred_test"]
    y_test_s    = r["y_test"]
    y_pred_train = r["y_pred_train"]
    y_train_s   = r["y_train"]

    # ── Metrics ─────────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>📊 Métriques de performance</div>", unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    metrics = [
        ("#e94560",  f"${r['rmse_real']/1000:.1f}K", "RMSE (réel)"),
        ("#a855f7",  f"{r['rmse_log']:.4f}" if r['rmse_log'] else "—", "RMSE (log)"),
        ("#00d4ff",  f"{r['r2']:.4f}", "R² Test"),
        ("#10b981",  f"{r['r2_train']:.4f}", "R² Train"),
        ("#f59e0b",  f"{r['train_time']:.2f}s", "Temps"),
    ]
    for col, (color, val, label) in zip([c1,c2,c3,c4,c5], metrics):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:{color}'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    if r['r2'] > 0.9:
        st.success(f"🏆 Excellent ! R² = {r['r2']:.4f}")
    elif r['r2'] > 0.8:
        st.info(f"👍 Bon modèle ! R² = {r['r2']:.4f}")
    else:
        st.warning(f"⚠️ Modèle perfectible. R² = {r['r2']:.4f}")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prédictions vs Réel", "📉 Résidus", "🌟 Feature Importance", "📚 Courbe d'apprentissage"])

    with tab1:
        y_real = np.expm1(y_test_s) if use_log else y_test_s
        y_pred_real = np.expm1(y_pred_test) if use_log else y_pred_test

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_real, y=y_pred_real, mode="markers",
                                 marker=dict(color="#e94560", opacity=.5, size=5),
                                 name="Prédictions"))
        min_v, max_v = y_real.min(), y_real.max()
        fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v],
                                 mode="lines", line=dict(color="#00d4ff", dash="dash", width=2),
                                 name="Ligne parfaite"))
        fig.update_layout(template="plotly_dark", height=450,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          title="Prédictions vs Valeurs réelles",
                          xaxis_title="Valeur réelle ($)", yaxis_title="Prédiction ($)")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        residuals = y_pred_test - y_test_s
        c1, c2 = st.columns(2)
        with c1:
            fig2 = px.scatter(x=y_pred_test, y=residuals, opacity=.5,
                              color=np.abs(residuals), color_continuous_scale="RdYlGn_r",
                              labels={"x": "Prédiction", "y": "Résidu"},
                              title="Résidus vs Prédictions")
            fig2.add_hline(y=0, line_dash="dash", line_color="#00d4ff")
            fig2.update_layout(template="plotly_dark", height=380,
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)
        with c2:
            fig3 = px.histogram(x=residuals, nbins=50, color_discrete_sequence=["#a855f7"],
                                title="Distribution des résidus",
                                labels={"x": "Résidu", "count": "Fréquence"})
            fig3.update_layout(template="plotly_dark", height=380,
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        model_obj = r["trained_model"]
        feat_names = r["feature_names"]
        if hasattr(model_obj, "feature_importances_"):
            fi = model_obj.feature_importances_
        elif hasattr(model_obj, "named_steps"):
            inner = model_obj.named_steps.get("model")
            fi = getattr(inner, "coef_", None)
            if fi is not None: fi = np.abs(fi)
        else:
            fi = None

        if fi is not None:
            top = pd.Series(fi, index=feat_names).nlargest(20)
            fig4 = px.bar(x=top.values, y=top.index, orientation="h",
                          color=top.values, color_continuous_scale="Plasma",
                          labels={"x": "Importance", "y": "Feature"},
                          title="Top 20 features importantes")
            fig4.update_layout(template="plotly_dark", height=500,
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Feature importance non disponible pour ce modèle.")

    with tab4:
        st.info("⏳ Calcul de la courbe d'apprentissage (peut prendre quelques secondes)...")
        model_lc = build_model(r["model_name"], r["model_params"])
        train_sizes, train_scores, val_scores = learning_curve(
            model_lc, r["X_train"], r["y_train"],
            train_sizes=np.linspace(.1, 1.0, 8), cv=3, scoring="r2",
            n_jobs=-1
        )
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=train_sizes, y=train_scores.mean(axis=1),
                                  mode="lines+markers", name="Train R²",
                                  line=dict(color="#e94560", width=2.5)))
        fig5.add_trace(go.Scatter(x=train_sizes, y=val_scores.mean(axis=1),
                                  mode="lines+markers", name="Val R²",
                                  line=dict(color="#00d4ff", width=2.5)))
        fig5.update_layout(template="plotly_dark", height=400,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           title="Courbe d'apprentissage (R²)",
                           xaxis_title="Taille du jeu d'entraînement",
                           yaxis_title="R²")
        st.plotly_chart(fig5, use_container_width=True)

else:
    st.markdown("""
    <div style='background:rgba(168,85,247,.08);border:1px dashed #a855f7;
                border-radius:12px;padding:2rem;text-align:center'>
        <div style='font-size:3rem'>🤖</div>
        <div style='color:#a855f7;font-size:1.1rem;font-weight:700;margin:.5rem 0'>
            Aucun modèle entraîné</div>
        <div style='color:#888'>Configurez les hyperparamètres dans la barre latérale
            puis cliquez sur <b>Entraîner</b></div>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;color:#555;font-size:.8rem;margin-top:2rem'>
  🤖 Page Entraînement — Exercice 3
</div>""", unsafe_allow_html=True)

