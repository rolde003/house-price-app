import streamlit as st
import hashlib
import time

# ── Utilisateurs autorisés (mot de passe hashé en SHA-256) ────────────────────
# Pour générer un hash : hashlib.sha256("monmotdepasse".encode()).hexdigest()
USERS = {
    "admin": {
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "name": "Administrateur",
        "role": "admin",
    },
    "user1": {
        "password_hash": hashlib.sha256("user123".encode()).hexdigest(),
        "name": "Utilisateur 1",
        "role": "user",
    },
}

MAX_ATTEMPTS = 3
LOCKOUT_SECONDS = 60


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def check_credentials(username: str, password: str) -> bool:
    if username not in USERS:
        return False
    return USERS[username]["password_hash"] == hash_password(password)


def login_page():
    """Affiche la page de connexion et gère l'authentification."""
    st.markdown("""
    <style>
      .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
      .login-card {
          max-width: 420px; margin: 6rem auto 0;
          background: rgba(255,255,255,.04);
          border: 1px solid rgba(233,69,96,.3);
          border-radius: 20px; padding: 2.5rem;
          box-shadow: 0 0 40px rgba(233,69,96,.15);
      }
      .login-title {
          text-align:center; font-size:2rem; font-weight:900;
          background: linear-gradient(90deg,#e94560,#a855f7);
          -webkit-background-clip:text; -webkit-text-fill-color:transparent;
          margin-bottom:0.3rem;
      }
      .login-sub { text-align:center; color:#666; font-size:.9rem; margin-bottom:1.5rem; }
      h1,h2,h3 { color:#fff !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='login-card'>
        <div class='login-title'>🏠 House Price App</div>
        <div class='login-sub'>Connectez-vous pour accéder à l'application</div>
    </div>
    """, unsafe_allow_html=True)

    # Centrer le formulaire
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)

        # Vérifier le verrouillage
        if "lockout_until" in st.session_state:
            remaining = st.session_state["lockout_until"] - time.time()
            if remaining > 0:
                st.error(f"🔒 Trop de tentatives. Réessayez dans {int(remaining)}s")
                return False

        username = st.text_input("👤 Nom d'utilisateur", placeholder="admin")
        password = st.text_input("🔑 Mot de passe", type="password", placeholder="••••••••")
        login_btn = st.button("🚀 Se connecter", use_container_width=True, type="primary")

        if login_btn:
            if not username or not password:
                st.warning("⚠️ Remplissez tous les champs.")
                return False

            if check_credentials(username, password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.session_state["user_name"] = USERS[username]["name"]
                st.session_state["user_role"] = USERS[username]["role"]
                st.session_state["login_attempts"] = 0
                st.success(f"✅ Bienvenue, {USERS[username]['name']} !")
                time.sleep(0.5)
                st.rerun()
            else:
                attempts = st.session_state.get("login_attempts", 0) + 1
                st.session_state["login_attempts"] = attempts
                remaining = MAX_ATTEMPTS - attempts
                if attempts >= MAX_ATTEMPTS:
                    st.session_state["lockout_until"] = time.time() + LOCKOUT_SECONDS
                    st.error(f"🔒 Compte verrouillé pendant {LOCKOUT_SECONDS}s")
                else:
                    st.error(f"❌ Identifiants incorrects. {remaining} tentative(s) restante(s).")

        st.markdown("""
        <div style='text-align:center;color:#444;font-size:.78rem;margin-top:1rem'>
            Compte démo : <b>admin</b> / <b>admin123</b>
        </div>""", unsafe_allow_html=True)

    return False


def logout():
    """Déconnexion."""
    for key in ["authenticated", "username", "user_name", "user_role"]:
        st.session_state.pop(key, None)
    st.rerun()


def require_auth():
    """
    À appeler en haut de chaque page.
    Redirige vers login si non authentifié.
    """
    if not st.session_state.get("authenticated", False):
        login_page()
        st.stop()


def show_user_info():
    """Affiche les infos utilisateur + bouton logout dans la sidebar."""
    name = st.session_state.get("user_name", "Utilisateur")
    role = st.session_state.get("user_role", "user")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div style='background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);
                border-radius:10px;padding:.8rem;text-align:center'>
        <div style='font-size:1.5rem'>👤</div>
        <div style='color:#e0e0e0;font-weight:700;font-size:.9rem'>{name}</div>
        <div style='color:#666;font-size:.75rem'>Rôle : {role}</div>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    if st.sidebar.button("🚪 Se déconnecter", use_container_width=True):
        logout()
