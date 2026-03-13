# 🏠 House Price Predictor — ML App

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.20-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

> Application ML multi-pages pour prédire le prix de vente de maisons (Ames Housing Dataset)

---

## ✨ Fonctionnalités

### 🔐 Sécurité
- Authentification login / mot de passe (hash SHA-256)
- Verrouillage après 3 tentatives échouées
- HTTPS automatique sur Streamlit Cloud
- Validation de toutes les entrées utilisateur
- Logs horodatés de toutes les actions

### 📊 Page 1 — Données
- Upload CSV drag & drop
- Statistiques descriptives + distributions
- Matrice de corrélation interactive
- Analyse des valeurs manquantes

### 🤖 Page 2 — Entraînement
- 4 algorithmes : Random Forest · Gradient Boosting · Ridge · Lasso
- Métriques : RMSE · R² · MAE
- Feature importance + Courbe d'apprentissage

### 🔮 Page 3 — Prédiction
- Formulaire manuel (3 onglets)
- Prédiction batch CSV + export
- Jauge de prix interactive

---

## 🚀 Installation locale

```bash
# Cloner le repo
git clone https://github.com/VOTRE_USERNAME/house-price-app.git
cd house-price-app

# Environnement virtuel
python3 -m venv .venv
source .venv/bin/activate

# Dépendances
pip install -r requirements.txt

# Lancer
streamlit run app.py
```

Compte démo : **admin** / **admin123**

---

## 📁 Structure

```
app/
├── app.py                 ← Accueil + authentification
├── auth.py                ← Module de login
├── validation.py          ← Validation des entrées
├── logger.py              ← Système de logs
├── pages/
│   ├── 1_Data.py
│   ├── 2_Training.py
│   └── 3_Prediction.py
├── .streamlit/
│   └── config.toml        ← Thème dark
├── requirements.txt
├── train.csv
└── test.csv
```

---

## 🔒 Sécurité

| Mesure | Statut |
|--------|--------|
| HTTPS | ✅ Streamlit Cloud |
| Auth SHA-256 | ✅ |
| Anti-brute force | ✅ Lockout 60s |
| Validation entrées | ✅ |
| Logs | ✅ Fichier quotidien |

---

## 🤖 Performances typiques

| Modèle | R² | RMSE |
|--------|----|------|
| Random Forest | ~0.88 | ~$22K |
| Gradient Boosting | ~0.90 | ~$19K |
| Ridge | ~0.82 | ~$28K |

---

## 🛠️ Stack technique

Streamlit · scikit-learn · Plotly · Pandas · NumPy

---

Made with ❤️ — Exercice 3 ML Multi-Pages
