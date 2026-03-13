import pandas as pd
import numpy as np
from typing import Tuple, List

# ── Limites acceptées pour chaque feature numérique du dataset Iris ─────────────
IRIS_BOUNDS = {
    "sepal_length": (4.0, 8.0),
    "sepal_width":  (2.0, 4.5),
    "petal_length": (1.0, 7.0),
    "petal_width":  (0.1, 2.5),
}

# ── Limites pour les features clés du dataset House Prices ──────────────────────
HOUSE_BOUNDS = {
    "OverallQual":  (1,   10),
    "OverallCond":  (1,   10),
    "YearBuilt":    (1872, 2010),
    "GrLivArea":    (300, 6000),
    "LotArea":      (1000, 200000),
    "GarageCars":   (0,   4),
    "GarageArea":   (0,   1500),
    "TotalBsmtSF":  (0,   5000),
    "1stFlrSF":     (300, 5000),
    "FullBath":     (0,   4),
    "BedroomAbvGr": (0,   10),
}


def validate_iris_inputs(sepal_length, sepal_width, petal_length, petal_width
                         ) -> Tuple[bool, List[str]]:
    """Valide les entrées du classifieur Iris. Retourne (valide, liste_erreurs)."""
    errors = []
    values = {
        "sepal_length": sepal_length,
        "sepal_width":  sepal_width,
        "petal_length": petal_length,
        "petal_width":  petal_width,
    }
    for name, val in values.items():
        lo, hi = IRIS_BOUNDS[name]
        if not (lo <= val <= hi):
            errors.append(f"⚠️ {name.replace('_',' ').title()} : {val} hors de [{lo}, {hi}]")
    return len(errors) == 0, errors


def validate_csv_upload(df: pd.DataFrame, required_cols: List[str]
                        ) -> Tuple[bool, List[str]]:
    """Vérifie qu'un CSV uploadé contient les colonnes requises et n'est pas vide."""
    errors = []
    if df is None or df.empty:
        errors.append("❌ Le fichier CSV est vide.")
        return False, errors
    if len(df) > 10_000:
        errors.append(f"❌ Fichier trop grand : {len(df)} lignes (max 10 000).")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        errors.append(f"❌ Colonnes manquantes : {', '.join(missing)}")
    return len(errors) == 0, errors


def validate_house_inputs(input_dict: dict) -> Tuple[bool, List[str]]:
    """Valide les entrées manuelles pour la prédiction de prix."""
    errors = []
    for feat, (lo, hi) in HOUSE_BOUNDS.items():
        val = input_dict.get(feat)
        if val is not None and not (lo <= val <= hi):
            errors.append(f"⚠️ {feat} : {val} hors de [{lo}, {hi}]")
    return len(errors) == 0, errors


def sanitize_text(text: str, max_length: int = 200) -> str:
    """Nettoie une entrée texte (supprime HTML/scripts)."""
    import re
    text = re.sub(r"<[^>]+>", "", text)          # strip HTML
    text = re.sub(r"[<>\"'%;()&+]", "", text)    # strip chars dangereux
    return text[:max_length].strip()
