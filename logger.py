import logging
import os
from datetime import datetime

# ── Création du dossier logs ───────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, f"app_{datetime.now().strftime('%Y%m%d')}.log")


def get_logger(name: str = "house_price_app") -> logging.Logger:
    """Retourne un logger configuré pour l'application."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Déjà configuré

    logger.setLevel(logging.INFO)

    # Format
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler fichier
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Handler console
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ── Logger global ──────────────────────────────────────────────────────────────
logger = get_logger()


def log_prediction(username: str, model_name: str, inputs: dict, prediction: float):
    logger.info(f"PREDICTION | user={username} | model={model_name} | "
                f"inputs={inputs} | prediction=${prediction:,.0f}")


def log_training(username: str, model_name: str, params: dict,
                 r2: float, rmse: float, duration: float):
    logger.info(f"TRAINING | user={username} | model={model_name} | "
                f"params={params} | R2={r2:.4f} | RMSE=${rmse:,.0f} | "
                f"duration={duration:.2f}s")


def log_upload(username: str, filename: str, n_rows: int):
    logger.info(f"UPLOAD | user={username} | file={filename} | rows={n_rows}")


def log_login(username: str, success: bool):
    level = logging.INFO if success else logging.WARNING
    status = "SUCCESS" if success else "FAILED"
    logger.log(level, f"LOGIN {status} | user={username}")


def log_error(context: str, error: Exception):
    logger.error(f"ERROR | context={context} | {type(error).__name__}: {error}")
