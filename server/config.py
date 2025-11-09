import os
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# Load .env if present (at project root)
load_dotenv(dotenv_path=BASE_DIR / ".env")
DATA_DIR = BASE_DIR / "data"
SCRIPTS_DIR = BASE_DIR / "scripts"
MODELS_DIR = BASE_DIR / "server" / "models"
LOGS_DIR = BASE_DIR / "server" / "logs"

# Prefer MySQL from env; fallback to local MySQL energy_consumption; final fallback SQLite
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Please set it to your MySQL URI.")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

CRAWLER_HEADLESS = os.getenv("CRAWLER_HEADLESS", "True").lower() == "true"

CRAWLER_DAILY_TIME = "02:00"
RETRAIN_YEARLY_DAY = 1
RETRAIN_YEARLY_MONTH = 1
RETRAIN_YEARLY_TIME = "03:00"

YEARS_BACK_FOR_TRAINING = 3

USE_WOA_BY_DEFAULT = True
WOA_N_AGENTS = 5
WOA_MAX_ITER = 10

