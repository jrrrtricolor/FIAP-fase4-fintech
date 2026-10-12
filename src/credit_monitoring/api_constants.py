from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = Path.joinpath(PROJECT_ROOT, "model")
DEFAULT_MODEL_FILE = Path.joinpath(DEFAULT_MODEL_PATH, "model.pkl")

MLFLOW_MODEL_NAME = "home_credit_model"
