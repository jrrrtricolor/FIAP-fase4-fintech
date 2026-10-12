from ml_prep_kit import (
    ModelStore,
)
from .api_constants import (
    DEFAULT_MODEL_PATH,
    MLFLOW_MODEL_NAME
)
def main():
    """Le a base do MLflow e disponibiliza o modelo para build da imagem docker"""
    model_store = ModelStore.from_directory(
        model_name=MLFLOW_MODEL_NAME,
        directory="data/processed/monitoring"
    )

    model_store.export(DEFAULT_MODEL_PATH)

if __name__ == "__main__":
    main()
