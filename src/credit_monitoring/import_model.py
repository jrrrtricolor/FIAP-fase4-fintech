import argparse
from ml_prep_kit import (
    ModelStore,
)
from .api_constants import (
    DEFAULT_MODEL_PATH,
    MLFLOW_MODEL_NAME
)
def main(rollback):
    model_store = ModelStore.from_directory(
        model_name=MLFLOW_MODEL_NAME,
        directory="data/processed/monitoring"
    )

    if (rollback):
        model_store.export(DEFAULT_MODEL_PATH, "production_rollback")
    else:
        model_store.export(DEFAULT_MODEL_PATH)

if __name__ == "__main__":
    """Le a base do MLflow e disponibiliza o modelo para build da imagem docker

    Exemplo:
        python -m src.credit_monitoring.import_model --rollback
    """

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rollback", type=bool, default=False)

    main(**vars(parser.parse_args()))
