"""Restaura a versão anterior usando a implementação de Registry do framework."""

import os
from pathlib import Path

from ml_prep_kit import ExperimentTracker

from .api_constants import MLFLOW_MODEL_NAME


def rollback_production_model():
    """Troca o alias de produção; a API carrega a versão ao reiniciar.

    Exemplo:
        version = rollback_production_model()
    """
    database = Path.cwd() / "data/processed/monitoring/mlflow.db"
    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{database}")
    tracker = ExperimentTracker("tech-challenge-4-credit-monitoring", tracking_uri=uri)
    return tracker.rollback_model_version(MLFLOW_MODEL_NAME)


def main():
    """Executa o rollback explícito e informa a necessidade de recarregar a API.

    Exemplo:
        python -m src.credit_monitoring.rollback_model
    """
    version = rollback_production_model()
    print(
        f"Modelo {MLFLOW_MODEL_NAME}: versão {version}. Reinicie a API para carregá-la."
    )


if __name__ == "__main__":
    main()
