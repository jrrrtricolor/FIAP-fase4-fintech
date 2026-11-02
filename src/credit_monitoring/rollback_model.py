"""Restaura a versão anterior usando a implementação de Registry do framework."""

import os
from pathlib import Path

from ml_prep_kit import ExperimentTracker

from .api_constants import MLFLOW_MODEL_NAME


def rollback_production_model():
    """Restaura o alias aprovado para a próxima atualização periódica da API.

    Exemplo:
        version = rollback_production_model()
    """
    database = Path.cwd() / "data/processed/monitoring/mlflow.db"
    uri = os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{database}"
    tracker = ExperimentTracker(
        "tech-challenge-4-credit-monitoring",
        tracking_uri=uri,
        artifact_location=os.environ.get("MLFLOW_ARTIFACT_LOCATION") or None,
    )
    return tracker.rollback_approved_model_version(MLFLOW_MODEL_NAME)


def main():
    """Executa o rollback explícito e informa a atualização periódica da API.

    Exemplo:
        python -m src.credit_monitoring.rollback_model
    """
    version = rollback_production_model()
    print(
        f"Modelo {MLFLOW_MODEL_NAME}: versão {version}. "
        "A API atualizará na próxima verificação periódica."
    )


if __name__ == "__main__":
    main()
