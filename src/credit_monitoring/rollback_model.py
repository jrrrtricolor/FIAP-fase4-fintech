"""Restore the model version held by the production_rollback alias."""

import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from .api_constants import MLFLOW_MODEL_NAME

PRODUCTION_ALIAS = "production"
ROLLBACK_ALIAS = "production_rollback"


def rollback_production_model():
	"""Move production to the rollback version and preserve the current version."""
	default_tracking_file = Path.cwd() / "data/processed/monitoring/mlflow.db"
	tracking_uri = os.environ.get(
		"MLFLOW_TRACKING_URI", f"sqlite:///{default_tracking_file}"
	)
	mlflow.set_tracking_uri(tracking_uri)
	client = MlflowClient()

	versions = client.search_model_versions(f"name = '{MLFLOW_MODEL_NAME}'")
	if not versions:
		raise ValueError(f"Modelo registrado não encontrado: {MLFLOW_MODEL_NAME}.")

	rollback_version = next(
		(version for version in versions if ROLLBACK_ALIAS in version.aliases),
		None,
	)
	if rollback_version is None:
		raise ValueError(
			f"O modelo {MLFLOW_MODEL_NAME} não possui a versão "
			f"'{ROLLBACK_ALIAS}' para rollback."
		)

	production_version = next(
		(version for version in versions if PRODUCTION_ALIAS in version.aliases),
		None,
	)
	if production_version is not None:
		client.set_registered_model_alias(
			MLFLOW_MODEL_NAME,
			ROLLBACK_ALIAS,
			production_version.version,
		)
	client.set_registered_model_alias(
		MLFLOW_MODEL_NAME,
		PRODUCTION_ALIAS,
		rollback_version.version,
	)
	return str(rollback_version.version)


def main():
	version = rollback_production_model()
	print(f"Modelo {MLFLOW_MODEL_NAME} revertido para a versão {version}.")


if __name__ == "__main__":
	main()
