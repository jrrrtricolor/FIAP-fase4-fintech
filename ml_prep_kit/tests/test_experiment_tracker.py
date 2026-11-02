"""Verifica a configuração de localização dos artefatos do MLflow."""

from mlflow.tracking import MlflowClient

from ml_prep_kit import ExperimentTracker


def test_from_directory_uses_artifact_location_environment_variable(
    tmp_path, monkeypatch
):
    artifact_location = (tmp_path / "shared-artifacts").as_uri()
    monkeypatch.setenv("MLFLOW_ARTIFACT_LOCATION", artifact_location)

    tracker = ExperimentTracker.from_directory("env-artifacts", tmp_path / "tracking")

    experiment = MlflowClient(tracking_uri=tracker.tracking_uri).get_experiment(
        tracker.experiment_id
    )
    assert experiment.artifact_location == artifact_location


def test_constructor_uses_artifact_location_environment_variable(tmp_path, monkeypatch):
    artifact_location = (tmp_path / "shared-artifacts").as_uri()
    monkeypatch.setenv("MLFLOW_ARTIFACT_LOCATION", artifact_location)

    tracker = ExperimentTracker(
        "env-artifacts-direct", tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}"
    )

    experiment = MlflowClient(tracking_uri=tracker.tracking_uri).get_experiment(
        tracker.experiment_id
    )
    assert experiment.artifact_location == artifact_location


def test_from_directory_keeps_default_artifact_location_without_environment_variable(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("MLFLOW_ARTIFACT_LOCATION", raising=False)

    tracker = ExperimentTracker.from_directory("local-artifacts", tmp_path / "tracking")

    experiment = MlflowClient(tracking_uri=tracker.tracking_uri).get_experiment(
        tracker.experiment_id
    )
    assert experiment.artifact_location == (
        (tmp_path / "tracking" / "artifacts").as_uri()
    )
