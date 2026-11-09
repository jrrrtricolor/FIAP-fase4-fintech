"""Verifica a configuração de localização dos artefatos do MLflow."""

from mlflow.tracking import MlflowClient

from ml_prep_kit import ExperimentTracker


def test_from_directory_uses_explicit_artifact_location(tmp_path):
    artifact_location = (tmp_path / "shared-artifacts").as_uri()
    tracker = ExperimentTracker.from_directory(
        "explicit-artifacts",
        tmp_path / "tracking",
        artifact_location=artifact_location,
    )

    experiment = MlflowClient(tracking_uri=tracker.tracking_uri).get_experiment(
        tracker.experiment_id
    )
    assert experiment.artifact_location == artifact_location


def test_constructor_uses_explicit_artifact_location(tmp_path):
    artifact_location = (tmp_path / "shared-artifacts").as_uri()

    tracker = ExperimentTracker(
        "explicit-artifacts-direct",
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_location=artifact_location,
    )

    experiment = MlflowClient(tracking_uri=tracker.tracking_uri).get_experiment(
        tracker.experiment_id
    )
    assert experiment.artifact_location == artifact_location


def test_from_directory_keeps_default_artifact_location(tmp_path):
    tracker = ExperimentTracker.from_directory("local-artifacts", tmp_path / "tracking")

    experiment = MlflowClient(tracking_uri=tracker.tracking_uri).get_experiment(
        tracker.experiment_id
    )
    assert experiment.artifact_location == (
        (tmp_path / "tracking" / "artifacts").as_uri()
    )


def test_from_directory_uses_explicit_tracking_uri(tmp_path):
    uri = f"sqlite:///{tmp_path / 'shared.db'}"

    tracker = ExperimentTracker.from_directory(
        "explicit-uri", tmp_path / "tracking", tracking_uri=uri
    )

    assert tracker.tracking_uri == uri
