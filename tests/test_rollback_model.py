"""Verifica promoção explícita, preservação de produção e rollback no SQLite."""

from unittest.mock import DEFAULT, Mock

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier

from credit_monitoring import CreditMonitoring
from credit_monitoring.rollback_model import rollback_production_model
from ml_prep_kit import ExperimentTracker, ModelPredictor


@pytest.fixture
def registry(tmp_path):
    """Disponibiliza três versões em um Registry isolado do projeto.

    Exemplo:
        def test_registry(registry): ...
    """
    tracker = ExperimentTracker.from_directory("registry-test", tmp_path)
    client = MlflowClient(tracking_uri=tracker.tracking_uri)
    client.create_registered_model("home_credit_model")
    for version in range(3):
        client.create_model_version("home_credit_model", str(tmp_path / str(version)))
    client.set_registered_model_alias("home_credit_model", "production", "1")
    return tracker, client


def aliases(client):
    """Obtém os aliases ativos para comparar mudanças no Registry.

    Exemplo:
        current = aliases(client)
    """
    registered = client.get_registered_model("home_credit_model")
    return {name: str(version) for name, version in registered.aliases.items()}


def test_explicit_promotion_uses_selected_version_and_preserves_rollback(registry):
    """Promove a versão escolhida mesmo quando já existe uma versão mais recente.

    Exemplo:
        pytest tests/test_rollback_model.py -k explicit
    """
    tracker, client = registry
    assert tracker.promote_model_version("home_credit_model", "2") == "2"
    assert aliases(client) == {"production": "2", "production_rollback": "1"}
    tracker.promote_model_version("home_credit_model", "2")
    assert aliases(client)["production_rollback"] == "1"


def test_rollback_swaps_production_aliases(registry):
    """O Registry preserva a versão substituída ao restaurar o alias anterior.

    Exemplo:
        pytest tests/test_rollback_model.py -k swaps
    """
    tracker, client = registry
    tracker.promote_model_version("home_credit_model", "2")
    assert tracker.rollback_model_version("home_credit_model") == "1"
    assert aliases(client) == {"production": "1", "production_rollback": "2"}


def test_rollback_fails_without_rollback_alias(registry):
    """A ausência de rollback não altera a versão de produção.

    Exemplo:
        pytest tests/test_rollback_model.py -k without
    """
    tracker, client = registry
    with pytest.raises(ValueError, match="não possui a versão"):
        tracker.rollback_model_version("home_credit_model")
    assert aliases(client) == {"production": "1"}


def test_baseline_setup_preserves_existing_production(registry):
    """Repetir o setup não substitui uma versão já disponibilizada.

    Exemplo:
        pytest tests/test_rollback_model.py -k setup
    """
    tracker, client = registry
    assert tracker.initialize_production_version("home_credit_model", "3") == "1"
    assert aliases(client) == {"production": "1"}
    client.delete_registered_model_alias("home_credit_model", "production")
    assert tracker.initialize_production_version("home_credit_model", "2") == "2"
    assert aliases(client) == {"production": "2"}


def test_invalid_version_does_not_change_aliases(registry):
    """Recusa uma versão inexistente antes de trocar os aliases.

    Exemplo:
        pytest tests/test_rollback_model.py -k invalid
    """
    from mlflow.exceptions import MlflowException

    tracker, client = registry
    with pytest.raises(MlflowException):
        tracker.promote_model_version("home_credit_model", "99")
    assert aliases(client) == {"production": "1"}


def test_registration_returns_the_created_version(monkeypatch):
    """O consumidor recebe a versão criada, sem buscar a última por aproximação.

    Exemplo:
        pytest tests/test_rollback_model.py -k registration
    """
    tracker = ExperimentTracker.__new__(ExperimentTracker)
    tracker.tracking_uri = None
    register = Mock(return_value=Mock(version="4"))
    monkeypatch.setattr("mlflow.register_model", register)
    version = tracker.register_model("runs:/run/model", "home_credit_model")
    assert version.version == "4"
    register.assert_called_once_with(
        model_uri="runs:/run/model", name="home_credit_model"
    )


def fail_publication(tracker, client, monkeypatch, compensation_error=False):
    """Injeta falha na troca de produção após persistir seu rollback.

    Exemplo:
        fail_publication(tracker, client, monkeypatch)
    """
    restoration = RuntimeError("compensation failed") if compensation_error else DEFAULT
    writes = Mock(
        wraps=client.set_registered_model_alias,
        side_effect=[DEFAULT, RuntimeError("publication failed"), restoration],
    )
    monkeypatch.setattr(client, "set_registered_model_alias", writes)
    monkeypatch.setattr(
        tracker, "_registered_versions", Mock(return_value=(client, []))
    )


@pytest.mark.parametrize("rollback,target", [(None, "3"), ("1", "3"), ("1", "1")])
def test_failed_publication_preserves_aliases(registry, monkeypatch, rollback, target):
    """Uma segunda escrita recusada preserva produção e seu rollback anterior.

    Exemplo:
        pytest tests/test_rollback_model.py -k preserves_aliases
    """
    tracker, client = registry
    client.set_registered_model_alias("home_credit_model", "production", "2")
    if rollback is not None:
        client.set_registered_model_alias(
            "home_credit_model", "production_rollback", rollback
        )
    before = aliases(client)
    fail_publication(tracker, client, monkeypatch)
    with pytest.raises(RuntimeError, match="publication failed"):
        tracker.promote_model_version("home_credit_model", target)
    assert aliases(client) == before


def test_failed_compensation_logs_error_and_raises_original(
    registry, monkeypatch, caplog
):
    """Uma compensação indisponível fica visível sem esconder o erro original.

    Exemplo:
        pytest tests/test_rollback_model.py -k compensation
    """
    tracker, client = registry
    tracker.promote_model_version("home_credit_model", "2")
    fail_publication(tracker, client, monkeypatch, compensation_error=True)
    with pytest.raises(RuntimeError, match="publication failed"):
        tracker.rollback_model_version("home_credit_model")
    assert "Falha ao restaurar alias production_rollback" in caplog.text
    assert "compensation failed" in caplog.text
    assert aliases(client) == {"production": "2", "production_rollback": "2"}


def approved_version(tracker):
    """Produz uma versão real com origem e decisão vinculadas no SQLite.

    Exemplo:
        version = approved_version(tracker)
    """
    features, labels = pd.DataFrame({"x": [0.0, 1.0]}), pd.Series([0, 1])
    record = tracker.train_model("model", DummyClassifier(), features, labels)
    tracker.approve_baseline_model(record["model_uri"])
    return tracker.register_approved_model(record["model_uri"], "home_credit_model")


@pytest.fixture
def publication(tmp_path):
    """Disponibiliza uma referência publicada e um candidato aprovado independente.

    Exemplo:
        def test_publish(publication): ...
    """
    tracker = ExperimentTracker.from_directory("publication", tmp_path)
    first, candidate = approved_version(tracker), approved_version(tracker)
    tracker.promote_approved_model_version("home_credit_model", first.version, None)
    return tracker, first, candidate


def test_approved_publication_and_command_rollback(publication, monkeypatch):
    """Publica a versão aprovada e comprova o retorno pelo comando de rollback.

    Exemplo:
        pytest tests/test_rollback_model.py -k command_rollback
    """
    tracker, first, candidate = publication
    name = "home_credit_model"
    tracker.promote_approved_model_version(name, candidate.version, first.version)
    assert tracker.model_alias_version(name, "production_rollback") == str(
        first.version
    )
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracker.tracking_uri)
    assert rollback_production_model() == str(first.version)
    uri, version = ModelPredictor.resolve_approved_mlflow_uri(
        f"models:/{name}@production", tracker.tracking_uri
    )
    assert version == str(first.version)
    assert ModelPredictor.from_mlflow(uri, tracking_uri=tracker.tracking_uri)


def test_rollback_forwards_artifact_location(monkeypatch, tmp_path):
    """Exemplo: pytest tests/test_rollback_model.py -k forwards_artifact."""
    location = (tmp_path / "shared-artifacts").as_uri()
    monkeypatch.setenv("MLFLOW_ARTIFACT_LOCATION", location)
    tracker = Mock()
    factory = Mock(return_value=tracker)
    monkeypatch.setattr("credit_monitoring.rollback_model.ExperimentTracker", factory)

    rollback_production_model()

    assert factory.call_args.kwargs["artifact_location"] == location


def test_changed_reference_blocks_stale_candidate(publication):
    """Uma publicação concorrente exige nova comparação antes de trocar produção.

    Exemplo:
        pytest tests/test_rollback_model.py -k stale_candidate
    """
    tracker, first, candidate = publication
    with pytest.raises(ValueError, match="Produção mudou"):
        tracker.promote_approved_model_version(
            "home_credit_model", candidate.version, "99"
        )
    assert tracker.model_alias_version("home_credit_model") == str(first.version)
    assert (
        tracker.model_alias_version("home_credit_model", "production_rollback") is None
    )


def test_failed_source_run_cannot_replace_production(publication):
    """Mesmo um artefato antes aprovado não pode ser servido após falha da run.

    Exemplo:
        pytest tests/test_rollback_model.py -k failed_source
    """
    tracker, first, candidate = publication
    client = MlflowClient(tracking_uri=tracker.tracking_uri)
    model = client.get_logged_model(candidate.source.removeprefix("models:/"))
    client.set_terminated(model.source_run_id, status="FAILED")
    with pytest.raises(ValueError):
        tracker.promote_approved_model_version(
            "home_credit_model", candidate.version, first.version
        )
    assert tracker.model_alias_version("home_credit_model") == str(first.version)


def test_unapproved_rollback_preserves_current_version(publication):
    """Não move produção quando a versão de retorno deixou de estar aprovada.

    Exemplo:
        pytest tests/test_rollback_model.py -k unapproved_rollback
    """
    tracker, first, candidate = publication
    tracker.promote_approved_model_version(
        "home_credit_model", candidate.version, first.version
    )
    client = MlflowClient(tracking_uri=tracker.tracking_uri)
    client.set_model_version_tag(
        "home_credit_model", first.version, "approval_status", "rejected"
    )
    with pytest.raises(ValueError):
        tracker.rollback_approved_model_version("home_credit_model")
    assert tracker.model_alias_version("home_credit_model") == str(candidate.version)


def test_monitoring_compares_current_production_instead_of_old_baseline(publication):
    """Congela a referência ativa usada tanto pelo drift quanto pela aprovação.

    Exemplo:
        pytest tests/test_rollback_model.py -k current_production
    """
    tracker, first, _ = publication
    workflow = CreditMonitoring.__new__(CreditMonitoring)
    workflow.tracker, workflow.tags = tracker, {}
    workflow.baseline = {"model_uri": "models:/m-obsolete", "threshold": 0.3}
    workflow.load_reference_model()
    assert workflow.reference_version == str(first.version)
    assert workflow.reference_threshold == 0.5
    assert (
        workflow.tags["reference_model_uri"]
        == f"models:/home_credit_model/{first.version}"
    )
    assert workflow.predictor.model is not None
