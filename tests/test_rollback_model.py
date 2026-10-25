"""Verifica promoção explícita, preservação de produção e rollback no SQLite."""

from unittest.mock import Mock

import pytest
from mlflow.tracking import MlflowClient

from credit_monitoring.rollback_model import rollback_production_model
from ml_prep_kit import ExperimentTracker


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


def test_rollback_swaps_production_aliases(registry, monkeypatch):
    """O comando usa o backend informado e preserva a versão substituída.

    Exemplo:
        pytest tests/test_rollback_model.py -k swaps
    """
    tracker, client = registry
    tracker.promote_model_version("home_credit_model", "2")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracker.tracking_uri)
    assert rollback_production_model() == "1"
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
