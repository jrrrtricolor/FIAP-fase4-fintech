from types import SimpleNamespace
from unittest.mock import patch

import pytest

from credit_monitoring.rollback_model import rollback_production_model


@patch("credit_monitoring.rollback_model.mlflow.set_tracking_uri")
@patch("credit_monitoring.rollback_model.MlflowClient")
def test_rollback_swaps_production_aliases(client_class, set_tracking_uri, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///test.db")
    client = client_class.return_value
    client.search_model_versions.return_value = [
        SimpleNamespace(version="1", aliases=["production"]),
        SimpleNamespace(version="2", aliases=["production_rollback"]),
    ]

    version = rollback_production_model()

    assert version == "2"
    set_tracking_uri.assert_called_once_with("sqlite:///test.db")
    assert client.set_registered_model_alias.call_args_list == [
        (("home_credit_model", "production_rollback", "1"),),
        (("home_credit_model", "production", "2"),),
    ]


@patch("credit_monitoring.rollback_model.mlflow.set_tracking_uri")
@patch("credit_monitoring.rollback_model.MlflowClient")
def test_rollback_fails_without_rollback_alias(client_class, set_tracking_uri):
    client_class.return_value.search_model_versions.return_value = [
        SimpleNamespace(version="1", aliases=["production"])
    ]

    with pytest.raises(ValueError, match="não possui a versão"):
        rollback_production_model()

    client_class.return_value.set_registered_model_alias.assert_not_called()