"""Contratos HTTP, carregamento e inferência sem dados reais do projeto."""

from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from credit_monitoring import api
from ml_prep_kit import FeaturePreprocessor, ModelFactory, ModelPredictor

PROFILE = {
    "AGE_YEARS": 35.5,
    "AMT_INCOME_TOTAL": 120000.0,
    "AMT_CREDIT": 250000.0,
    "AMT_ANNUITY": 12000.0,
    "CNT_CHILDREN": 0,
    "NAME_INCOME_TYPE": "Working",
    "NAME_EDUCATION_TYPE": "Secondary",
    "NAME_HOUSING_TYPE": "House",
    "FLAG_OWN_CAR": 1,
    "FLAG_OWN_REALTY": 0,
}


def prediction_payload():
    """Cria um perfil independente para cada caso de teste.

    Exemplo:
        payload = prediction_payload()
    """
    return {"prediction_data": [deepcopy(PROFILE)]}


@pytest.fixture
def predictor():
    """Fornece uma fachada real com estimador controlado.

    Exemplo:
        def test_inference(predictor): ...
    """
    model = Mock(classes_=np.array([1, 0]))
    model.predict_proba.return_value = np.array([[0.8, 0.2]])
    return ModelPredictor(model=model)


def test_load_model_defaults_to_production_alias(monkeypatch, predictor):
    """Resolve o alias antes de carregar para informar a versão efetiva.

    Exemplo:
        pytest tests/test_api.py -k defaults
    """
    monkeypatch.delenv("MODEL_URI", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    resolve = Mock(return_value=("models:/home_credit_model/4", "4"))
    loader = Mock(return_value=predictor)
    monkeypatch.setattr(ModelPredictor, "resolve_mlflow_uri", resolve)
    monkeypatch.setattr(ModelPredictor, "from_mlflow", loader)
    assert api.load_model() == (predictor, "4")
    resolve.assert_called_once_with(api.DEFAULT_MODEL_URI, api.DEFAULT_TRACKING_URI)
    loader.assert_called_once_with(
        model_uri="models:/home_credit_model/4", tracking_uri=api.DEFAULT_TRACKING_URI
    )


def test_load_model_ignores_unverified_version_override(monkeypatch, predictor):
    """A versão exibida vem do Registry, nunca de um texto informado no ambiente.

    Exemplo:
        pytest tests/test_api.py -k override
    """
    monkeypatch.setenv("MODEL_URI", "models:/candidate@staging")
    monkeypatch.setenv("MODEL_VERSION", "99")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///custom.db")
    resolve = Mock(return_value=("models:/candidate/17", "17"))
    monkeypatch.setattr(ModelPredictor, "resolve_mlflow_uri", resolve)
    monkeypatch.setattr(ModelPredictor, "from_mlflow", Mock(return_value=predictor))
    assert api.load_model() == (predictor, "17")
    resolve.assert_called_once_with("models:/candidate@staging", "sqlite:///custom.db")


def test_registry_resolution_pins_alias(monkeypatch):
    """Preserva a versão resolvida mesmo se o alias mudar antes do carregamento.

    Exemplo:
        pytest tests/test_api.py -k pins_alias
    """
    client = Mock()
    client.get_model_version_by_alias.return_value.version = "17"
    monkeypatch.setattr("mlflow.tracking.MlflowClient", Mock(return_value=client))
    result = ModelPredictor.resolve_mlflow_uri("models:/credit@production", "sqlite://")
    assert result == ("models:/credit/17", "17")
    client.get_model_version_by_alias.assert_called_once_with("credit", "production")


def test_registry_resolution_preserves_explicit_version(monkeypatch):
    """Valida também uma versão numérica explicitamente configurada.

    Exemplo:
        pytest tests/test_api.py -k explicit_version
    """
    client = Mock()
    client.get_model_version.return_value.version = "2"
    monkeypatch.setattr("mlflow.tracking.MlflowClient", Mock(return_value=client))
    assert ModelPredictor.resolve_mlflow_uri("models:/credit/2", "sqlite://") == (
        "models:/credit/2",
        "2",
    )
    client.get_model_version.assert_called_once_with("credit", "2")


@pytest.mark.parametrize("uri", ["runs:/run/model", "models:/m-123", "model/local"])
def test_unregistered_uris_preserve_uri_without_fabricating_version(monkeypatch, uri):
    """URIs de artefato continuam carregáveis sem inventar uma versão do Registry.

    Exemplo:
        pytest tests/test_api.py -k unregistered_uris
    """
    client = Mock()
    monkeypatch.setattr("mlflow.tracking.MlflowClient", client)
    assert ModelPredictor.resolve_mlflow_uri(uri, "sqlite://") == (uri, "unregistered")
    client.assert_not_called()


def test_startup_loads_model_once_and_marks_api_ready(predictor):
    """A criação da aplicação não carrega artefatos; o startup carrega uma vez.

    Exemplo:
        pytest tests/test_api.py -k startup_loads
    """
    loader = Mock(return_value=(predictor, "17"))
    app = api.create_app(model_loader=loader)
    loader.assert_not_called()
    with TestClient(app) as client:
        response = client.get("/ready")
        assert client.get("/health").status_code == 200
        assert "/predict" in client.get("/openapi.json").json()["paths"]
    assert response.status_code == 200
    assert response.json() == {"ready": True, "model_version": "17"}
    loader.assert_called_once_with()


def test_startup_failure_preserves_health_and_rejects_inference():
    """Uma falha de carga mantém diagnóstico acessível e impede inferências.

    Exemplo:
        pytest tests/test_api.py -k startup_failure
    """
    app = api.create_app(model_loader=Mock(side_effect=RuntimeError("indisponível")))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.get("/metrics").status_code == 200
        assert client.post("/predict", json=prediction_payload()).status_code == 503


def test_predict_returns_model_scores_with_single_inference(predictor):
    """Seleciona a classe positiva mesmo invertida e executa o modelo uma vez.

    Exemplo:
        pytest tests/test_api.py -k single_inference
    """
    app = api.create_app(predictor=predictor, model_version="4")
    with TestClient(app) as client:
        response = client.post("/predict", json=prediction_payload())
    assert response.status_code == 200
    assert response.json()["credit_scores"] == [0.8]
    assert response.json()["model_version"] == "4"
    assert response.json()["inference_time_ms"] >= 0
    assert response.headers["X-Request-ID"]
    predictor.model.predict_proba.assert_called_once()


@pytest.mark.parametrize(
    "column,value",
    [
        ("AGE_YEARS", 17),
        ("AMT_CREDIT", -1),
        ("FLAG_OWN_CAR", 2),
        ("NAME_INCOME_TYPE", "   "),
    ],
)
def test_contract_rejects_invalid_profiles_before_inference(predictor, column, value):
    """Bloqueia violações do contrato sem executar o Pipeline.

    Exemplo:
        pytest tests/test_api.py -k invalid_profiles
    """
    payload = prediction_payload()
    payload["prediction_data"][0][column] = value
    with TestClient(api.create_app(predictor=predictor)) as client:
        assert client.post("/predict", json=payload).status_code == 422
    predictor.model.predict_proba.assert_not_called()


@pytest.mark.parametrize(
    "payload", [{"prediction_data": []}, {"prediction_data": [{}]}]
)
def test_contract_rejects_empty_and_missing_features(predictor, payload):
    """Recusa lotes vazios e perfis sem as colunas exigidas.

    Exemplo:
        pytest tests/test_api.py -k empty_and_missing
    """
    with TestClient(api.create_app(predictor=predictor)) as client:
        assert client.post("/predict", json=payload).status_code == 422
    predictor.model.predict_proba.assert_not_called()


def test_predict_returns_service_unavailable_without_model():
    """Não informa prontidão nem aceita inferências sem modelo.

    Exemplo:
        pytest tests/test_api.py -k without_model
    """
    with TestClient(api.create_app(model_loader=None)) as client:
        assert client.get("/ready").status_code == 503
        response = client.post("/predict", json=prediction_payload())
    assert response.status_code == 503
    assert response.json()["detail"] == "Modelo indisponível."


@pytest.mark.parametrize(
    "path,column",
    [
        ("income-types", "NAME_INCOME_TYPE"),
        ("education-types", "NAME_EDUCATION_TYPE"),
        ("housing-types", "NAME_HOUSING_TYPE"),
    ],
)
def test_category_routes_use_predictor(path, column):
    """Mantém os três endpoints de categorias implementados na PR.

    Exemplo:
        pytest tests/test_api.py -k category_routes
    """
    predictor = Mock()
    predictor.get_categorical_column_values.return_value = ["categoria"]
    with TestClient(api.create_app(predictor=predictor)) as client:
        assert client.get(f"/{path}").json() == ["categoria"]
    predictor.get_categorical_column_values.assert_called_once_with(column)


def test_categories_reuse_fitted_encoder_without_custom_store():
    """Consulta o encoder de pipelines existentes sem exigir novo treinamento.

    Exemplo:
        pytest tests/test_api.py -k fitted_encoder
    """
    frame = pd.DataFrame({"income": ["Working", "Student", "Working", "Student"]})
    preprocessor = FeaturePreprocessor(categorical_columns=["income"])
    pipeline = ModelFactory().create_pipeline(
        preprocessor.create_pipeline(), "classification", "logistic_regression", {}
    )
    pipeline["preprocessor"].transformers[0][1].steps.pop(0)
    pipeline.fit(frame, [0, 1, 1, 0])
    predictor = ModelPredictor(model=pipeline)
    assert predictor.get_categorical_column_values("income") == ["Student", "Working"]
    with pytest.raises(ValueError, match="Coluna categórica ausente"):
        predictor.get_categorical_column_values("missing")
