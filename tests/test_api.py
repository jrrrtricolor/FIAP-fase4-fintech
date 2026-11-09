"""Contratos HTTP, carregamento e inferência sem dados reais do projeto."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST

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
    monkeypatch.setattr(ModelPredictor, "resolve_approved_mlflow_uri", resolve)
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
    monkeypatch.setattr(ModelPredictor, "resolve_approved_mlflow_uri", resolve)
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


@pytest.fixture(autouse=True)
def clean_model_cache():
    """Isola artefatos de cada teste sem desativar o cache utilizado pela API.

    Exemplo:
        pytest tests/test_api.py
    """
    api.cached_predictor.cache_clear()
    yield
    api.cached_predictor.cache_clear()


@pytest.fixture
def refresh_clock(monkeypatch):
    """Controla somente o relógio da recarga, preservando o tempo do servidor.

    Exemplo:
        refresh_clock.return_value += 31
    """
    clock = Mock(return_value=100.0)
    monkeypatch.setattr(api, "monotonic", clock)
    return clock


@pytest.fixture
def approved_registry(monkeypatch):
    """Disponibiliza uma versão ligada ao artefato aprovado em uma run concluída.

    Exemplo:
        client, version, run = approved_registry
    """
    tags = {"approval_status": "approved", "approval_model_uri": "models:/m-valid"}
    version = approved_version(tags)
    run = SimpleNamespace(
        info=SimpleNamespace(status="FINISHED"), data=SimpleNamespace(tags=dict(tags))
    )
    client = Mock()
    client.get_model_version_by_alias.return_value = version
    client.get_model_version.return_value = version
    client.get_logged_model.return_value = SimpleNamespace(source_run_id="run")
    client.get_run.return_value = run
    monkeypatch.setattr("mlflow.tracking.MlflowClient", Mock(return_value=client))
    return client, version, run


def approved_version(tags):
    """Cria metadados independentes da versão para testar divergências de origem.

    Exemplo:
        version = approved_version(approval_tags)
    """
    return SimpleNamespace(
        version="2",
        model_id="m-valid",
        status="READY",
        run_id="run",
        source="models:/m-valid",
        tags=dict(tags),
    )


@pytest.mark.parametrize("model_id", [None, "m-valid"])
@pytest.mark.parametrize("uri", ["models:/credit@production", "models:/credit/2"])
def test_approved_resolution_checks_exact_model_and_finished_run(
    approved_registry, model_id, uri
):
    """Confere alias, versão e origem, incluindo o Registry SQLite sem model_id.

    Exemplo:
        pytest tests/test_api.py -k approved_resolution
    """
    client, version, _ = approved_registry
    version.model_id = model_id
    result = ModelPredictor.resolve_approved_mlflow_uri(uri, "sqlite://")
    assert result == ("models:/credit/2", "2")
    client.get_logged_model.assert_called_once_with("m-valid")
    client.get_run.assert_called_once_with("run")


@pytest.mark.parametrize("status", ["RUNNING", "FAILED", "KILLED"])
def test_incomplete_or_failed_run_cannot_be_served(approved_registry, status):
    """Recusa versões cujas tags dizem aprovado mas a execução não terminou bem.

    Exemplo:
        pytest tests/test_api.py -k cannot_be_served
    """
    _, _, run = approved_registry
    run.info.status = status
    with pytest.raises(ValueError, match="modelo aprovado"):
        ModelPredictor.resolve_approved_mlflow_uri("models:/credit/2", "sqlite://")


@pytest.mark.parametrize("source", ["version", "run"])
@pytest.mark.parametrize(
    "field,value",
    [("approval_status", "rejected"), ("approval_model_uri", "models:/other")],
)
def test_approval_must_match_version_and_run(approved_registry, source, field, value):
    """Recusa aprovação ausente ou pertencente a outro modelo da mesma execução.

    Exemplo:
        pytest tests/test_api.py -k approval_must_match
    """
    _, version, run = approved_registry
    tags = version.tags if source == "version" else run.data.tags
    tags[field] = value
    with pytest.raises(ValueError, match="modelo aprovado"):
        ModelPredictor.resolve_approved_mlflow_uri("models:/credit/2", "sqlite://")


def test_registered_and_logged_models_must_share_source_run(approved_registry):
    """Detecta uma versão associada à run errada antes da desserialização.

    Exemplo:
        pytest tests/test_api.py -k share_source_run
    """
    _, version, _ = approved_registry
    version.run_id = "another-run"
    with pytest.raises(ValueError, match="modelo aprovado"):
        ModelPredictor.resolve_approved_mlflow_uri("models:/credit/2", "sqlite://")


@pytest.mark.parametrize("uri", ["runs:/run/model", "models:/m-valid", "model/local"])
def test_api_rejects_unregistered_uri_overrides(monkeypatch, uri):
    """Impede contornar a aprovação usando MODEL_URI com artefato direto.

    Exemplo:
        pytest tests/test_api.py -k uri_overrides
    """
    monkeypatch.setenv("MODEL_URI", uri)
    loader = Mock(side_effect=AssertionError("Não deve carregar artefato."))
    monkeypatch.setattr(ModelPredictor, "from_mlflow", loader)
    with pytest.raises(ValueError, match="versão aprovada"):
        api.load_model()
    loader.assert_not_called()


def test_same_version_rechecks_approval_without_reloading(monkeypatch, predictor):
    """Revalida as tags a cada consulta e mantém o artefato imutável em cache.

    Exemplo:
        pytest tests/test_api.py -k without_reloading
    """
    resolver = Mock(return_value=("models:/credit/2", "2"))
    loader = Mock(return_value=predictor)
    monkeypatch.setattr(ModelPredictor, "resolve_approved_mlflow_uri", resolver)
    monkeypatch.setattr(ModelPredictor, "from_mlflow", loader)
    assert api.load_model() == api.load_model() == (predictor, "2")
    assert resolver.call_count == 2
    loader.assert_called_once()
    resolver.side_effect = ValueError("aprovação revogada")
    with pytest.raises(ValueError, match="revogada"):
        api.load_model()


def test_ready_refreshes_approved_version_and_supports_rollback(
    predictor, refresh_clock
):
    """Carrega uma nova versão e acompanha rollback após o intervalo de consulta.

    Exemplo:
        pytest tests/test_api.py -k supports_rollback
    """
    loader = Mock(side_effect=[(predictor, "1"), (predictor, "2"), (predictor, "1")])
    application = api.create_app(model_loader=loader)
    with TestClient(application) as client:
        assert client.get("/ready").json()["model_version"] == "1"
        assert loader.call_count == 1
        refresh_clock.return_value += 31
        assert client.get("/ready").json()["model_version"] == "2"
        refresh_clock.return_value += 31
        assert client.get("/ready").json()["model_version"] == "1"
    assert not application.state.reload_error


def test_failed_reload_keeps_previous_model_available(
    approved_api, refresh_clock, monkeypatch
):
    """Mantém o modelo aprovado em memória mesmo com arquivo inacessível e sem cache.

    Exemplo:
        pytest tests/test_api.py -k failed_reload
    """
    with TestClient(approved_api) as client:
        api.cached_predictor.cache_clear()
        loader = Mock(side_effect=ValueError("arquivo inacessível"))
        monkeypatch.setattr(ModelPredictor, "from_mlflow", loader)
        refresh_clock.return_value += 31
        response = client.post("/predict", json=prediction_payload())
        assert response.status_code == 200
        assert response.json()["model_version"] == "2"
        assert client.get("/ready").json() == {"ready": True, "model_version": "2"}
    assert approved_api.state.reload_error


def test_concurrent_refresh_keeps_snapshot_version_consistent(predictor):
    """Uma troca durante a inferência não muda a versão informada nessa resposta.

    Exemplo:
        pytest tests/test_api.py -k snapshot_version
    """
    application = api.create_app(predictor=predictor, model_version="1")
    payload = api.PredictionRequest(prediction_data=[PROFILE])
    snapshot = application.state.active_model
    application.state.active_model = (predictor, "2")
    response = api.predict(payload, snapshot)
    assert response.model_version == "1"
    assert response.credit_scores == [0.8]


def test_requests_keep_serving_during_another_refresh(predictor, refresh_clock):
    """Uma recarga em andamento não bloqueia a inferência na versão já carregada.

    Exemplo:
        pytest tests/test_api.py -k keep_serving
    """
    loader = Mock(return_value=(predictor, "1"))
    application = api.create_app(model_loader=loader)
    with TestClient(application) as client:
        refresh_clock.return_value += 31
        with application.state.model_lock:
            response = client.post("/predict", json=prediction_payload())
        assert response.status_code == 200
        assert response.json()["model_version"] == "1"
        assert loader.call_count == 1


def test_unavailable_api_recovers_after_approval(predictor, refresh_clock):
    """Repete a consulta após o intervalo e recupera a prontidão sem reiniciar.

    Exemplo:
        pytest tests/test_api.py -k recovers_after_approval
    """
    loader = Mock(side_effect=[ValueError("sem aprovação"), (predictor, "2")])
    application = api.create_app(model_loader=loader)
    with TestClient(application) as client:
        assert client.get("/ready").status_code == 503
        refresh_clock.return_value += 31
        assert client.get("/ready").json() == {"ready": True, "model_version": "2"}
    assert not application.state.reload_error


@pytest.mark.parametrize(
    "source",
    [
        "models:/m-other",
        "models:/credit/2",
        "models:/m-valid@prod",
        "file:/model",
        "models:/m-",
    ],
)
@pytest.mark.parametrize("model_id", [None, "m-valid"])
def test_approval_rejects_source_that_does_not_identify_approved_artifact(
    approved_registry, source, model_id
):
    """Não usa tags para adivinhar a identidade de uma origem inválida ou divergente.

    Exemplo:
        pytest tests/test_api.py -k rejects_source
    """
    _, version, _ = approved_registry
    version.source = source
    version.model_id = model_id
    with pytest.raises(ValueError):
        ModelPredictor.resolve_approved_mlflow_uri("models:/credit/2", "sqlite://")


@pytest.fixture
def approved_api(monkeypatch, approved_registry, predictor):
    """Usa o carregador real com Registry controlado e artefato conhecido.

    Exemplo:
        with TestClient(approved_api) as client: ...
    """
    monkeypatch.setenv("MODEL_URI", "models:/credit@production")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite://")
    monkeypatch.setattr(ModelPredictor, "from_mlflow", Mock(return_value=predictor))
    return api.create_app()


def revoke_approval(registry, reason):
    """Altera uma evidência de aprovação depois de o modelo entrar em produção.

    Exemplo:
        revoke_approval(approved_registry, "version")
    """
    _, version, run = registry
    if reason == "failed_run":
        run.info.status = "FAILED"
    else:
        tags = version.tags if reason == "version" else run.data.tags
        tags["approval_status"] = "rejected"


@pytest.mark.parametrize("reason", ["version", "run", "failed_run"])
def test_active_revocation_blocks_inference(
    approved_api, approved_registry, refresh_clock, reason
):
    """Revogação comprovada interrompe prontidão e inferência na versão em memória.

    Exemplo:
        pytest tests/test_api.py -k active_revocation
    """
    with TestClient(approved_api) as client:
        assert client.get("/ready").status_code == 200
        revoke_approval(approved_registry, reason)
        refresh_clock.return_value += 31
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json=prediction_payload()).status_code == 503
        assert not api.MODEL_INFO.collect()[0].samples
    assert approved_api.state.reload_error


def test_rejected_candidate_preserves_approved_active_version(
    approved_api, approved_registry, refresh_clock
):
    """Revalida a versão numérica anterior quando o alias aponta a um candidato ruim.

    Exemplo:
        pytest tests/test_api.py -k rejected_candidate
    """
    registry, previous, _ = approved_registry
    candidate = deepcopy(previous)
    candidate.version, candidate.tags["approval_status"] = "3", "rejected"
    with TestClient(approved_api) as client:
        registry.get_model_version_by_alias.return_value = candidate
        registry.get_model_version.side_effect = [candidate, previous, previous]
        refresh_clock.return_value += 31
        assert client.get("/ready").json() == {"ready": True, "model_version": "2"}
        assert client.post("/predict", json=prediction_payload()).status_code == 200
    assert approved_api.state.reload_error


@pytest.mark.parametrize(
    "error,status",
    [
        (RuntimeError("offline"), 200),
        (MlflowException("offline", INTERNAL_ERROR), 200),
        (MlflowException("removida", RESOURCE_DOES_NOT_EXIST), 503),
    ],
)
def test_registry_failure_preserves_only_existing_version(
    approved_api, approved_registry, refresh_clock, error, status
):
    """Mantém inferência na indisponibilidade, mas a bloqueia se a versão foi apagada.

    Exemplo:
        pytest tests/test_api.py -k registry_failure
    """
    registry, _, _ = approved_registry
    with TestClient(approved_api) as client:
        registry.get_model_version_by_alias.side_effect = error
        registry.get_model_version.side_effect = error
        refresh_clock.return_value += 31
        assert client.get("/ready").status_code == status
        response = client.post("/predict", json=prediction_payload())
        assert response.status_code == status
        if status == 200:
            assert response.json()["model_version"] == "2"
    assert approved_api.state.reload_error


def test_revoked_model_recovers_only_after_new_approval(
    approved_api, approved_registry, refresh_clock
):
    """A recuperação revalida a aprovação mesmo que o artefato continue no cache.

    Exemplo:
        pytest tests/test_api.py -k recovers_only
    """
    _, version, _ = approved_registry
    with TestClient(approved_api) as client:
        version.tags["approval_status"] = "rejected"
        refresh_clock.return_value += 31
        assert client.get("/ready").status_code == 503
        version.tags["approval_status"] = "approved"
        refresh_clock.return_value += 31
        assert client.get("/ready").json() == {"ready": True, "model_version": "2"}
    assert not approved_api.state.reload_error


def assert_runtime_metrics(client, ready, error):
    """Confere os sinais operacionais publicados pelo endpoint Prometheus.

    Exemplo:
        assert_runtime_metrics(client, ready=1, error=0)
    """
    metrics = client.get("/metrics").text
    assert f"credit_monitoring_api_ready {ready}.0" in metrics
    assert f"credit_monitoring_model_reload_error {error}.0" in metrics


def test_runtime_metrics_follow_failure_recovery_and_preservation(
    predictor, refresh_clock
):
    """Distingue API sem modelo de falha de recarga com versão anterior disponível.

    Exemplo:
        pytest tests/test_api.py -k runtime_metrics_follow
    """
    loader = Mock(side_effect=[RuntimeError(), (predictor, "1"), RuntimeError()])
    with TestClient(api.create_app(model_loader=loader)) as client:
        assert_runtime_metrics(client, ready=0, error=1)
        refresh_clock.return_value += 31
        assert client.get("/ready").status_code == 200
        assert_runtime_metrics(client, ready=1, error=0)
        refresh_clock.return_value += 31
        assert client.get("/ready").status_code == 200
        assert_runtime_metrics(client, ready=1, error=1)
