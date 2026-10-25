from unittest.mock import Mock

from fastapi.testclient import TestClient

from credit_monitoring import api


def prediction_payload():
    return {
        "prediction_data": [
            {
                "AGE_YEARS": 35,
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
        ]
    }


def test_load_model_defaults_to_production_alias(monkeypatch):
    monkeypatch.delenv("MODEL_URI", raising=False)
    monkeypatch.delenv("MODEL_VERSION", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    predictor = Mock()
    load_from_mlflow = Mock(return_value=predictor)
    monkeypatch.setattr(api.ModelPredictor, "from_mlflow", load_from_mlflow)

    loaded_predictor, version = api.load_model()

    assert loaded_predictor is predictor
    assert version == ""
    load_from_mlflow.assert_called_once_with(
        model_uri=api.DEFAULT_MODEL_URI,
        tracking_uri=api.DEFAULT_TRACKING_URI,
    )
    assert api.DEFAULT_MODEL_URI == "models:/home_credit_model@production"


def test_load_model_uses_environment_overrides(monkeypatch):
    monkeypatch.setenv("MODEL_URI", "models:/candidate@staging")
    monkeypatch.setenv("MODEL_VERSION", "17")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///custom.db")
    predictor = Mock()
    load_from_mlflow = Mock(return_value=predictor)
    monkeypatch.setattr(api.ModelPredictor, "from_mlflow", load_from_mlflow)

    loaded_predictor, version = api.load_model()

    assert loaded_predictor is predictor
    assert version == "17"
    load_from_mlflow.assert_called_once_with(
        model_uri="models:/candidate@staging",
        tracking_uri="sqlite:///custom.db",
    )


def test_startup_loads_model_and_marks_api_ready():
    predictor = Mock()
    loader = Mock(return_value=(predictor, "version-17"))
    app = api.create_app(model_loader=loader)

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"ready": True, "model_version": "version-17"}
    loader.assert_called_once_with()


def test_predict_returns_model_scores():
    predictor = Mock()
    predictor.positive_scores.return_value = [0.8]
    predictor.predict_proba.return_value = [[0.2, 0.8]]
    app = api.create_app(
        predictor=predictor,
        model_version="version-4",
        model_loader=None,
    )

    with TestClient(app) as client:
        response = client.post("/predict", json=prediction_payload())

    assert response.status_code == 200
    result = response.json()
    assert result["credit_scores"] == [0.8]
    assert result["model_version"] == "version-4"
    assert result["inference_time_ms"] >= 0
    predictor.positive_scores.assert_called_once()
    predictor.predict_proba.assert_called_once()


def test_predict_returns_service_unavailable_without_model():
    app = api.create_app(model_loader=None)

    with TestClient(app) as client:
        response = client.post("/predict", json=prediction_payload())

    assert response.status_code == 503
    assert response.json()["detail"] == "Modelo indisponível."
