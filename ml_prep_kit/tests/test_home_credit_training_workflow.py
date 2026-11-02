"""Proveniência e isolamento dos Pipelines no treinamento compartilhado."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from ml_prep_kit import HomeCreditTrainingWorkflow

TRAIN_FEATURES = pd.DataFrame(
    {
        "AGE_YEARS": [25.0, 30.0, 35.0, 40.0],
        "AMT_INCOME_TOTAL": [100.0, 150.0, 200.0, 250.0],
        "AMT_CREDIT": [200.0, 250.0, 300.0, 350.0],
        "AMT_ANNUITY": [10.0, 15.0, 20.0, 25.0],
        "CNT_CHILDREN": [0, 1, 0, 1],
        "NAME_INCOME_TYPE": ["Working", "Student", "Working", "Student"],
        "NAME_EDUCATION_TYPE": ["Secondary"] * 4,
        "NAME_HOUSING_TYPE": ["House"] * 4,
        "FLAG_OWN_CAR": [0, 1, 1, 0],
        "FLAG_OWN_REALTY": [1, 1, 0, 0],
    }
)


def fit_record(name, model, features, target, **kwargs):
    """Ajusta um Pipeline real sem dependências externas de rastreamento.

    Exemplo:
        record = fit_record("dummy", model, features, [0, 1, 1, 0])
    """
    model.fit(features, target)
    return {"run_id": f"{name}-run", "model_uri": f"runs:/{name}/model"}


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    """Isola o armazenamento e mantém preparação e modelos reais.

    Exemplo:
        def test_train(workflow): ...
    """
    tracker = Mock(tracking_uri="sqlite:///tracking.db")
    tracker.train_model.side_effect = fit_record
    monkeypatch.setattr(
        "ml_prep_kit.home_credit_training_workflow.ExperimentTracker.from_directory",
        Mock(return_value=tracker),
    )
    workflow = HomeCreditTrainingWorkflow(tmp_path, tmp_path, "test", sample_size=10)
    workflow.source_hash = "source-digest"
    workflow.datasets = {"treino": (TRAIN_FEATURES.copy(), [0, 1, 1, 0])}
    workflow.partitions = {"treino": TRAIN_FEATURES.copy()}
    workflow.preparation.save_experiment = Mock()
    return workflow


def test_workflow_forwards_explicit_artifact_location(tmp_path, monkeypatch):
    location = (tmp_path / "shared-artifacts").as_uri()
    factory = Mock(return_value=Mock())
    monkeypatch.setattr(
        "ml_prep_kit.home_credit_training_workflow.ExperimentTracker.from_directory",
        factory,
    )

    HomeCreditTrainingWorkflow(
        tmp_path,
        tmp_path,
        "test",
        sample_size=10,
        artifact_location=location,
    )

    assert factory.call_args.kwargs["artifact_location"] == location


def test_training_records_share_source_and_split_metadata(workflow):
    """Associa modelos e runs ao mesmo recorte e à origem dos dados.

    Exemplo:
        pytest ml_prep_kit/tests/test_home_credit_training_workflow.py -k metadata
    """
    models, records = workflow.train_models()
    assert set(models) == {"dummy", "logistic_regression"}
    assert records["logistic_regression"]["run_id"] == "logistic_regression-run"
    for call in workflow.tracker.train_model.call_args_list:
        assert call.args[2] is workflow.datasets["treino"][0]
        assert call.args[3] is workflow.datasets["treino"][1]
        assert call.kwargs["tags"] == {"source_sha256": "source-digest"}
        assert call.kwargs["parameters"]["sample_size"] == 10
        assert call.kwargs["parameters"]["train_rows"] == 4
        assert call.kwargs["parameters"]["seed"] == 42


def test_persisted_experiment_preserves_provenance_and_model_link(workflow):
    """Liga a referência persistida à origem, ao limiar e à run do modelo.

    Exemplo:
        pytest ml_prep_kit/tests/test_home_credit_training_workflow.py -k persisted
    """
    workflow.save_experiment({"threshold": 0.4, "training_run_id": "logistic-run"})
    call = workflow.preparation.save_experiment.call_args
    assert call.args[:2] == (workflow.processed_directory, workflow.partitions)
    assert call.args[2] == {
        "tracking_uri": "sqlite:///tracking.db",
        "source_sha256": "source-digest",
        "random_state": 42,
        "sample_size": 10,
        "threshold": 0.4,
        "training_run_id": "logistic-run",
    }


def test_models_do_not_share_fitted_preprocessors(workflow):
    """Cada estimador mantém transformações próprias após o ajuste.

    Exemplo:
        pytest ml_prep_kit/tests/test_home_credit_training_workflow.py -k share_fitted
    """
    models, _ = workflow.train_models()
    dummy = models["dummy"]["preprocessor"]
    logistic = models["logistic_regression"]["preprocessor"]
    assert dummy is not logistic
    assert dummy.named_transformers_["num"] is not logistic.named_transformers_["num"]
    assert dummy.named_transformers_["cat"] is not logistic.named_transformers_["cat"]


def test_scaling_ignores_validation_and_test_data(workflow):
    """Valores extremos fora do treino não alteram a normalização aprendida.

    Exemplo:
        pytest ml_prep_kit/tests/test_home_credit_training_workflow.py -k scaling
    """
    extreme = TRAIN_FEATURES.assign(AMT_CREDIT=999999.0)
    workflow.datasets["validacao"] = (extreme, [0, 1, 1, 0])
    workflow.datasets["teste"] = (extreme, [0, 1, 1, 0])
    models, _ = workflow.train_models()
    expected = TRAIN_FEATURES[workflow.preparation.numeric_features].max().to_numpy()
    for model in models.values():
        scaler = model["preprocessor"].named_transformers_["num"]["min_max_scaler"]
        np.testing.assert_allclose(scaler.data_max_, expected)
