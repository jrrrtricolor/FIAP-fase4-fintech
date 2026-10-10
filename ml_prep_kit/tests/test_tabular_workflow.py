"""Verifica os pontos críticos do fluxo simples de Home Credit."""

import ast
import json
from pathlib import Path
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from ml_prep_kit import (
    DriftMonitor,
    ExperimentTracker,
    FeaturePreprocessor,
    HomeCreditData,
    ModelEvaluator,
    ModelFactory,
    ModelPredictor,
)


@pytest.fixture
def raw():
    """Fornece uma base sintética de mil registros aos testes via fixture.

    Exemplo:
        def test_sample_size(raw):
            assert len(raw) == 1000
    """
    rng = np.random.default_rng(42)
    age = rng.uniform(20, 75, 1000)
    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(1000),
            "TARGET": (age > 50).astype(int),
            "DAYS_BIRTH": -age * 365.25,
            "AMT_INCOME_TOTAL": rng.lognormal(10, 1, 1000),
            "AMT_CREDIT": rng.lognormal(12, 1, 1000),
            "AMT_ANNUITY": np.where(np.arange(1000) % 5 == 0, np.nan, 500.0),
            "CNT_CHILDREN": np.zeros(1000, dtype=int),
            "NAME_INCOME_TYPE": " Working ",
            "NAME_EDUCATION_TYPE": "Secondary",
            "NAME_HOUSING_TYPE": "House",
            "FLAG_OWN_CAR": "Y",
            "FLAG_OWN_REALTY": "N",
        }
    )


def test_preparation_and_split_preserve_data_and_isolation(raw):
    """Verifica conversões, reprodução da amostra e isolamento das partições.

    Exemplo:
        pytest ml_prep_kit/tests/test_tabular_workflow.py -k preparation
    """
    data = HomeCreditData(sample_size=200)
    original = raw.copy(deep=True)
    sampled = data.sample(raw)
    pd.testing.assert_frame_equal(sampled, data.sample(raw))
    prepared = data.prepare(sampled)
    pd.testing.assert_frame_equal(raw, original)
    assert prepared["AMT_ANNUITY"].isna().sum() == sampled["AMT_ANNUITY"].isna().sum()
    assert set(prepared["NAME_INCOME_TYPE"]) == {"Working"}
    np.testing.assert_allclose(prepared["AGE_YEARS"], -sampled["DAYS_BIRTH"] / 365.25)
    partitions = data.split(prepared)
    assert [len(p) for p in partitions.values()] == [120, 20, 20, 40]
    joined = pd.concat(partitions.values())
    assert not joined["SK_ID_CURR"].duplicated().any()
    assert set(joined["SK_ID_CURR"]) == set(prepared["SK_ID_CURR"])
    with pytest.raises(ValueError, match="sobreposição"):
        data.validate_partitions(dict(partitions, teste=partitions["treino"]))


@pytest.mark.parametrize(
    "column,value",
    [
        ("AGE_YEARS", -1.0),
        ("AMT_INCOME_TOTAL", np.inf),
        ("AMT_INCOME_TOTAL", np.nan),
        ("NAME_INCOME_TYPE", " "),
        ("FLAG_OWN_CAR", 2),
        ("TARGET", 3),
    ],
)
def test_contract_rejects_invalid_values(raw, column, value):
    """Verifica a rejeição de valores que violam o contrato de dados.

    Exemplo:
        pytest ml_prep_kit/tests/test_tabular_workflow.py -k invalid_values
    """
    data = HomeCreditData()
    prepared = data.prepare(raw)
    prepared.loc[0, column] = value
    with pytest.raises(pa.errors.SchemaErrors):
        data.contract.validate(prepared, lazy=True)


def test_contract_accepts_missing_target_and_rejects_duplicate_ids(raw):
    """Verifica que o alvo é opcional, mas identificadores devem ser únicos.

    Exemplo:
        pytest ml_prep_kit/tests/test_tabular_workflow.py -k missing_target
    """
    data = HomeCreditData()
    prepared = data.prepare(raw).drop(columns="TARGET")
    assert len(data.contract.validate(prepared)) == len(raw)
    prepared.loc[0, "SK_ID_CURR"] = prepared.loc[1, "SK_ID_CURR"]
    with pytest.raises(pa.errors.SchemaErrors):
        data.contract.validate(prepared, lazy=True)


def test_scenarios_preserve_source_and_invalid_batch_never_predicts(raw, tmp_path):
    """Verifica cenários reproduzíveis e rejeição de lotes antes da predição.

    Exemplo:
        pytest ml_prep_kit/tests/test_tabular_workflow.py -k scenarios
    """
    data = HomeCreditData()
    prepared = data.prepare(raw)
    original = prepared.copy(deep=True)
    scenarios, metadata = data.scenarios(prepared, prepared)
    pd.testing.assert_frame_equal(prepared, original)
    pd.testing.assert_frame_equal(
        scenarios["concept_drift"][data.features], prepared[data.features]
    )
    assert metadata["changed_labels"] > 0
    reordered, _ = data.scenarios(prepared.sample(frac=1, random_state=7), prepared)
    for name in scenarios:
        pd.testing.assert_frame_equal(scenarios[name], reordered[name])
    predictor = Mock()
    monitor = DriftMonitor(
        data.numeric_features,
        data.categorical_features + data.binary_features,
        tmp_path,
    )
    result = monitor.evaluate_batch(
        "invalid",
        scenarios["dados_invalidos"],
        prepared,
        predictor,
        data.contract,
        data.target,
    )
    assert result["status"] == "rejeitado"
    assert result["metrics"]["predicted_rows"] == 0
    predictor.positive_scores.assert_not_called()
    assert set(result["quality_failures"]["column"]) == {
        "SK_ID_CURR",
        "AGE_YEARS",
        "AMT_INCOME_TOTAL",
    }


def test_experiment_roundtrip_and_overlapping_partitions_are_rejected(raw, tmp_path):
    """Verifica a persistência das partições e a rejeição de IDs sobrepostos.

    Exemplo:
        pytest ml_prep_kit/tests/test_tabular_workflow.py -k roundtrip
    """
    data = HomeCreditData(sample_size=200)
    partitions = data.split(data.prepare(data.sample(raw)))
    data.save_experiment(tmp_path, partitions, {"model_uri": "models:/fixture"})
    restored = data.load_experiment(tmp_path)
    for name, frame in partitions.items():
        pd.testing.assert_frame_equal(frame, restored["partitions"][name])
    restored["partitions"]["teste"] = restored["partitions"]["treino"]
    joblib.dump(restored, tmp_path / "experiment.joblib")
    with pytest.raises(ValueError, match="sobreposição"):
        data.load_experiment(tmp_path)


def test_mlflow_reload_and_drift_detection(raw, tmp_path):
    """Verifica registro, recarga, inferência e detecção de drift no fluxo.

    Exemplo:
        pytest ml_prep_kit/tests/test_tabular_workflow.py -k mlflow_reload
    """
    from mlflow.tracking import MlflowClient

    data = HomeCreditData()
    partitions = data.split(data.prepare(raw))
    train, test = partitions["treino"], partitions["teste"]
    preprocessor = FeaturePreprocessor(
        data.numeric_features, data.categorical_features, data.binary_features
    ).create_pipeline()
    model = ModelFactory().create_pipeline(
        preprocessor, "classification", "logistic_regression", {"max_iter": 500}
    )
    tracker = ExperimentTracker(
        "integration",
        f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_location=(tmp_path / "artifacts").as_uri(),
    )
    with tracker.run("training") as training:
        model.fit(train[data.features], train[data.target])
        uri = tracker.log_model(model, features=train[data.features])
    predictor = ModelPredictor.from_mlflow(uri, tracking_uri=tracker.tracking_uri)
    predictor.assert_equivalent(ModelPredictor(model), test[data.features])
    future = test[data.features].copy()
    future["AMT_INCOME_TOTAL"] *= 1000
    future["NAME_INCOME_TYPE"] = "Unknown"
    future["AMT_ANNUITY"] = np.nan
    scaler = (
        model.named_steps["preprocessor"]
        .named_transformers_["num"]
        .named_steps["min_max_scaler"]
    )
    original_max = scaler.data_max_.copy()
    assert np.isfinite(predictor.predict_proba(future)).all()
    np.testing.assert_array_equal(scaler.data_max_, original_max)
    evaluation = ModelEvaluator().evaluate_binary_classification(
        test[data.target], predictor.positive_scores(test[data.features])
    )
    tracker.log_evaluation(training.info.run_id, {"teste": evaluation})
    assert "teste_roc_auc" in MlflowClient().get_run(training.info.run_id).data.metrics
    monitor = DriftMonitor(
        data.numeric_features,
        data.categorical_features + data.binary_features,
        tmp_path / "reports",
    )
    shifted = partitions["producao"].copy()
    shifted[["AMT_INCOME_TOTAL", "AMT_CREDIT"]] *= 100
    result = monitor.evaluate_batch(
        "shifted", shifted, train, predictor, data.contract, data.target
    )
    assert result["metrics"]["drifted_features"] >= 2
    assert (tmp_path / "reports/shifted.html").is_file()
    result = monitor.evaluate_batch(
        "unlabeled",
        shifted.drop(columns=data.target),
        train,
        predictor,
        data.contract,
        data.target,
    )
    assert result["metrics"]["performance_alert"] is None
    assert result["evaluation"] is None
    with pytest.raises(RuntimeError):
        with tracker.run("failure") as failed:
            raise RuntimeError("expected")
    assert MlflowClient().get_run(failed.info.run_id).info.status == "FAILED"


def test_notebooks_only_orchestrate_framework_components():
    """Verifica que os notebooks não definem funções, classes ou lambdas.

    Exemplo:
        pytest ml_prep_kit/tests/test_tabular_workflow.py -k notebooks
    """
    root = Path(__file__).resolve().parents[2]
    for path in (root / "notebooks").glob("0[123]_*.ipynb"):
        for cell in json.loads(path.read_text())["cells"]:
            if cell["cell_type"] == "code":
                tree = ast.parse("".join(cell["source"]))
                assert not any(
                    isinstance(
                        node,
                        (
                            ast.FunctionDef,
                            ast.AsyncFunctionDef,
                            ast.ClassDef,
                            ast.Lambda,
                        ),
                    )
                    for node in ast.walk(tree)
                ), path.name
