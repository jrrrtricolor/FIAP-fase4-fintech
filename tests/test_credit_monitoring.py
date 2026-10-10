"""Verifica contratos, isolamento e rastreamento do monitoramento de crédito."""

import ast
import json
import sqlite3
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest
from mlflow.tracking import MlflowClient

from credit_monitoring import EXPERIMENT_NAME, CreditMonitoring, run_workflow
from ml_prep_kit import (
    ExperimentTracker,
    FeaturePreprocessor,
    HomeCreditData,
    ModelEvaluator,
    ModelFactory,
    ModelPredictor,
)


def sample_data():
    """Gera dados rotulados e anuidade ausente para testar imputação e drift.

    Exemplo:
        frame = sample_data()
    """
    rng = np.random.default_rng(42)
    age = rng.uniform(20, 75, 1000)
    frame = pd.DataFrame({"SK_ID_CURR": np.arange(1000), "AGE_YEARS": age})
    return frame.assign(
        TARGET=(age > 45).astype(int),
        AMT_INCOME_TOTAL=rng.lognormal(10, 0.5, 1000),
        AMT_CREDIT=rng.lognormal(12, 0.5, 1000),
        AMT_ANNUITY=np.where(np.arange(1000) % 7 == 0, np.nan, 500.0),
        CNT_CHILDREN=np.zeros(1000, dtype=int),
        **sample_categories(),
    )


def sample_categories():
    """Define atributos categóricos constantes para isolar as mudanças numéricas.

    Exemplo:
        categories = sample_categories()
    """
    return {
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Secondary",
        "NAME_HOUSING_TYPE": "House",
        "FLAG_OWN_CAR": np.ones(1000, dtype=int),
        "FLAG_OWN_REALTY": np.zeros(1000, dtype=int),
    }


def baseline_model(data):
    """Monta o pipeline usado como modelo original nos testes.

    Exemplo:
        model = baseline_model(HomeCreditData())
    """
    preprocessor = FeaturePreprocessor(
        data.numeric_features,
        data.categorical_features,
        data.binary_features,
    ).create_pipeline()
    return ModelFactory().create_pipeline(
        preprocessor,
        "classification",
        "logistic_regression",
        {"max_iter": 500, "class_weight": "balanced"},
    )


def baseline_tracker(directory):
    """Cria o mesmo backend SQLite utilizado pelos notebooks e pelo workflow.

    Exemplo:
        tracker = baseline_tracker(directory)
    """
    return ExperimentTracker.from_directory(
        "v1", directory / "monitoring", "baseline_artifacts"
    )


def save_baseline(data, directory, partitions, tracker):
    """Persiste partições e modelo original para a atualização da v2.

    Exemplo:
        save_baseline(data, directory, partitions, tracker)
    """
    train = partitions["treino"]
    record = tracker.train_model(
        "v1", baseline_model(data), train[data.features], train[data.target]
    )
    metadata = {
        "threshold": 0.5,
        "tracking_uri": tracker.tracking_uri,
        "training_run_id": record["run_id"],
        "model_uri": record["model_uri"],
        "source_sha256": "synthetic",
        "random_state": 42,
        "sample_size": 1000,
    }
    data.save_experiment(directory, partitions, metadata)


@pytest.fixture
def baseline(tmp_path):
    """Disponibiliza uma referência isolada, com modelo e banco persistidos.

    Exemplo:
        def test_versions(baseline):
            data, directory, original = baseline
    """
    data, original = HomeCreditData(sample_size=1000), sample_data()
    directory = tmp_path / "processed"
    tracker = baseline_tracker(directory)
    save_baseline(data, directory, data.split(original), tracker)
    return data, directory, original


def write_shifted_data(data, directory, original):
    """Gera drift e um extremo em produção para revelar vazamento no scaler.

    Exemplo:
        write_shifted_data(data, directory, original)
    """
    reference = data.load_experiment(directory)
    production_id = reference["partitions"]["producao"].iloc[0]["SK_ID_CURR"]
    shifted = data.create_drift_version(original, income_factor=0.3, credit_factor=2)
    shifted.loc[shifted["SK_ID_CURR"].eq(production_id), "AMT_CREDIT"] = 1e15
    shifted["NAME_INCOME_TYPE"] = " Working "
    shifted.sample(frac=1, random_state=7).to_csv(
        directory / "credit_scoring_v2.csv", index=False
    )


@pytest.fixture
def trained_version(baseline):
    """Executa a atualização e preserva a assinatura da referência para comparação.

    Exemplo:
        def test_candidate(trained_version):
            data, directory, result, original_hash = trained_version
    """
    data, directory, original = baseline
    original_hash = data.source_hash(directory / "experiment.joblib")
    write_shifted_data(data, directory, original)
    return data, directory, run_workflow(directory), original_hash


def test_training_preserves_baseline_and_partition_membership(trained_version):
    """Confere que a v2 não modifica a referência nem move IDs entre partições.

    Exemplo:
        pytest tests/test_credit_monitoring.py -k partition_membership
    """
    data, directory, result, original_hash = trained_version
    reference = data.load_experiment(directory)
    restored = data.load_experiment(result["output_dir"])
    assert result["status"] == "treinado"
    assert data.source_hash(directory / "experiment.joblib") == original_hash
    for name, part in reference["partitions"].items():
        assert (
            restored["partitions"][name]["SK_ID_CURR"].tolist()
            == part["SK_ID_CURR"].tolist()
        )
    assert set(restored["partitions"]["treino"]["NAME_INCOME_TYPE"]) == {"Working"}


def test_normalization_uses_training_data_only(trained_version):
    """Garante que o extremo de produção não participa do ajuste do Min-Max.

    Exemplo:
        pytest tests/test_credit_monitoring.py -k normalization
    """
    data, _, result, _ = trained_version
    restored = data.load_experiment(result["output_dir"])
    predictor = ModelPredictor.from_mlflow(
        result["model_uri"], tracking_uri=restored["tracking_uri"]
    )
    numeric = predictor.model.named_steps["preprocessor"].named_transformers_["num"]
    np.testing.assert_allclose(
        numeric.named_steps["min_max_scaler"].data_max_,
        restored["partitions"]["treino"][data.numeric_features].max(),
    )
    sizes = {metrics["sample_size"] for metrics in result["comparison"].values()}
    assert sizes == {100}


def test_mlflow_persists_metrics_in_database_and_links_runs(trained_version):
    """Consulta o SQLite para comprovar persistência e confere a linhagem no MLflow.

    Exemplo:
        pytest tests/test_credit_monitoring.py -k database
    """
    data, directory, result, _ = trained_version
    reference = data.load_experiment(directory)
    run = MlflowClient().get_run(result["training_run_id"])
    assert run.data.tags["baseline_run_id"] == reference["training_run_id"]
    assert run.data.tags["mlflow.parentRunId"] == result["workflow_run_id"]
    database = directory / "monitoring/mlflow.db"
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        metric = connection.execute(
            "SELECT value FROM metrics WHERE run_uuid = ? AND key = 'teste_roc_auc'",
            (result["training_run_id"],),
        ).fetchone()
    assert metric[0] == pytest.approx(result["comparison"]["v2"]["roc_auc"])


def test_no_alert_skips_training_and_saves_global_outcome(baseline, monkeypatch):
    """Confere a ausência de retreinamento e o registro do desfecho sem drift.

    Exemplo:
        pytest tests/test_credit_monitoring.py -k no_alert
    """
    _, directory, original = baseline
    original.to_csv(directory / "credit_scoring_v2.csv", index=False)
    train = Mock(side_effect=AssertionError("Não deveria treinar."))
    monkeypatch.setattr(CreditMonitoring, "train_model", train)
    result = run_workflow(directory)
    train.assert_not_called()
    assert result["status"] == "sem_drift"
    assert not (Path(result["output_dir"]) / "experiment.joblib").exists()
    run = MlflowClient().get_run(result["workflow_run_id"])
    assert run.info.status == "FINISHED" and run.data.tags["status"] == "sem_drift"


def test_monitoring_preserves_lineage_and_evidence(trained_version):
    """Confere a ligação entre workflow, cenários e candidato e os relatórios salvos.

    Exemplo:
        pytest tests/test_credit_monitoring.py -k monitoring_preserves
    """
    _, _, result, _ = trained_version
    client = MlflowClient()
    monitoring = client.get_run(result["monitoring_run_id"])
    training = client.get_run(result["training_run_id"])
    assert monitoring.data.tags["mlflow.parentRunId"] == result["workflow_run_id"]
    assert training.data.tags["monitoring_run_id"] == monitoring.info.run_id
    children = client.search_runs(
        [monitoring.info.experiment_id],
        filter_string=f"tags.`mlflow.parentRunId` = '{monitoring.info.run_id}'",
    )
    assert {run.info.run_name for run in children} == {"controle", "data_drift"}
    assert all(run.info.status == "FINISHED" for run in children)
    artifacts = {item.path for item in client.list_artifacts(monitoring.info.run_id)}
    assert {"histogramas.png", "summary.json", "pipeline.jsonl"} <= artifacts


def invalid_data(original, problem):
    """Introduz uma violação por caso, preservando o DataFrame original.

    Exemplo:
        invalid = invalid_data(original, "missing_target")
    """
    invalid = original.copy(deep=True)
    if problem == "invalid_income":
        invalid.loc[0, "AMT_INCOME_TOTAL"] = np.nan
    elif problem == "missing_id":
        invalid = invalid.iloc[1:]
    else:
        invalid = invalid.drop(columns="TARGET")
    return invalid


@pytest.mark.parametrize("problem", ["invalid_income", "missing_id", "missing_target"])
def test_invalid_version_is_blocked_before_model_loading(
    baseline, monkeypatch, problem
):
    """Bloqueia a base inválida e registra a execução completa como falha.

    Exemplo:
        pytest tests/test_credit_monitoring.py -k invalid_version
    """
    _, directory, original = baseline
    invalid_data(original, problem).to_csv(
        directory / "credit_scoring_v2.csv", index=False
    )
    load = Mock(side_effect=AssertionError("Não deveria carregar o modelo."))
    monkeypatch.setattr(ModelPredictor, "from_mlflow", load)
    with pytest.raises((ValueError, pa.errors.SchemaErrors)):
        run_workflow(directory)
    load.assert_not_called()
    experiment = MlflowClient().get_experiment_by_name(EXPERIMENT_NAME)
    runs = MlflowClient().search_runs([experiment.experiment_id])
    assert len(runs) == 1 and runs[0].info.status == "FAILED"


def test_performance_alert_triggers_training_without_feature_drift(baseline):
    """Confere que uma mudança dos desfechos também aciona treinamento.

    Exemplo:
        pytest tests/test_credit_monitoring.py -k performance_alert
    """
    _, directory, original = baseline
    concept = original.copy(deep=True)
    concept["TARGET"] = 1 - concept["TARGET"]
    concept.to_csv(directory / "credit_scoring_v2.csv", index=False)
    result = run_workflow(directory)
    assert result["alerts"] == {"drift_alert": 0, "performance_alert": 1}
    assert result["status"] == "treinado"


def test_evaluation_failure_marks_training_and_workflow_failed(baseline, monkeypatch):
    """Confere falha global quando a avaliação é interrompida depois do fit.

    Exemplo:
        pytest tests/test_credit_monitoring.py -k evaluation_failure
    """
    data, directory, original = baseline
    write_shifted_data(data, directory, original)
    failure = Mock(side_effect=RuntimeError("Falha simulada depois do fit"))
    monkeypatch.setattr(ModelEvaluator, "evaluate_models", failure)
    with pytest.raises(RuntimeError, match="Falha simulada"):
        run_workflow(directory)
    result = json.loads(
        next((directory / "credit_monitoring").glob("*/result.json")).read_text()
    )
    assert result["status"] == "falha" and result["failed_stage"] == "avaliacao"
    for name in ("workflow_run_id", "training_run_id"):
        assert MlflowClient().get_run(result[name]).info.status == "FAILED"
    assert "model_uri" not in result


def test_missing_baseline_is_recorded_in_mlflow(tmp_path):
    """Confere que uma falha na carga também chega ao banco e ao resultado local.

    Exemplo:
        pytest tests/test_credit_monitoring.py -k missing_baseline
    """
    directory = tmp_path / "processed"
    with pytest.raises(FileNotFoundError):
        run_workflow(directory)
    result = json.loads(
        next((directory / "credit_monitoring").glob("*/result.json")).read_text()
    )
    assert result["failed_stage"] == "carga"
    assert MlflowClient().get_run(result["workflow_run_id"]).info.status == "FAILED"


def assert_documented_short_functions(path, names=None):
    """Verifica o limite físico, incluindo assinatura, docstring e comentários.

    Exemplo:
        assert_documented_short_functions(Path("src/credit_monitoring.py"))
    """
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and (names is None or node.name in names):
            assert node.end_lineno - node.lineno + 1 <= 20, f"{path}: {node.name}"
            assert "Exemplo:" in (ast.get_docstring(node) or ""), node.name


@pytest.mark.parametrize(
    "filename,names",
    [
        ("src/credit_monitoring.py", None),
        ("tests/test_credit_monitoring.py", None),
        (
            "ml_prep_kit/src/ml_prep_kit/home_credit_data.py",
            {"prepare_version", "_clean_version", "_align_version", "model_datasets"},
        ),
        (
            "ml_prep_kit/src/ml_prep_kit/experiment_tracker.py",
            {
                "run",
                "workflow",
                "step",
                "save_workflow_result",
                "log_evaluation",
                "from_directory",
                "fit_model",
                "train_model",
            },
        ),
        (
            "ml_prep_kit/src/ml_prep_kit/model_evaluator.py",
            {"save_evaluation", "_save_evaluation_tables"},
        ),
    ],
)
def test_changed_functions_have_examples_and_at_most_twenty_lines(filename, names):
    """Impede regressão do padrão de documentação e tamanho acordado no projeto.

    Exemplo:
        pytest tests/test_credit_monitoring.py -k twenty_lines
    """
    root = Path(__file__).resolve().parents[1]
    assert_documented_short_functions(root / filename, names)


def test_model_datasets_exclude_identifiers_target_and_production():
    """Garante isolamento de produção e ausência de ID e alvo nas features.

    Exemplo:
        pytest tests/test_credit_monitoring.py -k model_datasets
    """
    data = HomeCreditData()
    partitions = data.split(sample_data())
    datasets = data.model_datasets(partitions)
    assert set(datasets) == {"treino", "validacao", "teste"}
    for name, (features, target) in datasets.items():
        assert list(features.columns) == data.features
        pd.testing.assert_series_equal(target, partitions[name][data.target])
        assert not set(features.index) & set(partitions["producao"].index)
    with pytest.raises(ValueError, match="sobreposição"):
        data.model_datasets(dict(partitions, teste=partitions["treino"]))


@pytest.fixture
def evaluation_tables():
    """Fornece um relatório real e tabelas pequenas para testar sua persistência.

    Exemplo:
        def test_report(evaluation_tables):
            assert "reports" in evaluation_tables
    """
    report = ModelEvaluator().evaluate_binary_classification([0, 1], [0.1, 0.8])
    return {
        "reports": {"modelo": {"teste": report}},
        "global_metrics": pd.DataFrame([report["metrics"]]),
        "class_metrics": report["by_class"],
        "scores": {"modelo": {"teste": np.array([0.1, 0.8])}},
    }


@pytest.fixture(params=[False, True])
def exported_evaluation(request, tmp_path, evaluation_tables):
    """Exporta avaliações com e sem intervalos para verificar as duas formas de uso.

    Exemplo:
        def test_export(exported_evaluation):
            directory, with_intervals = exported_evaluation
    """
    intervals = pd.DataFrame({"metric": ["roc_auc"], "lower": [0.9], "upper": [1.0]})
    thresholds = pd.DataFrame({"threshold": [0.5], "f1": [1.0]})
    ModelEvaluator().save_evaluation(
        evaluation_tables,
        intervals if request.param else None,
        thresholds,
        {},
        tmp_path,
    )
    return tmp_path, request.param


def test_evaluation_export_omits_individual_scores(exported_evaluation):
    """Verifica relatório completo e ausência dos scores individuais no JSON.

    Exemplo:
        pytest tests/test_credit_monitoring.py -k omits_individual
    """
    directory, with_intervals = exported_evaluation
    saved = json.loads((directory / "classification_report.json").read_text())
    assert saved["reports"]["modelo"]["teste"]["metrics"]["roc_auc"] == 1.0
    assert "scores" not in saved
    assert (saved["confidence_intervals"] is not None) == with_intervals


def test_evaluation_export_preserves_tables(exported_evaluation, evaluation_tables):
    """Confere métricas e gravação de intervalos apenas quando calculados.

    Exemplo:
        pytest tests/test_credit_monitoring.py -k preserves_tables
    """
    directory, with_intervals = exported_evaluation
    assert (directory / "confidence_intervals.csv").exists() == with_intervals
    expected = evaluation_tables["global_metrics"]
    pd.testing.assert_frame_equal(
        pd.read_csv(directory / "global_metrics.csv", index_col=0), expected
    )
    assert (directory / "metrics_by_class.csv").exists()
    assert (directory / "threshold_metrics.csv").exists()


def test_evaluation_without_intervals_removes_previous_interval_table(
    tmp_path, evaluation_tables
):
    """Evita apresentar intervalos de uma execução anterior como evidência atual.

    Exemplo:
        pytest tests/test_credit_monitoring.py -k previous_interval
    """
    evaluator = ModelEvaluator()
    intervals = pd.DataFrame({"metric": ["roc_auc"], "lower": [0.9], "upper": [1.0]})
    evaluator.save_evaluation(evaluation_tables, intervals, None, {}, tmp_path)
    evaluator.save_evaluation(evaluation_tables, None, None, {}, tmp_path)
    assert not (tmp_path / "confidence_intervals.csv").exists()
    saved = json.loads((tmp_path / "classification_report.json").read_text())
    assert saved["confidence_intervals"] is None


def test_fit_requires_tracking_before_modifying_model(tmp_path):
    """Bloqueia ajuste sem execução ativa, evitando modelos sem rastreabilidade.

    Exemplo:
        pytest tests/test_credit_monitoring.py -k fit_requires
    """
    tracker = ExperimentTracker.from_directory("fit", tmp_path)
    model = Mock()
    with pytest.raises(RuntimeError, match="tracker.run"):
        tracker.fit_model(model, [[0], [1]], [0, 1])
    model.fit.assert_not_called()
