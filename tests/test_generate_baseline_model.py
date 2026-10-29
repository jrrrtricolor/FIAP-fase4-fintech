"""Avaliação do baseline com limiar de validação e evidências rastreáveis."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from credit_monitoring.generate_baseline_model import BaselineModelTrainer

REPORTS = {
    "dummy": {"teste": {"metrics": {"recall": 0.0}}},
    "logistic_regression": {"teste": {"metrics": {"recall": 0.7}}},
}
EVALUATION = {
    "scores": {"logistic_regression": {"teste": "test-scores"}},
    "reports": REPORTS,
}
VALIDATION = {"scores": {"logistic_regression": {"validacao": "validation-scores"}}}


@pytest.fixture
def trainer(tmp_path):
    """Prepara um coordenador isolado, sem carregar dados reais ou criar runs.

    Exemplo:
        def test_threshold(trainer): ...
    """
    trainer = BaselineModelTrainer.__new__(BaselineModelTrainer)
    trainer.models = {"dummy": object(), "logistic_regression": object()}
    trainer.datasets = {
        name: (f"X_{name}", f"y_{name}") for name in ("treino", "validacao", "teste")
    }
    trainer.processed_path = tmp_path
    trainer.workflow = SimpleNamespace(preparation=SimpleNamespace(random_state=42))
    trainer.training_records = {
        name: {"run_id": f"{name}-run", "model_uri": f"runs:/{name}/model"}
        for name in trainer.models
    }
    trainer.tracker = Mock()
    trainer.evaluator = configured_evaluator(tmp_path)
    return trainer


def configured_evaluator(tmp_path):
    """Fornece resultados de validação e teste distinguíveis entre si.

    Exemplo:
        evaluator = configured_evaluator(tmp_path)
    """
    evaluator = Mock()
    evaluator.evaluate_models.side_effect = [VALIDATION, EVALUATION]
    evaluator.select_threshold.return_value = (0.35, "threshold-results")
    evaluator.bootstrap_binary_metrics.return_value = "confidence-intervals"
    evaluator.save_evaluation.return_value = tmp_path / "evaluation"
    return evaluator


def test_threshold_selection_uses_validation_only(trainer):
    """Impede escolher o limiar com os rótulos da partição de teste.

    Exemplo:
        pytest tests/test_generate_baseline_model.py -k validation_only
    """
    assert trainer.evaluate_models() is EVALUATION
    first, second = trainer.evaluator.evaluate_models.call_args_list
    assert first.args[1] == {"validacao": trainer.datasets["validacao"]}
    trainer.evaluator.select_threshold.assert_called_once_with(
        "y_validacao", "validation-scores"
    )
    assert second.args[1] is trainer.datasets
    assert second.args[2] == {"dummy": 0.5, "logistic_regression": 0.35}
    assert trainer.selected_threshold == 0.35


def test_test_confidence_uses_frozen_validation_threshold(trainer):
    """Calcula a incerteza do teste com o limiar já escolhido na validação.

    Exemplo:
        pytest tests/test_generate_baseline_model.py -k test_confidence
    """
    trainer.evaluate_models()
    trainer.evaluator.bootstrap_binary_metrics.assert_called_once_with(
        "y_teste",
        "test-scores",
        threshold=0.35,
        n_resamples=500,
        confidence_level=0.95,
        random_state=42,
    )


def test_evaluation_artifacts_and_metrics_stay_linked_to_training(trainer):
    """Preserva os relatórios de cada modelo na respectiva run de treinamento.

    Exemplo:
        pytest tests/test_generate_baseline_model.py -k artifacts
    """
    trainer.evaluate_models()
    assert trainer.tracker.log_evaluation.call_count == 2
    for name, record in trainer.training_records.items():
        trainer.tracker.log_evaluation.assert_any_call(
            record["run_id"],
            REPORTS[name],
            artifacts=[trainer.processed_path / "evaluation"],
            parameters={"threshold": 0.35 if name == "logistic_regression" else 0.5},
        )


def test_baseline_initialization_uses_exact_registered_version(trainer):
    """Evita promover outra versão registrada entre o treino e a inicialização.

    Exemplo:
        pytest tests/test_generate_baseline_model.py -k exact_registered_version
    """
    trainer.tracker.register_approved_model.return_value = SimpleNamespace(version="7")
    trainer.register_model()
    trainer.tracker.register_approved_model.assert_called_once_with(
        "runs:/logistic_regression/model", "home_credit_model", 0.5
    )
    trainer.tracker.promote_approved_model_version.assert_called_once_with(
        "home_credit_model", "7", None
    )
    trainer.tracker.promote_latest_model_version.assert_not_called()


def test_evaluation_persists_threshold_and_confidence_intervals(trainer):
    """Salva as evidências que permitem interpretar o resultado do baseline.

    Exemplo:
        pytest tests/test_generate_baseline_model.py -k confidence_intervals
    """
    trainer.evaluate_models()
    trainer.evaluator.save_evaluation.assert_called_once_with(
        EVALUATION,
        "confidence-intervals",
        "threshold-results",
        figures=None,
        directory=trainer.processed_path / "model/evaluation",
    )


@pytest.fixture
def persisted_reference(trainer):
    """Simula uma referência existente e sua leitura pelo framework.

    Exemplo:
        def test_reuse(trainer, persisted_reference): ...
    """
    artifact = trainer.processed_path / "experiment.joblib"
    artifact.write_bytes(b"baseline-preservado")
    trainer.workflow.preparation.load_experiment = Mock(
        return_value={"model_uri": "runs:/saved/model"}
    )
    return artifact


def test_reinitialization_preserves_published_reference(trainer, persisted_reference):
    """Recusa novo bootstrap antes de preparar, treinar ou escrever evidências.

    Exemplo:
        pytest tests/test_generate_baseline_model.py -k reinitialization
    """
    trainer.tracker.model_alias_version.return_value = "7"
    trainer.workflow = Mock()
    with pytest.raises(ValueError, match="Produção já inicializada"):
        trainer.run()
    assert trainer.workflow.mock_calls == []
    assert trainer.evaluator.mock_calls == []
    trainer.tracker.register_approved_model.assert_not_called()
    assert persisted_reference.read_bytes() == b"baseline-preservado"


def test_initialization_reuses_saved_baseline(trainer, persisted_reference):
    """Publica o modelo persistido sem treinar novamente nem reescrever a referência.

    Exemplo:
        pytest tests/test_generate_baseline_model.py -k reuses_saved
    """
    trainer.tracker.model_alias_version.return_value = None
    trainer.tracker.register_approved_model.return_value = SimpleNamespace(version="8")
    trainer.run()
    trainer.tracker.register_approved_model.assert_called_once_with(
        "runs:/saved/model", "home_credit_model", 0.5
    )
    trainer.tracker.promote_approved_model_version.assert_called_once_with(
        "home_credit_model", "8", None
    )
    trainer.workflow.preparation.load_experiment.assert_called_once_with(
        trainer.processed_path
    )
    assert trainer.evaluator.mock_calls == []
    assert persisted_reference.read_bytes() == b"baseline-preservado"
