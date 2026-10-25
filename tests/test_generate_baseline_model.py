from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

from credit_monitoring.generate_baseline_model import BaselineModelTrainer


def test_evaluate_models_uses_validation_threshold_and_logs_reports(tmp_path):
    trainer = BaselineModelTrainer.__new__(BaselineModelTrainer)
    trainer.models = {"dummy": object(), "logistic_regression": object()}
    trainer.datasets = {
        "treino": ("X_train", "y_train"),
        "validacao": ("X_validation", "y_validation"),
        "teste": ("X_test", "y_test"),
    }
    trainer.processed_path = tmp_path
    trainer.workflow = SimpleNamespace(
        preparation=SimpleNamespace(random_state=42)
    )
    trainer.training_records = {
        "dummy": {"run_id": "dummy-run"},
        "logistic_regression": {"run_id": "logistic-run"},
    }
    trainer.tracker = Mock()

    validation = {
        "scores": {"logistic_regression": {"validacao": "validation-scores"}}
    }
    reports = {
        "dummy": {"teste": {"metrics": {"recall": 0.0}}},
        "logistic_regression": {
            "teste": {"metrics": {"recall": 0.7}}
        },
    }
    evaluation = {
        "scores": {"logistic_regression": {"teste": "test-scores"}},
        "reports": reports,
    }
    trainer.evaluator = Mock()
    trainer.evaluator.evaluate_models.side_effect = [validation, evaluation]
    trainer.evaluator.select_threshold.return_value = (0.35, "threshold-results")
    trainer.evaluator.bootstrap_binary_metrics.return_value = "confidence-intervals"
    trainer.evaluator.save_evaluation.return_value = tmp_path / "evaluation"

    result = trainer.evaluate_models()

    assert result is evaluation
    assert trainer.selected_threshold == 0.35
    assert trainer.threshold_results == "threshold-results"
    trainer.evaluator.evaluate_models.assert_has_calls(
        [
            call(
                trainer.models,
                {"validacao": ("X_validation", "y_validation")},
                thresholds={"dummy": 0.5, "logistic_regression": 0.5},
            ),
            call(
                trainer.models,
                trainer.datasets,
                {"dummy": 0.5, "logistic_regression": 0.35},
            ),
        ]
    )
    trainer.evaluator.bootstrap_binary_metrics.assert_called_once_with(
        "y_test",
        "test-scores",
        threshold=0.35,
        n_resamples=500,
        confidence_level=0.95,
        random_state=42,
    )
    trainer.evaluator.save_evaluation.assert_called_once_with(
        evaluation,
        "confidence-intervals",
        "threshold-results",
        figures=None,
        directory=Path(tmp_path) / "model/evaluation",
    )
    assert trainer.tracker.log_evaluation.call_args_list == [
        call(
            "dummy-run",
            reports["dummy"],
            artifacts=[tmp_path / "evaluation"],
            parameters={"threshold": 0.5},
        ),
        call(
            "logistic-run",
            reports["logistic_regression"],
            artifacts=[tmp_path / "evaluation"],
            parameters={"threshold": 0.35},
        ),
    ]
