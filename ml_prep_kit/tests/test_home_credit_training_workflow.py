import unittest
from unittest.mock import Mock, patch

import pandas as pd

from ml_prep_kit import HomeCreditTrainingWorkflow


class TestHomeCreditTrainingWorkflow(unittest.TestCase):
    @patch("ml_prep_kit.home_credit_training_workflow.FeaturePreprocessor")
    @patch("ml_prep_kit.home_credit_training_workflow.ModelFactory")
    @patch("ml_prep_kit.home_credit_training_workflow.ExperimentTracker.from_directory")
    def test_train_and_save_share_provenance_metadata(
        self, tracker_factory, model_factory, preprocessor_factory
    ):
        tracker = Mock()
        tracker.tracking_uri = "sqlite:///tracking.db"
        tracker.train_model.side_effect = [
            {"run_id": "dummy-run", "model_uri": "models:/dummy"},
            {"run_id": "logistic-run", "model_uri": "models:/logistic"},
        ]
        tracker_factory.return_value = tracker
        model_factory.return_value.create_pipeline.return_value = Mock()
        preprocessor_factory.return_value.create_pipeline.return_value = Mock()

        workflow = HomeCreditTrainingWorkflow(
            "data/raw",
            "data/processed",
            "training",
            sample_size=10,
            random_state=7,
        )
        workflow.source_hash = "source-digest"
        workflow.datasets = {"treino": (pd.DataFrame({"feature": [1, 2]}), [0, 1])}
        workflow.partitions = {"treino": pd.DataFrame({"feature": [1, 2]})}
        workflow.preparation.save_experiment = Mock()

        models, records = workflow.train_models()
        workflow.save_experiment({"threshold": 0.4, "training_run_id": "logistic-run"})

        self.assertEqual(set(models), {"dummy", "logistic_regression"})
        self.assertEqual(records["logistic_regression"]["run_id"], "logistic-run")
        workflow.preparation.save_experiment.assert_called_once_with(
            workflow.processed_directory,
            workflow.partitions,
            {
                "tracking_uri": "sqlite:///tracking.db",
                "source_sha256": "source-digest",
                "random_state": 7,
                "sample_size": 10,
                "threshold": 0.4,
                "training_run_id": "logistic-run",
            },
        )


if __name__ == "__main__":
    unittest.main()