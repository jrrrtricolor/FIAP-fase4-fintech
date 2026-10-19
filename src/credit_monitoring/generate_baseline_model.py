from pathlib import Path

from ml_prep_kit import (
    HomeCreditTrainingWorkflow,
    ModelEvaluator,
)

from .api_constants import MLFLOW_MODEL_NAME


class BaselineModelTrainer:
    """Orchestrates baseline preparation, evaluation, and model promotion."""

    def __init__(self, project_root=None, sample_size=20_000, random_state=42):
        self.project_root = Path.cwd() if project_root is None else Path(project_root)
        self.raw_path = self.project_root / "data/raw/home-credit-default-risk"
        self.processed_path = self.project_root / "data/processed"
        self.workflow = HomeCreditTrainingWorkflow(
            self.raw_path,
            self.processed_path,
            "tech-challenge-4-fintech-training",
            sample_size=sample_size,
            random_state=random_state,
        )
        self.evaluator = ModelEvaluator()
        self.tracker = self.workflow.tracker
        self.datasets = None
        self.models = None
        self.training_records = None
        self.evaluation = None
        self.selected_threshold = None
        self.threshold_results = None
        self.confidence_intervals = None
        self.evaluation_path = None

    def prepare_data(self):
        clean_df = self.workflow.load_data()
        _, self.datasets = self.workflow.split_data(clean_df)
        return self.datasets

    def train_models(self):
        self.models, self.training_records = self.workflow.train_models()
        return self.models

    def evaluate_models(self):
        thresholds = self.select_validation_threshold()
        self.evaluate_datasets(thresholds)
        self.save_evaluation(thresholds)
        return self.evaluation

    def select_validation_threshold(self):
        if self.datasets is None or self.models is None:
            raise RuntimeError(
                "Prepare os dados e treine os modelos antes da avaliação."
            )
        datasets = self.datasets
        models = self.models
        X_validation, y_validation = datasets["validacao"]
        validation = self.evaluator.evaluate_models(
            models,
            {"validacao": (X_validation, y_validation)},
            thresholds={"dummy": 0.5, "logistic_regression": 0.5},
        )
        self.selected_threshold, self.threshold_results = (
            self.evaluator.select_threshold(
                y_validation,
                validation["scores"]["logistic_regression"]["validacao"],
            )
        )
        return {
            "dummy": 0.5,
            "logistic_regression": self.selected_threshold,
        }

    def evaluate_datasets(self, thresholds):
        if self.datasets is None or self.models is None:
            raise RuntimeError(
                "Prepare os dados e treine os modelos antes da avaliação."
            )
        datasets = self.datasets
        self.evaluation = self.evaluator.evaluate_models(
            self.models,
            datasets,
            thresholds,
        )
        self.confidence_intervals = self.evaluator.bootstrap_binary_metrics(
            datasets["teste"][1],
            self.evaluation["scores"]["logistic_regression"]["teste"],
            threshold=self.selected_threshold,
            n_resamples=500,
            confidence_level=0.95,
            random_state=self.workflow.preparation.random_state,
        )

    def save_evaluation(self, thresholds):
        self.evaluation_path = self.evaluator.save_evaluation(
            self.evaluation,
            self.confidence_intervals,
            self.threshold_results,
            figures=None,
            directory=self.processed_path / "model/evaluation",
        )
        self._log_evaluations(thresholds)

    def _log_evaluations(self, thresholds):
        if (
            self.evaluation is None
            or self.training_records is None
            or self.evaluation_path is None
        ):
            raise RuntimeError("Avalie e salve os resultados antes de registrá-los.")
        evaluation = self.evaluation
        training_records = self.training_records
        evaluation_path = self.evaluation_path
        for name, reports in evaluation["reports"].items():
            self.tracker.log_evaluation(
                training_records[name]["run_id"],
                reports,
                artifacts=[evaluation_path],
                parameters={"threshold": thresholds[name]},
            )

    def save_experiment(self):
        if self.training_records is None:
            raise RuntimeError("Treine os modelos antes de salvar o experimento.")
        baseline_record = self.training_records["logistic_regression"]
        self.workflow.save_experiment(
            metadata={
                "threshold": self.selected_threshold,
                "training_run_id": baseline_record["run_id"],
                "model_uri": baseline_record["model_uri"],
            },
        )

    def register_model(self):
        if self.training_records is None:
            raise RuntimeError("Treine os modelos antes de registrar o modelo.")
        baseline_record = self.training_records["logistic_regression"]
        self.tracker.register_model(baseline_record["model_uri"], MLFLOW_MODEL_NAME)
        self.tracker.promote_latest_model_version(
            registered_model_name=MLFLOW_MODEL_NAME
        )

    def run(self):
        self.prepare_data()
        self.train_models()
        self.evaluate_models()
        self.save_experiment()
        self.register_model()


def main():
    BaselineModelTrainer().run()

if __name__ == "__main__":
    main()
