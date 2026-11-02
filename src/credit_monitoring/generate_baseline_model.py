"""Produz o baseline compartilhado pelos notebooks, pelo workflow e pela API."""

import os
from pathlib import Path

from ml_prep_kit import HomeCreditTrainingWorkflow, ModelEvaluator

from .api_constants import MLFLOW_MODEL_NAME


class BaselineModelTrainer:
    """Coordena preparação, treinamento, avaliação e registro do baseline.

    Exemplo:
        BaselineModelTrainer(sample_size=20_000).run()
    """

    def __init__(self, project_root=None, sample_size=20_000, random_state=42):
        """Define os dados locais e reutiliza o workflow de treinamento.

        Exemplo:
            trainer = BaselineModelTrainer(".", sample_size=20_000)
        """
        self.project_root = Path.cwd() if project_root is None else Path(project_root)
        self.processed_path = self.project_root / "data/processed"
        self.workflow = HomeCreditTrainingWorkflow(
            self.project_root / "data/raw/home-credit-default-risk",
            self.processed_path,
            "tech-challenge-4-fintech-training",
            sample_size=sample_size,
            random_state=random_state,
            artifact_location=os.environ.get("MLFLOW_ARTIFACT_LOCATION") or None,
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI") or None,
        )
        self.evaluator = ModelEvaluator()
        self.tracker = self.workflow.tracker
        self.datasets = None
        self.models = None
        self.training_records = None

    def prepare_data(self):
        """Limpa os dados e reserva as quatro partições antes do treinamento.

        Exemplo:
            datasets = trainer.prepare_data()
        """
        _, self.datasets = self.workflow.split_data(self.workflow.load_data())
        return self.datasets

    def train_models(self):
        """Treina e registra o Dummy e a regressão logística pelo framework.

        Exemplo:
            models = trainer.train_models()
        """
        self.models, self.training_records = self.workflow.train_models()
        return self.models

    def require_trained(self):
        """Exige preparação e treinamento antes de avaliar o baseline.

        Exemplo:
            trainer.require_trained()
        """
        if self.datasets is None or self.models is None:
            raise RuntimeError(
                "Prepare os dados e treine os modelos antes da avaliação."
            )

    def select_validation_threshold(self):
        """Seleciona o limiar usando exclusivamente a partição de validação.

        Exemplo:
            thresholds = trainer.select_validation_threshold()
        """
        self.require_trained()
        validation = self.evaluator.evaluate_models(
            self.models,
            {"validacao": self.datasets["validacao"]},
            thresholds={"dummy": 0.5, "logistic_regression": 0.5},
        )
        self.selected_threshold, self.threshold_results = (
            self.evaluator.select_threshold(
                self.datasets["validacao"][1],
                validation["scores"]["logistic_regression"]["validacao"],
            )
        )
        return {"dummy": 0.5, "logistic_regression": self.selected_threshold}

    def evaluate_datasets(self, thresholds):
        """Avalia os modelos e estima a incerteza sem reajustar o Pipeline.

        Exemplo:
            trainer.evaluate_datasets(thresholds)
        """
        self.require_trained()
        self.evaluation = self.evaluator.evaluate_models(
            self.models, self.datasets, thresholds
        )
        self.confidence_intervals = self.evaluator.bootstrap_binary_metrics(
            self.datasets["teste"][1],
            self.evaluation["scores"]["logistic_regression"]["teste"],
            threshold=self.selected_threshold,
            n_resamples=500,
            confidence_level=0.95,
            random_state=self.workflow.preparation.random_state,
        )

    def evaluate_models(self):
        """Seleciona o limiar, avalia os modelos e persiste suas evidências.

        Exemplo:
            evaluation = trainer.evaluate_models()
        """
        thresholds = self.select_validation_threshold()
        self.evaluate_datasets(thresholds)
        self.save_evaluation(thresholds)
        return self.evaluation

    def save_evaluation(self, thresholds):
        """Salva as tabelas e associa as métricas às runs de treinamento.

        Exemplo:
            trainer.save_evaluation(thresholds)
        """
        self.evaluation_path = self.evaluator.save_evaluation(
            self.evaluation,
            self.confidence_intervals,
            self.threshold_results,
            figures=None,
            directory=self.processed_path / "model/evaluation",
        )
        self._log_evaluations(thresholds)

    def _log_evaluations(self, thresholds):
        """Anexa as avaliações a cada modelo sem criar um treinamento adicional.

        Exemplo:
            trainer._log_evaluations({"dummy": 0.5, "logistic_regression": 0.4})
        """
        for name, reports in self.evaluation["reports"].items():
            self.tracker.log_evaluation(
                self.training_records[name]["run_id"],
                reports,
                artifacts=[self.evaluation_path],
                parameters={"threshold": thresholds[name]},
            )

    def save_experiment(self):
        """Persiste a referência usada na comparação e no retreinamento.

        Exemplo:
            trainer.save_experiment()
        """
        record = self.training_records["logistic_regression"]
        self.workflow.save_experiment(
            metadata={
                "threshold": self.selected_threshold,
                "training_run_id": record["run_id"],
                "model_uri": record["model_uri"],
            }
        )

    def register_model(self, model_uri=None, threshold=None):
        """Registra o baseline e inicializa produção apenas se ainda não existir.

        Exemplo:
            trainer.register_model()
        """
        if model_uri is None:
            model_uri = self.training_records["logistic_regression"]["model_uri"]
        threshold = (
            threshold
            if threshold is not None
            else getattr(self, "selected_threshold", 0.5)
        )
        self.tracker.approve_baseline_model(model_uri)
        version = self.tracker.register_approved_model(
            model_uri, MLFLOW_MODEL_NAME, threshold
        )
        self.tracker.promote_approved_model_version(
            MLFLOW_MODEL_NAME, version.version, None
        )

    def require_initial_setup(self):
        """Impede substituir a referência de um modelo já publicado.

        Exemplo:
            trainer.require_initial_setup()
        """
        if self.tracker.model_alias_version(MLFLOW_MODEL_NAME) is not None:
            raise ValueError("Produção já inicializada; preserve o baseline existente.")

    def run(self):
        """Executa o fluxo completo na ordem de suas dependências.

        Exemplo:
            trainer.run()
        """
        self.require_initial_setup()
        if (self.processed_path / "experiment.joblib").exists():
            existing = self.workflow.preparation.load_experiment(self.processed_path)
            self.register_model(existing["model_uri"], existing.get("threshold", 0.5))
            return
        self.prepare_data()
        self.train_models()
        self.evaluate_models()
        self.save_experiment()
        self.register_model()


def main():
    """Gera o baseline a partir dos dados brutos disponíveis no projeto.

    Exemplo:
        python -m src.credit_monitoring.generate_baseline_model
    """
    BaselineModelTrainer().run()


if __name__ == "__main__":
    main()
