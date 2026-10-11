"""Monitora o modelo de crédito e treina um candidato quando há alertas.

Fluxo: carga → limpeza/contrato → drift → normalização → treino → avaliação → MLflow.
Modelagem: sklearn.pipeline.Pipeline criado pelo ModelFactory do ml_prep_kit.
Execute depois dos notebooks 02 e 03: python -m src.credit_monitoring

Execuções, parâmetros e métricas: data/processed/monitoring/mlflow.db (SQLite).
Modelos e relatórios: data/processed/monitoring/credit_artifacts/.
Partições e resultado: data/processed/credit_monitoring/<execução>/.

A referência permanece preservada. A v2 simula mudanças nas entradas mantendo
os rótulos: retreinar não comprova melhora e não promove o modelo para produção.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from ml_prep_kit import (
    CSVDataLoader,
    DriftMonitor,
    ExperimentTracker,
    FeaturePreprocessor,
    HomeCreditData,
    ModelEvaluator,
    ModelFactory,
    ModelPredictor,
    VisualizationReporter,
)

MODEL_PARAMETERS = {"max_iter": 2_000, "class_weight": "balanced"}
EXPERIMENT_NAME = "tech-challenge-4-credit-monitoring"


def run_workflow(processed_dir, data_path=None, **drift_parameters):
    """Executa o monitoramento; os alertas determinam a necessidade de retreinar.

    Exemplo:
        result = run_workflow("data/processed", drift_share=0.20)
    """
    workflow = CreditMonitoring(processed_dir, data_path, drift_parameters)
    with workflow.tracker.workflow("credit_monitoring", workflow.result):
        workflow.step("carga", workflow.load_data)
        workflow.step("limpeza_e_contrato", workflow.clean_data)
        # Comparar as entradas antes de aprender uma nova normalização.
        workflow.step("deteccao_de_drift", workflow.detect_drift)
        if workflow.should_retrain():
            workflow.retrain()
    return workflow.result


class CreditMonitoring:
    """Conecta dados, monitoramento e treinamento aos componentes do ml_prep_kit.

    Exemplo:
        workflow = CreditMonitoring("data/processed")
        print(workflow.tracker.tracking_uri)
    """

    def __init__(self, processed_dir, data_path=None, drift_parameters=None):
        """Define entradas e uma saída exclusiva para preservar a referência.

        Exemplo:
            workflow = CreditMonitoring("data/processed")
        """
        self.processed_dir = Path(processed_dir).resolve()
        self.data_path = Path(data_path or self.processed_dir / "credit_scoring_v2.csv")
        self.drift_parameters = drift_parameters or {}
        self.data = HomeCreditData()
        self.evaluator = ModelEvaluator()
        execution_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        self.output_dir = self.processed_dir / "credit_monitoring" / execution_id
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.result = {"status": "iniciado", "output_dir": str(self.output_dir)}
        self.tracker = ExperimentTracker.from_directory(
            EXPERIMENT_NAME, self.processed_dir / "monitoring", "credit_artifacts"
        )

    def step(self, name, action):
        """Executa a etapa identificada nos logs e no desfecho de uma eventual falha.

        Exemplo:
            workflow.step("carga", workflow.load_data)
        """
        return self.tracker.step(name, action, self.result)

    def load_data(self):
        """Lê referência e lote atual; relaciona dados e modelo por hash e run ID.

        Exemplo:
            workflow.load_data()
        """
        self.baseline = self.data.load_experiment(self.processed_dir)
        self.data.random_state = self.baseline["random_state"]
        self.frame = CSVDataLoader(self.data_path.parent).load(
            {"current": self.data_path.name}
        )["current"]
        self.tags = {
            "source_sha256": self.data.source_hash(self.data_path),
            "baseline_run_id": self.baseline["training_run_id"],
            "baseline_model_uri": self.baseline["model_uri"],
            "workflow_run_id": self.result["workflow_run_id"],
            "random_state": self.data.random_state,
            "sample_size": len(self.frame),
        }
        self.tracker.log(tags=self.tags, metrics={"input_rows": len(self.frame)})

    def clean_data(self):
        """Limpa categorias, aplica o contrato e preserva os IDs de cada partição.

        Exemplo:
            workflow.clean_data()
        """
        # O CSV preparado já contém idade e flags convertidas pelos notebooks.
        self.partitions = self.data.prepare_version(
            self.frame, self.baseline["partitions"]
        )
        self.reference = self.baseline["partitions"]["treino"]
        self.scenarios = {
            "controle": self.baseline["partitions"]["producao"],
            "data_drift": self.partitions["producao"],
        }
        self.tracker.log(
            parameters={"contract": self.data.contract.name},
            metrics={"quality_passed": 1},
        )

    def prepare_monitor(self):
        """Recupera o modelo original e configura os testes KS/PSI do framework.

        Exemplo:
            monitor = workflow.prepare_monitor()
        """
        self.predictor = ModelPredictor.from_mlflow(
            self.baseline["model_uri"], tracking_uri=self.tracker.tracking_uri
        )
        return DriftMonitor(
            self.data.numeric_features,
            self.data.categorical_features + self.data.binary_features,
            self.output_dir / "drift",
            **self.drift_parameters,
        )

    def detect_drift(self):
        """Compara controle e lote atual; o framework registra métricas e relatórios.

        Exemplo:
            workflow.detect_drift()
        """
        monitor = self.prepare_monitor()
        monitoring = monitor.evaluate_scenarios(
            self.scenarios,
            self.reference,
            self.predictor,
            self.data.contract,
            self.data.target,
            self.tracker,
            threshold=self.baseline["threshold"],
            tags=self.tags,
            figures={"histogramas": self.drift_histograms()},
        )
        self.drift_metrics = monitoring["results"]["data_drift"]["metrics"]
        self.result["monitoring_run_id"] = monitoring["run_id"]

    def drift_histograms(self):
        """Sobrepõe referência e produção, com bins comuns e escala log monetária.

        Exemplo:
            figure = workflow.drift_histograms()
        """
        return VisualizationReporter().plot_overlaid_histograms(
            self.reference,
            self.partitions["producao"],
            columns=["AMT_INCOME_TOTAL", "AMT_CREDIT", "AGE_YEARS"],
            log_columns=["AMT_INCOME_TOTAL", "AMT_CREDIT"],
            labels=("Referência v1", "Produção v2"),
        )

    def should_retrain(self):
        """Aciona treinamento por drift de entradas ou degradação do desempenho.

        Exemplo:
            needs_training = workflow.should_retrain()
        """
        alerts = {
            name: int(self.drift_metrics.get(name) or 0)
            for name in ("drift_alert", "performance_alert")
        }
        self.result["alerts"] = alerts
        self.tracker.log(metrics=alerts)
        if any(alerts.values()):
            return True
        self.result["status"] = "sem_drift"
        return False

    def retrain(self):
        """Rastreia todo o ciclo do candidato; falhas na avaliação invalidam o treino.

        Exemplo:
            workflow.retrain()
        """
        self.tags["monitoring_run_id"] = self.result["monitoring_run_id"]
        with self.tracker.run("credit_candidate", tags=self.tags) as run:
            self.result["training_run_id"] = run.info.run_id
            self.tracker.log(
                parameters={
                    **MODEL_PARAMETERS,
                    "seed": self.data.random_state,
                    "train_rows": len(self.partitions["treino"]),
                }
            )
            self.step("configuracao_da_normalizacao", self.configure_normalization)
            self.step("treinamento", self.train_model)
            self.step("avaliacao", self.evaluate_model)
            self.step("registro_mlflow", self.register_model)

    def configure_normalization(self):
        """Monta o Pipeline do scikit-learn com preparação e classificador juntos.

        Exemplo:
            workflow.configure_normalization()
        """
        preprocessor = FeaturePreprocessor(
            self.data.numeric_features,
            self.data.categorical_features,
            self.data.binary_features,
        ).create_pipeline()
        self.model = ModelFactory(random_state=self.data.random_state).create_pipeline(
            preprocessor, "classification", "logistic_regression", MODEL_PARAMETERS
        )

    def train_model(self):
        """Ajusta normalização e classificador somente na partição de treino.

        Exemplo:
            workflow.train_model()
        """
        train = self.partitions["treino"]
        # Validação, teste e produção não participam do fit.
        self.tracker.fit_model(
            self.model, train[self.data.features], train[self.data.target]
        )

    def select_threshold(self):
        """Escolhe o limiar na validação, preservando o teste para comparação final.

        Exemplo:
            workflow.select_threshold()
        """
        validation = self.partitions["validacao"]
        scores = ModelPredictor(self.model).positive_scores(
            validation[self.data.features]
        )
        self.threshold, self.threshold_metrics = self.evaluator.select_threshold(
            validation[self.data.target], scores
        )

    def evaluate_model(self):
        """Compara referência e candidato nas mesmas partições, sem alterar o teste.

        Exemplo:
            workflow.evaluate_model()
        """
        self.select_threshold()
        self.evaluation = self.evaluator.evaluate_models(
            {"v1": self.predictor.model, "v2": self.model},
            self.data.model_datasets(self.partitions),
            thresholds={"v1": self.baseline["threshold"], "v2": self.threshold},
        )
        self.result["comparison"] = {
            name: report["teste"]["metrics"]
            for name, report in self.evaluation["reports"].items()
        }

    def register_model(self):
        """Salva candidato, confere sua recarga e persiste evidências no MLflow.

        Exemplo:
            workflow.register_model()
        """
        self.result["model_uri"] = self.tracker.log_model(
            self.model, features=self.partitions["treino"][self.data.features]
        )
        self.verify_saved_model()
        self.save_evaluation()
        self.result.update(
            threshold=self.threshold, tracking_uri=self.tracker.tracking_uri
        )
        self.data.save_experiment(
            self.output_dir,
            self.partitions,
            {**self.tags, **self.result, "status": "treinado"},
        )
        self.result["status"] = "treinado"

    def verify_saved_model(self):
        """Exige equivalência dos scores antes e depois de recuperar o modelo salvo.

        Exemplo:
            workflow.verify_saved_model()
        """
        reloaded = ModelPredictor.from_mlflow(
            self.result["model_uri"], tracking_uri=self.tracker.tracking_uri
        )
        reloaded.assert_equivalent(
            ModelPredictor(self.model), self.partitions["teste"][self.data.features]
        )

    def save_evaluation(self):
        """Salva comparação completa e registra as métricas do candidato no banco.

        Exemplo:
            workflow.save_evaluation()
        """
        directory = self.evaluator.save_evaluation(
            self.evaluation,
            intervals=None,
            thresholds=self.threshold_metrics,
            figures=None,
            directory=self.output_dir / "evaluation",
        )
        self.tracker.log_evaluation(
            None,
            self.evaluation["reports"]["v2"],
            parameters={"threshold": self.threshold},
            artifacts=[directory],
        )

def main():
    """Lê parâmetros do terminal e apresenta o resultado completo em JSON.

    Exemplo:
        python -m src.credit_monitoring --drift-share 0.20
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--data", dest="data_path", type=Path, help="CSV preparado atual"
    )
    parser.add_argument("--ks-threshold", type=float, default=0.01)
    parser.add_argument("--psi-threshold", type=float, default=0.20)
    parser.add_argument("--drift-share", type=float, default=0.20)
    result = run_workflow(**vars(parser.parse_args()))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
