"""Registro reutilizável de experimentos com MLflow."""

import io
import logging
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from time import perf_counter
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Centraliza operações comuns de tracking com MLflow.

    Use esta classe para configurar o experimento uma única vez e registrar
    parâmetros, métricas, artefatos e modelos de forma padronizada.

    Exemplo:
        tracker = ExperimentTracker(
            experiment_name="experimento-classificacao-textual",
            tracking_uri="sqlite:///mlflow.db",
        )

        run_id = tracker.log_training_run(
            run_name="logistic_regression",
            model=pipeline,
            parameters={"model_name": "logistic_regression"},
            metrics={"roc_auc": 0.82},
        )
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
        artifact_location: str | None = None,
    ) -> None:
        """Configura o experimento usado pelos registros.

        Exemplo:
            tracker = ExperimentTracker(
                experiment_name="experimento-classificacao-textual",
                tracking_uri="sqlite:///mlflow.db",
            )
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        self.experiment_id = (
            experiment.experiment_id
            if experiment
            else client.create_experiment(
                experiment_name, artifact_location=artifact_location
            )
        )
        mlflow.set_experiment(experiment_name)

    @contextmanager
    def run(self, name, parameters=None, tags=None, nested=False):
        """Rastreia execução completa, inclusive duração e falha no treinamento.

        Exemplo:
            with tracker.run("logistic_regression", parameters={"seed": 42}):
                model.fit(X_train, y_train)
                model_uri = tracker.log_model(model, features=X_train)
        """
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        started = perf_counter()
        with mlflow.start_run(
            experiment_id=self.experiment_id, run_name=name, nested=nested
        ) as run:
            self.log(parameters=parameters, tags=tags)
            try:
                yield run
            finally:
                mlflow.log_metric("run_duration_seconds", perf_counter() - started)

    def log(self, parameters=None, metrics=None, tags=None, artifacts=None):
        """Registra evidências na execução ativa; métricas indefinidas são omitidas.

        Exemplo:
            with tracker.run("avaliacao"):
                tracker.log(
                    parameters={"threshold": 0.5},
                    metrics={"roc_auc": 0.63, "recall": None},
                    tags={"split": "teste"},
                    artifacts=["data/processed/classification_report.json"],
                )
        """
        if parameters:
            mlflow.log_params(self._stringify_values(parameters))
        if metrics:
            mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
        if tags:
            mlflow.set_tags(self._stringify_values(tags))
        for artifact in artifacts or []:
            path = Path(artifact)
            if path.is_dir():
                mlflow.log_artifacts(str(path), artifact_path=path.name)
            else:
                mlflow.log_artifact(str(path))

    def log_model(
        self, model, features=None, name="model", registered_model_name=None
    ) -> str:
        """Salva pipeline local do framework e assinatura, sem amostras de clientes.

        Cloudpickle preserva os transformers Python do framework. Como Joblib,
        a recarga é destinada a artefatos confiáveis gerados pelo próprio fluxo.

        Exemplo:
            with tracker.run("treinamento"):
                pipeline.fit(X_train, y_train)
                model_uri = tracker.log_model(pipeline, features=X_train)
        """
        signature = None
        if features is not None:
            signature = mlflow.models.infer_signature(features, model.predict(features))
        info = mlflow.sklearn.log_model(
            sk_model=model,
            name=name,
            signature=signature,
            serialization_format="cloudpickle",
            registered_model_name=registered_model_name,
        )
        return info.model_uri

    def log_evaluation(self, run_id, reports, artifacts=None, parameters=None):
        """Anexa avaliação final à execução que treinou o modelo.

        Exemplo:
            tracker.log_evaluation(
                training_run_id,
                {"teste": test_report},
                parameters={"selected_threshold": threshold},
            )
        """
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        with mlflow.start_run(run_id=run_id):
            metrics = {
                f"{split}_{key}": value
                for split, report in reports.items()
                for key, value in report["metrics"].items()
            }
            self.log(metrics=metrics, artifacts=artifacts, parameters=parameters)

    def log_training_run(
        self,
        run_name: str,
        model: Any,
        parameters: dict[str, Any],
        metrics: dict[str, float],
        artifacts: list[str | Path] | None = None,
        model_name: str = "model",
        registered_model_name: str | None = None,
    ) -> str:
        """Registra parâmetros, métricas, artefatos e modelo treinado.

        Exemplo:
            run_id = tracker.log_training_run(
                run_name="random_forest",
                model=pipeline,
                parameters={"n_estimators": 100},
                metrics={"f1": 0.74},
                artifacts=["model/report.html"],
            )
        """
        logger.info(
            "Registrando execução no MLflow.",
            extra={
                "evento": "registro_sklearn_mlflow_iniciado",
                "nome_execucao": run_name,
                "nome_experimento": self.experiment_name,
                "quantidade_artefatos": len(artifacts or []),
            },
        )

        with self.run(run_name, parameters=parameters) as run:
            self.log(metrics=metrics, artifacts=artifacts)
            self.log_model(
                model, name=model_name, registered_model_name=registered_model_name
            )

            logger.info(
                "Registro concluído no MLflow.",
                extra={
                    "evento": "registro_sklearn_mlflow_concluido",
                    "nome_execucao": run_name,
                    "execucao_id": run.info.run_id,
                    "nome_experimento": self.experiment_name,
                },
            )

            return run.info.run_id

    def log_pytorch_training_run(
        self,
        run_name: str,
        model: Any,
        parameters: dict[str, Any],
        metrics: dict[str, float],
        artifacts: list[str | Path] | None = None,
        input_example: Any | None = None,
        model_name: str = "model",
        registered_model_name: str | None = None,
    ) -> str:
        """Registra parâmetros, métricas, artefatos e modelo PyTorch.

        Exemplo:
            run_id = tracker.log_pytorch_training_run(
                run_name="text_multiclass_classifier",
                model=torch_model,
                parameters={"epochs": 10, "learning_rate": 0.001},
                metrics={"roc_auc": 0.88},
                input_example=X_valid_ready[:5],
            )
        """
        from mlflow import pytorch as mlflow_pytorch

        logger.info(
            "Registrando rede neural no MLflow.",
            extra={
                "evento": "registro_pytorch_mlflow_iniciado",
                "nome_execucao": run_name,
                "nome_experimento": self.experiment_name,
                "quantidade_artefatos": len(artifacts or []),
            },
        )

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(self._stringify_values(parameters))
            mlflow.log_metrics(metrics)

            for artifact in artifacts or []:
                mlflow.log_artifact(str(artifact))

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The given buffer is not writable.*",
                )
                with (
                    redirect_stdout(io.StringIO()),
                    redirect_stderr(io.StringIO()),
                ):
                    mlflow_pytorch.log_model(
                        pytorch_model=model,
                        name=model_name,
                        input_example=input_example,
                        serialization_format="pt2",
                        registered_model_name=registered_model_name,
                    )

            logger.info(
                "Rede neural registrada no MLflow.",
                extra={
                    "evento": "registro_pytorch_mlflow_concluido",
                    "nome_execucao": run_name,
                    "execucao_id": run.info.run_id,
                    "nome_experimento": self.experiment_name,
                },
            )

            return run.info.run_id

    def promote_latest_model_version(
        self,
        registered_model_name: str,
        alias: str = "production",
        tags: dict[str, Any] | None = None,
    ) -> str:
        """Promove e identifica a versão mais recente do modelo.

        Exemplo:
            version = tracker.promote_latest_model_version(
                registered_model_name="example_classifier",
                alias="champion",
            )
        """
        # Buscar versões registradas para o modelo informado.
        client = MlflowClient()
        versions = client.search_model_versions(f"name = '{registered_model_name}'")

        # Interromper quando o modelo ainda não existir no registro.
        if not versions:
            raise ValueError(
                f"Modelo registrado não encontrado: {registered_model_name}."
            )

        # Selecionar a versão mais recente do modelo.
        latest_version = max(
            versions,
            key=lambda version: int(version.version),
        )

        # Aplicar o alias na versão mais recente.
        client.set_registered_model_alias(
            registered_model_name,
            alias,
            latest_version.version,
        )

        # Tags ligam a versão do Registry ao código que a produziu.
        for key, value in (tags or {}).items():
            client.set_model_version_tag(
                registered_model_name,
                latest_version.version,
                key,
                str(value),
            )
        logger.info(
            "Modelo promovido para produção.",
            extra={
                "evento": "modelo_promovido_registry",
                "nome_modelo_registrado": registered_model_name,
                "versao": latest_version.version,
                "alias": alias,
            },
        )

        return str(latest_version.version)

    def _stringify_values(self, values: dict[str, Any]) -> dict[str, Any]:
        """Converte valores complexos para texto aceito pelo MLflow.

        Exemplo:
            parameters = tracker._stringify_values({"layers": [32, 16]})
        """
        formatted = {}

        for key, value in values.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                formatted[key] = value
            else:
                formatted[key] = str(value)

        return formatted
