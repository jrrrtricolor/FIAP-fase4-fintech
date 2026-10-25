"""Registro reutilizável de experimentos com MLflow."""

import io
import logging
import warnings
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from time import perf_counter
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from .model_evaluator import ModelEvaluator
from .structured_logging_configurator import StructuredLoggingConfigurator

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

    @classmethod
    def from_directory(cls, experiment_name, directory, artifact_subdir="artifacts"):
        """Configura tracking local em SQLite e artefatos em uma pasta separada.

        Exemplo:
            tracker = ExperimentTracker.from_directory(
                "credito", "data/processed/monitoring", "training_artifacts"
            )
        """
        directory = Path(directory).resolve()
        artifact_dir = directory / artifact_subdir
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            experiment_name,
            tracking_uri=f"sqlite:///{directory / 'mlflow.db'}",
            artifact_location=artifact_dir.as_uri(),
        )

    @contextmanager
    def run(self, name, parameters=None, tags=None, nested=None):
        """Rastreia duração e falhas; por padrão, execuções internas tornam-se filhas.

        Exemplo:
            with tracker.run("logistic_regression", parameters={"seed": 42}):
                model.fit(X_train, y_train)
        """
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        started = perf_counter()
        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=name,
            nested=mlflow.active_run() is not None if nested is None else nested,
        ) as run:
            self.log(parameters=parameters, tags=tags)
            try:
                yield run
            finally:
                mlflow.log_metric("run_duration_seconds", perf_counter() - started)

    @contextmanager
    def workflow(self, name, result):
        """Registra o desfecho global e a etapa da falha, preservando a exceção.

        Exemplo:
            with tracker.workflow("credito", result):
                tracker.step("carga", load_data, result)
        """
        self.workflow_logger = StructuredLoggingConfigurator.file_logger(
            f"workflow.{name}", Path(result["output_dir"]) / "workflow.jsonl"
        )
        with self.run(name) as run:
            result["workflow_run_id"] = run.info.run_id
            try:
                yield
            except Exception as error:
                result.update(status="falha", error_type=type(error).__name__)
                result["failed_stage"] = result.get("stage", "inicializacao")
                raise
            finally:
                self.save_workflow_result(result)

    def step(self, name, action, result):
        """Identifica e executa uma etapa dentro do contexto workflow.

        Exemplo:
            with tracker.workflow("credito", result):
                tracker.step("carga", load_data, result)
        """
        result["stage"] = name
        self.workflow_logger.info(
            "Etapa iniciada.",
            extra={"etapa": name, "workflow_id": result["workflow_run_id"]},
        )
        return action()

    def save_workflow_result(self, result):
        """Anexa resultado e log ao MLflow antes de encerrar a execução global.

        Exemplo:
            tracker.save_workflow_result(result)
        """
        directory = Path(result["output_dir"])
        self.workflow_logger.log(
            logging.ERROR if result["status"] == "falha" else logging.INFO,
            "Workflow encerrado.",
            extra=result,
        )
        ModelEvaluator.save_report(result, directory / "result.json")
        self.log(
            tags={"status": result["status"]},
            artifacts=[directory / "result.json", directory / "workflow.jsonl"],
        )

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
        """Anexa avaliação ao treino; run_id=None utiliza a execução já ativa.

        Exemplo:
            tracker.log_evaluation(
                training_run_id,
                {"teste": test_report},
                parameters={"selected_threshold": threshold},
            )
        """
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        with mlflow.start_run(run_id=run_id) if run_id else nullcontext():
            metrics = {
                f"{split}_{key}": value
                for split, report in reports.items()
                for key, value in report["metrics"].items()
            }
            self.log(metrics=metrics, artifacts=artifacts, parameters=parameters)

    def train_model(self, name, model, features, target, parameters=None, tags=None):
        """Treina e salva um pipeline em uma execução rastreada, incluindo falhas.

        Ajusta a instância recebida uma única vez nos dados informados. Retorna
        os identificadores usados para anexar avaliações e recuperar o modelo.

        Exemplo:
            record = tracker.train_model(
                "baseline", pipeline, X_train, y_train,
                parameters={"seed": 42}, tags={"source_sha256": source_hash},
            )
            model_uri = record["model_uri"]
            training_run_id = record["run_id"]
        """
        with self.run(name, parameters=parameters, tags=tags) as run:
            self.fit_model(model, features, target)
            model_uri = self.log_model(model, features=features)
            return {"run_id": run.info.run_id, "model_uri": model_uri}

    def fit_model(self, model, features, target):
        """Ajusta o pipeline na execução ativa; permite avaliar antes de persistir.

        Exemplo:
            with tracker.run("candidato"):
                tracker.fit_model(pipeline, X_train, y_train)
        """
        if mlflow.active_run() is None:
            raise RuntimeError("Abra uma execução com tracker.run antes do ajuste.")
        self.log(parameters={"train_rows": len(features)})
        started = perf_counter()
        try:
            return model.fit(features, target)
        finally:
            self.log(metrics={"fit_duration_seconds": perf_counter() - started})

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

    def register_model(self, model_uri, model_name):
        """Registra o artefato e devolve a versão exata, sem promovê-la.

        Exemplo:
            version = tracker.register_model(record["model_uri"], "credito")
        """
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        return mlflow.register_model(model_uri=model_uri, name=model_name)

    def promote_latest_model_version(
        self,
        registered_model_name: str,
        alias: str = "production",
        tags: dict[str, Any] | None = None,
        rollback_alias: str = "production_rollback",
    ) -> str:
        """Promove explicitamente a última versão, preservando o rollback.

        Exemplo:
            version = tracker.promote_latest_model_version("credito")
        """
        _, versions = self._registered_versions(registered_model_name)
        latest = max(int(version.version) for version in versions)
        return self.promote_model_version(
            registered_model_name, str(latest), alias, rollback_alias, tags
        )

    def promote_model_version(
        self,
        name,
        version,
        alias="production",
        rollback_alias="production_rollback",
        tags=None,
    ):
        """Promove uma versão escolhida; repetir a promoção preserva o rollback.

        Exemplo:
            tracker.promote_model_version("credito", "2")
        """
        client, _ = self._registered_versions(name)
        target = client.get_model_version(name, str(version))
        for key, value in (tags or {}).items():
            client.set_model_version_tag(name, str(version), key, str(value))
        self._move_model_alias(client, target, alias, rollback_alias)
        return str(version)

    def _move_model_alias(self, client, target, alias, rollback_alias):
        """Preserva a versão anterior somente quando o modelo ativo muda.

        Exemplo:
            tracker._move_model_alias(client, target, "prod", "previous")
        """
        if alias == rollback_alias:
            raise ValueError("Os aliases de produção e rollback devem ser distintos.")
        current = self.model_alias_version(target.name, alias)
        if current is not None and str(current) != str(target.version):
            client.set_registered_model_alias(target.name, rollback_alias, current)
        client.set_registered_model_alias(target.name, alias, str(target.version))

    def rollback_model_version(self, name):
        """Restaura o alias de rollback e preserva a versão que estava ativa.

        Exemplo:
            version = tracker.rollback_model_version("credito")
        """
        rollback = self.model_alias_version(name, "production_rollback")
        if rollback is None:
            raise ValueError(f"O modelo {name} não possui a versão de rollback.")
        return self.promote_model_version(name, rollback)

    def initialize_production_version(self, name, version):
        """Publica o primeiro baseline; preserva uma produção já definida.

        Exemplo:
            tracker.initialize_production_version("credito", "1")
        """
        current = self.model_alias_version(name)
        if current is not None:
            return str(current)
        return self.promote_model_version(name, version)

    def model_alias_version(self, name, alias="production"):
        """Consulta um alias sem criar ou alterar versões no Registry.

        Exemplo:
            current = tracker.model_alias_version("credito")
        """
        from mlflow.exceptions import MlflowException

        client = MlflowClient(tracking_uri=self.tracking_uri)
        try:
            aliases = client.get_registered_model(name).aliases
        except MlflowException as error:
            if error.error_code != "RESOURCE_DOES_NOT_EXIST":
                raise
            return None
        return str(aliases[alias]) if alias in aliases else None

    def _registered_versions(self, name):
        """Consulta versões no backend configurado por esta instância.

        Exemplo:
            client, versions = tracker._registered_versions("credito")
        """
        client = MlflowClient(tracking_uri=self.tracking_uri)
        versions = client.search_model_versions(f"name = '{name}'")
        if not versions:
            raise ValueError(f"Modelo registrado não encontrado: {name}.")
        return client, versions

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
