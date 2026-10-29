"""Fachada reutilizável para preparação e predição de modelos."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal


class _OnnxTextClassifier:
    """Adapta um classificador textual ONNX ao contrato Scikit-Learn.

    Exemplo:
        classifier = _OnnxTextClassifier(
            "model/classifier.onnx", classes=["negativo", "positivo"]
        )
        predictions = classifier.predict(["Excelente atendimento"])
    """

    def __init__(self, model_path: str | Path, classes: Sequence[str]) -> None:
        """Abre a sessão ONNX em CPU e define a ordem das classes de saída.

        Exemplo:
            classifier = _OnnxTextClassifier(
                "model/classifier.onnx", classes=["negativo", "positivo"]
            )
        """
        import numpy as np
        import onnxruntime as ort

        self._numpy = np
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) < 2:
            raise ValueError("O modelo ONNX não possui o contrato esperado.")

        self._input_name = inputs[0].name
        self._label_output = outputs[0].name
        self._probability_output = outputs[1].name
        self.classes_ = np.asarray(list(classes), dtype=object)

    def _prepare_input(self, texts: Sequence[str]) -> dict[str, Any]:
        """Converte textos em uma matriz de uma coluna para a entrada ONNX.

        Exemplo:
            inputs = classifier._prepare_input(["Excelente atendimento"])
        """
        if isinstance(texts, str):
            raise TypeError("A entrada deve ser uma coleção de textos.")
        materialized = list(texts)
        if not materialized:
            raise ValueError("A entrada deve possuir pelo menos um texto.")
        return {
            self._input_name: self._numpy.asarray(
                materialized,
                dtype=object,
            ).reshape(-1, 1)
        }

    def predict(self, texts: Sequence[str]) -> Any:
        """Retorna as classes calculadas pelo ONNX Runtime.

        Exemplo:
            labels = classifier.predict(["Excelente atendimento"])
        """
        return self._session.run(
            [self._label_output],
            self._prepare_input(texts),
        )[0]

    def predict_proba(self, texts: Sequence[str]) -> Any:
        """Retorna uma probabilidade por classe e texto.

        Exemplo:
            probabilities = classifier.predict_proba(
                ["Excelente atendimento"]
            )
        """
        probabilities = self._session.run(
            [self._probability_output],
            self._prepare_input(texts),
        )[0]
        if probabilities.shape[1] != len(self.classes_):
            raise ValueError("As classes não correspondem à saída ONNX.")
        return probabilities


class ModelPredictor:
    """Combina um modelo e um preprocessador opcional para inferência.

    A classe não conhece features, classes ou arquitetura específica. O modelo
    precisa oferecer ``predict`` e, quando scores forem necessários,
    ``predict_proba``. Um pipeline Scikit-Learn completo pode ser usado sem
    preprocessador separado.

    Exemplo:
        predictor = ModelPredictor(model=pipeline)
        classes = predictor.predict(["texto para classificação"])
    """

    def __init__(
        self,
        model: Any | None = None,
        preprocessor: Any | None = None,
    ) -> None:
        """Guarda as dependências de inferência recebidas.

        Exemplo:
            predictor = ModelPredictor(model=trained_pipeline)
        """
        self.model = model
        self.preprocessor = preprocessor

    @classmethod
    def from_mlflow(
        cls,
        model_uri: str,
        flavor: Literal["sklearn", "pyfunc"] = "sklearn",
        tracking_uri: str | None = None,
        preprocessor: Any | None = None,
    ) -> "ModelPredictor":
        """Carrega um modelo por URI do MLflow.

        ``model_uri`` aceita URIs de runs e do Model Registry, incluindo
        aliases como ``models:/example_classifier@champion``.
        Modelos PyTorch com tokenização própria devem ser registrados como
        wrapper Scikit-Learn ou modelo PyFunc para preservar o contrato de
        inferência.

        Exemplo:
            predictor = ModelPredictor.from_mlflow(
                "models:/example_classifier@champion",
                tracking_uri="sqlite:///mlflow.db",
            )
        """
        import mlflow
        import mlflow.pyfunc
        import mlflow.sklearn

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        loaders = {
            "sklearn": mlflow.sklearn.load_model,
            "pyfunc": mlflow.pyfunc.load_model,
        }
        model = loaders[flavor](model_uri)
        return cls(model=model, preprocessor=preprocessor)

    @staticmethod
    def resolve_mlflow_uri(model_uri: str, tracking_uri: str) -> tuple[str, str]:
        """Fixa um alias do Registry para carregar e informar a mesma versão.

        Exemplo:
            uri, version = ModelPredictor.resolve_mlflow_uri(uri, "sqlite:///mlflow.db")
        """
        from mlflow.tracking import MlflowClient

        reference = model_uri.removeprefix("models:/")
        registered = model_uri.startswith("models:/") and (
            "/" in reference or "@" in reference
        )
        if not registered:
            return model_uri, "unregistered"
        client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        name, version = ModelPredictor._registry_version(client, reference)
        return f"models:/{name}/{version}", str(version)

    @staticmethod
    def _registry_version(client, reference: str) -> tuple[str, str]:
        """Consulta a versão por alias ou pelo número explicitamente informado.

        Exemplo:
            name, version = ModelPredictor._registry_version(client, "credit@prod")
        """
        if "@" in reference:
            name, alias = reference.rsplit("@", 1)
            version = client.get_model_version_by_alias(name, alias).version
        else:
            name, version = reference.rsplit("/", 1)
            version = client.get_model_version(name, version).version
        return name, str(version)

    @staticmethod
    def resolve_approved_mlflow_uri(model_uri, tracking_uri):
        """Resolve apenas versões aprovadas, vinculadas ao modelo da run concluída.

        Exemplo:
            uri, version = ModelPredictor.resolve_approved_mlflow_uri(uri, tracking)
        """
        from mlflow.tracking import MlflowClient

        uri, version = ModelPredictor.resolve_mlflow_uri(model_uri, tracking_uri)
        if version == "unregistered":
            raise ValueError("A API exige uma versão aprovada no Model Registry.")
        name = uri.removeprefix("models:/").rsplit("/", 1)[0]
        client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        target = client.get_model_version(name, version)
        ModelPredictor._require_approved_version(client, target)
        return uri, version

    @staticmethod
    def _require_approved_version(client, target):
        """Confere aprovação e identidade do artefato antes de permitir sua carga.

        Exemplo:
            ModelPredictor._require_approved_version(client, version)
        """
        model_id = ModelPredictor._approved_model_id(target)
        logged = client.get_logged_model(model_id)
        run = client.get_run(logged.source_run_id)
        model_uri = f"models:/{model_id}"
        valid = run.info.status == "FINISHED" and (
            not target.run_id or target.run_id == logged.source_run_id
        )
        for tags in (target.tags, run.data.tags):
            valid = valid and tags.get("approval_status") == "approved"
            valid = valid and tags.get("approval_model_uri") == model_uri
        if not valid:
            raise ValueError("A versão não corresponde a um modelo aprovado.")

    @staticmethod
    def _approved_model_id(target):
        """Extrai a identidade da origem; o Registry SQLite não retorna model_id.

        Exemplo:
            model_id = ModelPredictor._approved_model_id(version)
        """
        source = target.source
        if not isinstance(source, str) or not source.startswith("models:/m-"):
            raise ValueError("A versão não aponta para um artefato identificado.")
        model_id = source.removeprefix("models:/")
        if model_id == "m-" or any(symbol in model_id for symbol in "/@"):
            raise ValueError("A origem do modelo não identifica um artefato único.")
        if target.model_id not in (None, model_id) or target.status != "READY":
            raise ValueError("A versão não possui um modelo pronto e consistente.")
        return model_id

    @classmethod
    def from_pkl(cls, model_file: str | Path) -> "ModelPredictor":
        """Carrega um pipeline local serializado com Pkl.

        Exemplo:
            predictor = ModelPredictor.from_pkl("model/pipeline.pkl")
        """
        import pickle

        data = None

        # Open the file in read-binary mode
        with open(model_file, "rb") as file:
            data = pickle.load(file)

        if data is None:
            raise ValueError("Failed loading pkl model")

        return cls(model=data)

    @classmethod
    def from_joblib(cls, model_path: str | Path) -> "ModelPredictor":
        """Carrega um pipeline local serializado com Joblib.

        Exemplo:
            predictor = ModelPredictor.from_joblib("model/pipeline.joblib")
        """
        import joblib

        return cls(model=joblib.load(model_path))

    @classmethod
    def from_onnx(
        cls,
        model_path: str | Path,
        classes: Sequence[str],
    ) -> "ModelPredictor":
        """Carrega um classificador textual otimizado para ONNX Runtime.

        Exemplo:
            predictor = ModelPredictor.from_onnx(
                "model/classifier.onnx", classes=["negativo", "positivo"]
            )
        """
        return cls(model=_OnnxTextClassifier(model_path, classes))

    def load_preprocessor(self, artifact_path: str | Path) -> None:
        """Carrega um preprocessador Joblib salvo localmente.

        Exemplo:
            predictor = ModelPredictor(model=trained_estimator)
            predictor.load_preprocessor("model/preprocessor.joblib")
        """
        import joblib

        self.preprocessor = joblib.load(artifact_path)

    def get_categorical_column_values(self, column_name: str) -> list[str]:
        """Consulta categorias aprendidas pelo encoder do Pipeline existente.

        Exemplo:
            values = predictor.get_categorical_column_values("NAME_INCOME_TYPE")
        """
        preprocessor = self._require_model()["preprocessor"]
        for name, transformer, columns in preprocessor.transformers_:
            if name == "cat" and column_name in columns:
                categories = transformer["encoder"].categories_[
                    list(columns).index(column_name)
                ]
                return categories.tolist()
        raise ValueError(f"Coluna categórica ausente: {column_name}")

    def prepare_input(self, X: Any) -> Any:
        """Aplica o preprocessador quando ele estiver configurado.

        Exemplo:
            predictor = ModelPredictor(model=model, preprocessor=preprocessor)
            transformed = predictor.prepare_input(X_test)
        """
        if self.preprocessor is None:
            return X

        if not hasattr(self.preprocessor, "transform"):
            raise TypeError("O preprocessador deve implementar transform().")

        return self.preprocessor.transform(X)

    def predict(self, X: Any) -> Any:
        """Prepara a entrada e retorna as classes ou valores previstos.

        Exemplo:
            predictions = predictor.predict(X_test)
        """
        model = self._require_model()
        return model.predict(self.prepare_input(X))

    def predict_proba(self, X: Any) -> Any:
        """Prepara a entrada e retorna scores por classe.

        Exemplo:
            probabilities = predictor.predict_proba(X_test)
        """
        model = self._require_model()

        if not hasattr(model, "predict_proba"):
            raise AttributeError("O modelo carregado não implementa predict_proba().")

        return model.predict_proba(self.prepare_input(X))

    def positive_scores(self, X: Any, positive_label: Any = 1) -> Any:
        """Seleciona a probabilidade pela classe, independentemente da ordem.

        Exemplo:
            scores = predictor.positive_scores(X_test, positive_label=1)
        """
        model = self._require_model()
        classes = list(model.classes_)
        if positive_label not in classes:
            raise ValueError("Classe positiva ausente no modelo.")
        return self.predict_proba(X)[:, classes.index(positive_label)]

    def assert_equivalent(self, other: "ModelPredictor", X: Any) -> None:
        """Confirma a equivalência de scores após persistência/recarga.

        Exemplo:
            reloaded = ModelPredictor.from_mlflow(model_uri)
            original = ModelPredictor(model=trained_pipeline)
            original.assert_equivalent(reloaded, X_test)
        """
        import numpy as np

        np.testing.assert_allclose(self.predict_proba(X), other.predict_proba(X))

    def _require_model(self) -> Any:
        """Retorna o modelo configurado ou interrompe a inferência.

        Exemplo:
            predictor = ModelPredictor(model=trained_pipeline)
            model = predictor._require_model()
        """
        if self.model is None:
            raise ValueError("Nenhum modelo foi configurado para predição.")

        if not hasattr(self.model, "predict"):
            raise TypeError("O modelo deve implementar predict().")

        return self.model
