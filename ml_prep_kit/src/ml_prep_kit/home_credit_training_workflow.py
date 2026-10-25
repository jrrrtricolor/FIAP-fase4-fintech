"""Preparação e treinamento compartilhados pelo notebook e pelo comando CLI."""

from pathlib import Path

from .csv_data_loader import CSVDataLoader
from .experiment_tracker import ExperimentTracker
from .feature_preprocessor import FeaturePreprocessor
from .home_credit_data import HomeCreditData
from .model_factory import ModelFactory


class HomeCreditTrainingWorkflow:
    """Reutiliza o framework para produzir um baseline rastreável.

    Exemplo:
        workflow = HomeCreditTrainingWorkflow("data/raw", "data/processed", "credito")
        workflow.split_data(workflow.load_data())
        models, records = workflow.train_models()
    """

    DEFAULT_MODEL_PARAMETERS = {
        "dummy": {"strategy": "prior"},
        "logistic_regression": {"max_iter": 2_000, "class_weight": "balanced"},
    }

    def __init__(
        self,
        raw_directory,
        processed_directory,
        experiment_name,
        sample_size=20_000,
        random_state=42,
        model_parameters=None,
    ):
        """Define origem, amostra e armazenamento SQL dos experimentos.

        Exemplo:
            workflow = HomeCreditTrainingWorkflow("raw", "processed", "credito")
        """
        self.configure_storage(raw_directory, processed_directory, experiment_name)
        self.preparation = HomeCreditData(sample_size, random_state)
        self.feature_columns = self.preparation.features
        self.model_parameters = dict(model_parameters or self.DEFAULT_MODEL_PARAMETERS)

    def configure_storage(self, raw_directory, processed_directory, experiment_name):
        """Define caminhos e o estado inicial das etapas do experimento.

        Exemplo:
            workflow.configure_storage("raw", "processed", "credito")
        """
        self.raw_directory = Path(raw_directory)
        self.processed_directory = Path(processed_directory)
        self.source_path = self.raw_directory / "application_train.csv"
        self.tracker = ExperimentTracker.from_directory(
            experiment_name,
            self.processed_directory / "monitoring",
            "training_artifacts",
        )
        self.source_hash = self.clean_df = self.partitions = self.datasets = None
        self.models, self.training_records = {}, {}

    @property
    def tracking_uri(self):
        """Retorna o backend SQL usado por todos os modelos do experimento.

        Exemplo:
            uri = workflow.tracking_uri
        """
        return self.tracker.tracking_uri

    def load_data(self):
        """Carrega, amostra e valida os dados sem ajustar transformadores.

        Exemplo:
            clean = workflow.load_data()
        """
        raw = CSVDataLoader(self.raw_directory).load(
            {"solicitacoes": self.source_path.name},
            read_csv_kwargs={"usecols": self.preparation.source_columns},
        )["solicitacoes"]
        self.clean_df = self.preparation.prepare(self.preparation.sample(raw))
        self.source_hash = self.preparation.source_hash(self.source_path)
        return self.clean_df

    def split_data(self, clean_df=None):
        """Separa treino, validação, teste e produção sem sobreposição de IDs.

        Exemplo:
            partitions, datasets = workflow.split_data(clean)
        """
        frame = self.clean_df if clean_df is None else clean_df
        if frame is None:
            raise RuntimeError("Carregue e prepare os dados antes de separá-los.")
        self.clean_df = frame
        self.partitions = self.preparation.split(frame)
        self.datasets = self.preparation.model_datasets(self.partitions)
        return self.partitions, self.datasets

    def create_model(self, name, parameters):
        """Monta um Pipeline independente para cada estimador.

        Exemplo:
            model = workflow.create_model("dummy", {"strategy": "prior"})
        """
        data = self.preparation
        preprocessor = FeaturePreprocessor(
            data.numeric_features, data.categorical_features, data.binary_features
        ).create_pipeline()
        return ModelFactory(random_state=data.random_state).create_pipeline(
            preprocessor, "classification", name, parameters
        )

    def train_models(self):
        """Ajusta os Pipelines somente no treino e registra os artefatos no MLflow.

        Exemplo:
            models, records = workflow.train_models()
        """
        if self.datasets is None:
            raise RuntimeError("Separe os dados antes de treinar os modelos.")
        for name, parameters in self.model_parameters.items():
            model = self.create_model(name, parameters)
            self.models[name] = model
            self.training_records[name] = self.train_model(name, model, parameters)
        return self.models, self.training_records

    def train_model(self, name, model, parameters):
        """Registra os parâmetros e a origem utilizados no ajuste do modelo.

        Exemplo:
            record = workflow.train_model("dummy", model, {"strategy": "prior"})
        """
        features, target = self.datasets["treino"]
        return self.tracker.train_model(
            name,
            model,
            features,
            target,
            parameters=self.training_parameters(name, parameters),
            tags={"source_sha256": self.source_hash},
        )

    def training_parameters(self, name, parameters):
        """Relaciona a configuração do estimador à amostra e à partição de treino.

        Exemplo:
            metadata = workflow.training_parameters("dummy", {"strategy": "prior"})
        """
        return {
            **parameters,
            "model": name,
            "seed": self.preparation.random_state,
            "train_rows": len(self.datasets["treino"][0]),
            "sample_size": self.preparation.sample_size,
        }

    def save_experiment(self, metadata):
        """Persiste partições e metadados para os demais consumidores.

        Exemplo:
            workflow.save_experiment({"threshold": 0.5, **record})
        """
        if self.partitions is None:
            raise RuntimeError("Separe os dados antes de salvar o experimento.")
        provenance = {
            "tracking_uri": self.tracking_uri,
            "source_sha256": self.source_hash,
            "random_state": self.preparation.random_state,
            "sample_size": self.preparation.sample_size,
            **metadata,
        }
        self.preparation.save_experiment(
            self.processed_directory, self.partitions, provenance
        )
