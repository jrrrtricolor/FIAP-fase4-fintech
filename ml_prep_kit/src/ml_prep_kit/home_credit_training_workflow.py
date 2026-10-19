"""Fluxo reutilizável de preparação e treinamento para Home Credit."""

from pathlib import Path

from .csv_data_loader import CSVDataLoader
from .experiment_tracker import ExperimentTracker
from .feature_preprocessor import FeaturePreprocessor
from .home_credit_data import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    SOURCE_COLUMNS,
    HomeCreditData,
)
from .model_factory import ModelFactory


class HomeCreditTrainingWorkflow:
    """Compartilha a preparação e o treinamento do baseline Home Credit."""

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
        self.raw_directory = Path(raw_directory)
        self.processed_directory = Path(processed_directory)
        self.source_path = self.raw_directory / "application_train.csv"
        self.preparation = HomeCreditData(
            sample_size=sample_size,
            random_state=random_state,
        )
        self.feature_columns = FEATURES
        self.model_parameters = (
            self.DEFAULT_MODEL_PARAMETERS
            if model_parameters is None
            else model_parameters
        )
        self.tracker = ExperimentTracker.from_directory(
            experiment_name,
            self.processed_directory / "monitoring",
            "training_artifacts",
        )
        self.source_hash = None
        self.clean_df = None
        self.partitions = None
        self.datasets = None
        self.models = {}
        self.training_records = {}

    @property
    def tracking_uri(self):
        return self.tracker.tracking_uri

    def load_data(self):
        """Carrega a origem, aplica a amostra e valida os dados preparados."""
        raw_df = CSVDataLoader(self.raw_directory).load(
            {"solicitacoes": self.source_path.name},
            read_csv_kwargs={"usecols": SOURCE_COLUMNS},
        )["solicitacoes"]
        self.clean_df = self.preparation.prepare(self.preparation.sample(raw_df))
        self.source_hash = self.preparation.source_hash(self.source_path)
        return self.clean_df

    def split_data(self, clean_df=None):
        """Cria partições sem sobreposição e datasets para modelagem."""
        frame = self.clean_df if clean_df is None else clean_df
        if frame is None:
            raise RuntimeError("Carregue e prepare os dados antes de separá-los.")
        self.clean_df = frame
        self.partitions = self.preparation.split(frame)
        self.datasets = self.preparation.model_datasets(self.partitions)
        return self.partitions, self.datasets

    def train_models(self):
        """Treina e registra os modelos com o pipeline de features compartilhado."""
        if self.datasets is None:
            raise RuntimeError("Separe os dados antes de treinar os modelos.")

        factory = ModelFactory(random_state=self.preparation.random_state)
        preprocessor = FeaturePreprocessor(
            numeric_columns=NUMERIC_FEATURES,
            categorical_columns=CATEGORICAL_FEATURES,
            binary_columns=BINARY_FEATURES,
        )
        X_train, y_train = self.datasets["treino"]

        for name, parameters in self.model_parameters.items():
            model = factory.create_pipeline(
                preprocessor=preprocessor.create_pipeline(),
                problem_type="classification",
                model_name=name,
                parameters=parameters,
            )
            self.models[name] = model
            self.training_records[name] = self.tracker.train_model(
                name,
                model,
                X_train,
                y_train,
                parameters={
                    **parameters,
                    "model": name,
                    "seed": self.preparation.random_state,
                    "train_rows": len(X_train),
                    "sample_size": self.preparation.sample_size,
                },
                tags={"source_sha256": self.source_hash},
            )

        return self.models, self.training_records

    def save_experiment(self, metadata):
        """Salva partições e metadados comuns junto dos dados do experimento."""
        if self.partitions is None:
            raise RuntimeError("Separe os dados antes de salvar o experimento.")
        self.preparation.save_experiment(
            self.processed_directory,
            self.partitions,
            {
                "tracking_uri": self.tracking_uri,
                "source_sha256": self.source_hash,
                "random_state": self.preparation.random_state,
                "sample_size": self.preparation.sample_size,
                **metadata,
            },
        )