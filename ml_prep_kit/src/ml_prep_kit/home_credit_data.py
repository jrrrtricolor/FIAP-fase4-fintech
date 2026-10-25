"""Preparação específica do recorte Home Credit usado neste projeto."""

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series
from sklearn.model_selection import train_test_split

TARGET = "TARGET"
NUMERIC_FEATURES = [
    "AGE_YEARS",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "CNT_CHILDREN",
]
CATEGORICAL_FEATURES = [
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_HOUSING_TYPE",
]
BINARY_FEATURES = ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
SOURCE_COLUMNS = ["SK_ID_CURR", "TARGET", "DAYS_BIRTH", *FEATURES[1:]]


class HomeCreditDataV1Schema(pa.DataFrameModel):
    """Compartilha as regras das features entre treinamento e API.

    Exemplo:
        validated = HomeCreditDataV1Schema.validate(features)
    """

    AGE_YEARS: Series[float] = pa.Field(
        in_range={"min_value": 18, "max_value": 100}, coerce=True
    )
    AMT_INCOME_TOTAL: Series[float] = pa.Field(gt=0, coerce=True)
    AMT_CREDIT: Series[float] = pa.Field(gt=0, coerce=True)
    AMT_ANNUITY: Series[float] = pa.Field(gt=0, coerce=True, nullable=True)
    CNT_CHILDREN: Series[int] = pa.Field(ge=0)

    NAME_INCOME_TYPE: Series[str] = pa.Field(str_matches=r".*\S.*")
    NAME_EDUCATION_TYPE: Series[str] = pa.Field(str_matches=r".*\S.*")
    NAME_HOUSING_TYPE: Series[str] = pa.Field(str_matches=r".*\S.*")

    FLAG_OWN_CAR: Series[int] = pa.Field(isin=[0, 1])
    FLAG_OWN_REALTY: Series[int] = pa.Field(isin=[0, 1])

    class Config:
        """Rejeita colunas inesperadas no contrato compartilhado.

        Exemplo:
            assert HomeCreditDataV1Schema.Config.strict
        """

        strict = True
        name = "credit_scoring_v1"

    @pa.check("AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", name="valor_finito")
    def finite_amounts(cls, series: pd.Series) -> pd.Series:
        """Rejeita valores monetários infinitos, preservando nulos permitidos.

        Exemplo:
            valid = HomeCreditDataV1Schema.finite_amounts(pd.Series([100.0]))
        """
        return np.isfinite(series)


class HomeCreditData:
    """Reúne as colunas, conversões e regras da base, diretamente em Python.

    As estatísticas de imputação e normalização pertencem ao FeaturePreprocessor.
    Esta classe só converte unidades, valida e separa os dados.

    Exemplo:
        data = HomeCreditData(sample_size=20_000, random_state=42)
        prepared = data.prepare(data.sample(raw))
        partitions = data.split(prepared)
    """

    target = TARGET
    numeric_features = NUMERIC_FEATURES
    categorical_features = CATEGORICAL_FEATURES
    binary_features = BINARY_FEATURES
    features = FEATURES
    source_columns = SOURCE_COLUMNS

    def __init__(self, sample_size=20_000, random_state=42):
        """Configura a amostragem e o contrato de validação das colunas preparadas.

        Exemplo:
            data = HomeCreditData(sample_size=20_000, random_state=42)
        """
        self.sample_size = sample_size
        self.random_state = random_state
        self.contract = HomeCreditDataV1Schema.to_schema().add_columns(
            {
                "SK_ID_CURR": pa.Column(int, unique=True),
                "TARGET": pa.Column(int, pa.Check.isin([0, 1]), required=False),
            }
        )

    def sample(self, frame):
        """Seleciona amostra estratificada pelo alvo e ordenada por identificador.

        Exemplo:
            data = HomeCreditData(sample_size=20_000)
            sampled = data.sample(raw)
        """
        self.validate_sample_source(frame)
        selected = frame.copy()
        if len(frame) > self.sample_size:
            selected, _ = train_test_split(
                frame,
                train_size=self.sample_size,
                stratify=frame[TARGET],
                random_state=self.random_state,
            )
        return selected.sort_values("SK_ID_CURR").reset_index(drop=True)

    def validate_sample_source(self, frame):
        """Exige IDs únicos e alvo binário antes da amostragem estratificada.

        Exemplo:
            data.validate_sample_source(raw)
        """
        if frame["SK_ID_CURR"].isna().any() or frame["SK_ID_CURR"].duplicated().any():
            raise ValueError("Identificadores devem ser preenchidos e únicos.")
        if frame[TARGET].isna().any() or set(frame[TARGET].unique()) != {0, 1}:
            raise ValueError("O alvo deve conter 0 e 1, sem nulos.")
        if len(frame) < self.sample_size:
            raise ValueError("A base é menor que a amostra solicitada.")

    def prepare(self, frame):
        """Converte idade/flags e valida; não remove registros nem preenche nulos.

        Exemplo:
            data = HomeCreditData()
            prepared = data.prepare(raw)
        """
        prepared = frame.copy()
        prepared["AGE_YEARS"] = -prepared.pop("DAYS_BIRTH") / 365.25
        for name in CATEGORICAL_FEATURES:
            prepared[name] = prepared[name].str.strip().astype(object)
        for name in BINARY_FEATURES:
            prepared[name] = prepared[name].map({"Y": 1, "N": 0})
        columns = ["SK_ID_CURR", TARGET, *FEATURES]
        if TARGET not in prepared:
            columns.remove(TARGET)
        return self.contract.validate(prepared[columns], lazy=True)

    def split(self, frame):
        """Reserva 60% treino, 10% validação, 10% teste e 20% produção.

        Exemplo:
            partitions = data.split(prepared)
            train = partitions["treino"]
            test = partitions["teste"]
        """
        development, production = train_test_split(
            frame,
            test_size=0.2,
            stratify=frame[TARGET],
            random_state=self.random_state,
        )
        train, evaluation = train_test_split(
            development,
            test_size=0.25,
            stratify=development[TARGET],
            random_state=self.random_state,
        )
        validation, test = train_test_split(
            evaluation,
            test_size=0.5,
            stratify=evaluation[TARGET],
            random_state=self.random_state,
        )
        partitions = {
            "treino": train,
            "validacao": validation,
            "teste": test,
            "producao": production,
        }
        self.validate_partitions(partitions)
        return partitions

    def validate_partitions(self, partitions):
        """Valida as quatro partições e rejeita identificadores sobrepostos.

        Exemplo:
            partitions = data.split(prepared)
            data.validate_partitions(partitions)
        """
        if set(partitions) != {"treino", "validacao", "teste", "producao"}:
            raise ValueError("As quatro partições são obrigatórias.")
        for frame in partitions.values():
            self.contract.validate(frame, lazy=True)
        if pd.concat(partitions.values())["SK_ID_CURR"].duplicated().any():
            raise ValueError("Há sobreposição de IDs entre partições.")

    def prepare_version(self, frame, reference_partitions):
        """Limpa a versão já convertida e preserva as partições para o treinamento.

        Exemplo:
            partitions_v2 = data.prepare_version(version_two, experiment["partitions"])
        """
        self.validate_partitions(reference_partitions)
        prepared = self._clean_version(frame)
        partitions = self._align_version(prepared, reference_partitions)
        self.validate_partitions(partitions)
        if set(partitions["treino"][TARGET].unique()) != {0, 1}:
            raise ValueError("O treino exige exemplos das duas classes do TARGET.")
        return partitions

    def model_datasets(self, partitions):
        """Separa features e alvo para modelagem; exclui IDs e produção reservada.

        Exemplo:
            datasets = data.model_datasets(partitions)
            X_train, y_train = datasets["treino"]
        """
        self.validate_partitions(partitions)
        return {
            name: (part[FEATURES], part[TARGET])
            for name, part in partitions.items()
            if name != "producao"
        }

    def _clean_version(self, frame):
        """Limpa categorias e exige o contrato e os rótulos, sem imputar valores.

        Exemplo:
            prepared = data._clean_version(version_two)
        """
        prepared = frame.copy(deep=True)
        for name in CATEGORICAL_FEATURES:
            prepared[name] = prepared[name].str.strip().astype(object)
        prepared = self.contract.validate(prepared, lazy=True)
        if TARGET not in prepared:
            raise ValueError("O treinamento da nova versão exige TARGET.")
        return prepared

    def _align_version(self, prepared, reference_partitions):
        """Exige todos os IDs da referência e recupera sua partição e ordem.

        Exemplo:
            partitions = data._align_version(prepared, experiment["partitions"])
        """
        reference_ids = pd.concat(reference_partitions.values())["SK_ID_CURR"]
        if set(prepared["SK_ID_CURR"]) != set(reference_ids):
            raise ValueError("A nova versão deve preservar todos os IDs da referência.")
        indexed = prepared.set_index("SK_ID_CURR", drop=False)
        return {
            name: indexed.loc[original["SK_ID_CURR"]].reset_index(drop=True)
            for name, original in reference_partitions.items()
        }

    @staticmethod
    def source_hash(path):
        """Calcula o SHA-256 do arquivo para identificar a versão dos dados.

        Exemplo:
            digest = HomeCreditData.source_hash(
                "data/raw/home-credit-default-risk/application_train.csv"
            )
        """
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    def save_experiment(self, directory, partitions, metadata):
        """Guarda partições exatas e URI do modelo em um artefato local privado.

        Exemplo:
            data.save_experiment(
                "data/processed",
                partitions,
                {"model_uri": model_uri, "threshold": threshold},
            )
        """
        self.validate_partitions(partitions)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {**metadata, "partitions": partitions}, directory / "experiment.joblib"
        )
        pd.concat(partitions.values()).sort_values("SK_ID_CURR").to_csv(
            directory / "credit_scoring.csv", index=False
        )

    def load_experiment(self, directory):
        """Carrega apenas o artefato confiável gerado pelo notebook de treinamento.

        Exemplo:
            experiment = data.load_experiment("data/processed")
            partitions = experiment["partitions"]
            model_uri = experiment["model_uri"]
        """
        experiment = joblib.load(Path(directory) / "experiment.joblib")
        self.validate_partitions(experiment["partitions"])
        return experiment

    def create_drift_version(
        self, frame, income_factor=0.70, credit_factor=1.40, output_path=None
    ):
        """Cria uma cópia validada com renda e crédito alterados; salva se solicitado.

        Preserva IDs, alvo e demais colunas. Os fatores simulam Data Drift,
        sem produzir novos desfechos nem modificar o DataFrame de origem.

        Exemplo:
            version_two = data.create_drift_version(
                version_one,
                income_factor=0.70,
                credit_factor=1.40,
                output_path="data/processed/credit_scoring_v2.csv",
            )
        """
        factors = np.asarray([income_factor, credit_factor], dtype=float)
        if not np.isfinite(factors).all() or (factors <= 0).any():
            raise ValueError("Os fatores devem ser positivos e finitos.")
        shifted = self.contract.validate(frame, lazy=True).copy(deep=True)
        shifted["AMT_INCOME_TOTAL"] *= factors[0]
        shifted["AMT_CREDIT"] *= factors[1]
        shifted = self.contract.validate(shifted, lazy=True)
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            shifted.to_csv(path, index=False)
        return shifted

    def scenarios(self, production, reference):
        """Cópias pareadas: renda ×0,70, crédito ×1,40 e desfechos sintéticos.

        Exemplo:
            scenarios, metadata = data.scenarios(
                partitions["producao"], partitions["treino"]
            )
            shifted = scenarios["data_drift"]
            changed_labels = metadata["changed_labels"]
        """
        # Mantém a intervenção reproduzível mesmo após reordenar as partições.
        production = production.sort_values("SK_ID_CURR")
        data_drift = self.create_drift_version(production)
        cutoff = float(reference["AMT_INCOME_TOTAL"].quantile(0.75))
        rng = np.random.default_rng(self.random_state)
        changed = (
            production["AMT_INCOME_TOTAL"].ge(cutoff)
            & production[TARGET].eq(0)
            & (rng.random(len(production)) < 0.5)
        )
        concept = production.copy()
        concept.loc[changed, TARGET] = 1
        invalid = production.head(20).copy()
        invalid.iloc[0, invalid.columns.get_loc("AMT_INCOME_TOTAL")] = np.nan
        invalid.iloc[1, invalid.columns.get_loc("AGE_YEARS")] = -5
        invalid.iloc[2, invalid.columns.get_loc("SK_ID_CURR")] = invalid.iloc[3][
            "SK_ID_CURR"
        ]
        return {
            "dados_invalidos": invalid,
            "controle": production.copy(),
            "data_drift": data_drift,
            "concept_drift": concept,
        }, {"cutoff": cutoff, "changed_labels": int(changed.sum())}
