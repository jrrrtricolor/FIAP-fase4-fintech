"""Preparação específica do recorte Home Credit usado neste projeto."""

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pandera.pandas as pa
from sklearn.model_selection import train_test_split


class HomeCreditData:
    """Reúne as colunas, conversões e regras da base, diretamente em Python.

    As estatísticas de imputação e normalização pertencem ao FeaturePreprocessor.
    Esta classe só converte unidades, valida e separa os dados.

    Exemplo:
        data = HomeCreditData(sample_size=20_000, random_state=42)
        prepared = data.prepare(data.sample(raw))
        partitions = data.split(prepared)
    """

    target = "TARGET"
    numeric_features = [
        "AGE_YEARS",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "CNT_CHILDREN",
    ]
    categorical_features = [
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_HOUSING_TYPE",
    ]
    binary_features = ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]
    features = numeric_features + categorical_features + binary_features
    source_columns = ["SK_ID_CURR", "TARGET", "DAYS_BIRTH", *features[1:]]

    def __init__(self, sample_size=20_000, random_state=42):
        """Configura a amostragem e o contrato de validação das colunas preparadas.

        Exemplo:
            data = HomeCreditData(sample_size=20_000, random_state=42)
        """
        self.sample_size = sample_size
        self.random_state = random_state
        positive = [pa.Check.gt(0), pa.Check(np.isfinite, name="valor_finito")]
        columns = {
            "SK_ID_CURR": pa.Column(int, unique=True),
            "TARGET": pa.Column(int, pa.Check.isin([0, 1]), required=False),
            "AGE_YEARS": pa.Column(float, pa.Check.in_range(18, 100)),
            "AMT_INCOME_TOTAL": pa.Column(float, positive, coerce=True),
            "AMT_CREDIT": pa.Column(float, positive, coerce=True),
            "AMT_ANNUITY": pa.Column(float, positive, nullable=True, coerce=True),
            "CNT_CHILDREN": pa.Column(int, pa.Check.ge(0)),
        }
        for name in self.categorical_features:
            columns[name] = pa.Column(str, pa.Check.str_matches(r".*\S.*"))
        for name in self.binary_features:
            columns[name] = pa.Column(int, pa.Check.isin([0, 1]))
        self.contract = pa.DataFrameSchema(
            columns, strict=True, name="credit_scoring_v1"
        )

    def sample(self, frame):
        """Seleciona amostra estratificada pelo alvo e ordenada por identificador.

        Exemplo:
            data = HomeCreditData(sample_size=20_000)
            sampled = data.sample(raw)
        """
        if frame["SK_ID_CURR"].isna().any() or frame["SK_ID_CURR"].duplicated().any():
            raise ValueError("Identificadores devem ser preenchidos e únicos.")
        if frame[self.target].isna().any() or set(frame[self.target].unique()) != {
            0,
            1,
        }:
            raise ValueError("O alvo deve conter 0 e 1, sem nulos.")
        if len(frame) < self.sample_size:
            raise ValueError("A base é menor que a amostra solicitada.")
        selected = frame.copy()
        if len(frame) > self.sample_size:
            selected, _ = train_test_split(
                frame,
                train_size=self.sample_size,
                stratify=frame[self.target],
                random_state=self.random_state,
            )
        return selected.sort_values("SK_ID_CURR").reset_index(drop=True)

    def prepare(self, frame):
        """Converte idade/flags e valida; não remove registros nem preenche nulos.

        Exemplo:
            data = HomeCreditData()
            prepared = data.prepare(raw)
        """
        prepared = frame.copy()
        prepared["AGE_YEARS"] = -prepared.pop("DAYS_BIRTH") / 365.25
        for name in self.categorical_features:
            prepared[name] = prepared[name].str.strip().astype(object)
        for name in self.binary_features:
            prepared[name] = prepared[name].map({"Y": 1, "N": 0})
        columns = ["SK_ID_CURR", self.target, *self.features]
        if self.target not in prepared:
            columns.remove(self.target)
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
            stratify=frame[self.target],
            random_state=self.random_state,
        )
        train, evaluation = train_test_split(
            development,
            test_size=0.25,
            stratify=development[self.target],
            random_state=self.random_state,
        )
        validation, test = train_test_split(
            evaluation,
            test_size=0.5,
            stratify=evaluation[self.target],
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
        if set(partitions["treino"][self.target].unique()) != {0, 1}:
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
            name: (part[self.features], part[self.target])
            for name, part in partitions.items()
            if name != "producao"
        }

    def _clean_version(self, frame):
        """Limpa categorias e exige o contrato e os rótulos, sem imputar valores.

        Exemplo:
            prepared = data._clean_version(version_two)
        """
        prepared = frame.copy(deep=True)
        for name in self.categorical_features:
            prepared[name] = prepared[name].str.strip().astype(object)
        prepared = self.contract.validate(prepared, lazy=True)
        if self.target not in prepared:
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
            & production[self.target].eq(0)
            & (rng.random(len(production)) < 0.5)
        )
        concept = production.copy()
        concept.loc[changed, self.target] = 1
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
