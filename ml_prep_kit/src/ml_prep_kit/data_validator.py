"""Utilitários de validação e resumo para dados tabulares."""

import pandas as pd


class DataValidator:
    """Gera relatórios simples para entender a qualidade dos dados.

    Use esta classe nas primeiras etapas de um projeto de dados. Ela ajuda a
    verificar volume, tipos de colunas, valores ausentes, duplicidades e
    colunas binárias. Os métodos retornam DataFrames pequenos, prontos para
    visualização em notebooks ou para logs de validação.

    Exemplo:
        validator = DataValidator()

        tables = {
            "orders": orders,
            "products": products,
        }

        summary = validator.summarize(tables)
        schema = validator.describe_schema(tables)
        missing = validator.columns_with_missing_values(orders)
    """

    def summarize(
        self,
        data: pd.DataFrame | dict[str, pd.DataFrame],
        table_name: str = "dataset",
    ) -> pd.DataFrame:
        """Retorna um resumo compacto de um ou vários DataFrames.

        Exemplo:
            summary = validator.summarize(df, "orders")
            summary = validator.summarize({"orders": orders})
        """
        if isinstance(data, pd.DataFrame):
            return self._summarize_one(data, table_name)

        return pd.concat(
            [self._summarize_one(df, name) for name, df in data.items()],
            ignore_index=True,
        ).sort_values("rows", ascending=False)

    def compare_targets(self, tables, target) -> pd.DataFrame:
        """Compara contagens e prevalência, sem atribuir causalidade.

        Exemplo:
            prevalence = validator.compare_targets(
                {"treino": train, "teste": test}, "TARGET"
            )
        """
        rows = []
        for name, frame in tables.items():
            counts = frame[target].value_counts(dropna=False).sort_index()
            rows.append(
                pd.DataFrame(
                    {
                        "dataset": name,
                        "target": counts.index,
                        "count": counts.to_numpy(),
                        "percent": counts.to_numpy() / len(frame) * 100,
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)

    def compare_categories(self, tables, columns) -> pd.DataFrame:
        """Compara contagens de categorias entre tabelas, incluindo valores ausentes.

        Exemplo:
            categories = validator.compare_categories(
                {"treino": train, "teste": test}, ["NAME_INCOME_TYPE"]
            )
        """
        rows = []
        for column in columns:
            counts = (
                pd.concat(
                    {
                        name: frame[column].value_counts(dropna=False)
                        for name, frame in tables.items()
                    },
                    axis=1,
                )
                .fillna(0)
                .astype(int)
            )
            counts.index.name = "category"
            rows.append(counts.reset_index().assign(column=column))
        return pd.concat(rows, ignore_index=True)

    def describe_numeric(self, frame, columns) -> pd.DataFrame:
        """Resume colunas numéricas com percentis de 1%, 50%, 95% e 99%.

        Exemplo:
            statistics = validator.describe_numeric(
                train, ["AMT_INCOME_TOTAL", "AMT_CREDIT"]
            )
        """
        return frame[columns].describe(percentiles=[0.01, 0.5, 0.95, 0.99]).T

    def select_dictionary(self, dictionary, columns, table_pattern) -> pd.DataFrame:
        """Seleciona descrições no formato Table/Row/Description/Special.

        Exemplo:
            descriptions = validator.select_dictionary(
                dictionary, ["TARGET", "AMT_CREDIT"], "application_train"
            )
        """
        return dictionary.loc[
            dictionary["Table"].str.contains(table_pattern, regex=False)
            & dictionary["Row"].isin(columns),
            ["Row", "Description", "Special"],
        ].rename(columns={"Row": "coluna", "Description": "descrição"})

    def _summarize_one(
        self,
        df: pd.DataFrame,
        table_name: str,
    ) -> pd.DataFrame:
        """Retorna o resumo de um DataFrame.

        Exemplo:
            summary = validator._summarize_one(orders, "orders")
        """
        return pd.DataFrame(
            [
                {
                    "dataset": table_name,
                    "rows": len(df),
                    "columns": df.shape[1],
                    "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
                }
            ]
        )

    def describe_schema(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Retorna tipos, nulos e cardinalidade das colunas.

        Exemplo:
            schema = validator.describe_schema({"orders": orders})
        """
        rows: list[dict[str, object]] = []

        for table_name, df in tables.items():
            for column in df.columns:
                rows.append(
                    {
                        "dataset": table_name,
                        "column": column,
                        "dtype": str(df[column].dtype),
                        "nulls": df[column].isna().sum(),
                        "null_rate": df[column].isna().mean(),
                        "unique_values": df[column].nunique(dropna=True),
                    }
                )

        return pd.DataFrame(rows)

    def columns_with_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retorna apenas as colunas que possuem valores ausentes.

        Exemplo:
            missing = validator.columns_with_missing_values(df)
        """
        report = pd.DataFrame(
            {
                "column": df.columns,
                "nulls": df.isna().sum().to_numpy(),
                "null_rate": df.isna().mean().to_numpy(),
            }
        )

        return report.loc[report["nulls"] > 0].sort_values(
            ["null_rate", "nulls"],
            ascending=False,
        )

    def duplicated_rows_by_table(
        self,
        tables: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Retorna linhas duplicadas por DataFrame.

        Exemplo:
            duplicates = validator.duplicated_rows_by_table({
                "orders": orders,
            })
        """
        return pd.DataFrame(
            [
                {
                    "dataset": table_name,
                    "duplicate_rows": df.duplicated().sum(),
                    "duplicate_rate": df.duplicated().mean(),
                }
                for table_name, df in tables.items()
            ]
        )

    def validate_binary_column(
        self,
        df: pd.DataFrame,
        column: str,
    ) -> dict[str, object]:
        """Verifica se uma coluna contém apenas 0 e 1, ignorando nulos.

        Exemplo:
            report = validator.validate_binary_column(df, "target")
        """
        valid_values = df[column].dropna().isin([0, 1])

        return {
            "column": column,
            "is_binary": bool(valid_values.all()),
            "invalid_rows": int((~valid_values).sum()),
        }
