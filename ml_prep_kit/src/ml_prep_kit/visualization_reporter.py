"""Utilitários simples para relatórios visuais em notebooks."""

import pandas as pd


class VisualizationReporter:
    """Cria tabelas e gráficos simples para análise gerencial.

    Use esta classe em notebooks quando precisar apresentar KPIs, rankings,
    comparações entre valores atuais e esperados, ou evolução ao longo do
    tempo. Os métodos usam pandas e matplotlib.

    Exemplo:
        reporter = VisualizationReporter()

        kpis = reporter.create_kpi_table([
            {
                "label": "Usuários",
                "value": "3.000",
                "note": "Amostra analisada.",
            },
        ])

        reporter.plot_comparison_bar(
            df=projection,
            label_column="scenario",
            current_column="current_value",
            expected_column="expected_value",
            title="Atual vs esperado",
            xlabel="Valor estimado",
        )
    """

    def create_kpi_table(
        self,
        cards: list[dict[str, str]],
    ) -> pd.DataFrame:
        """Retorna os KPIs em formato de tabela.

        Exemplo:
            kpis = reporter.create_kpi_table([
                {
                    "label": "Pedidos",
                    "value": "10.000",
                    "note": "Pedidos históricos.",
                },
            ])
        """
        return pd.DataFrame(cards)

    def plot_comparison_bar(
        self,
        df: pd.DataFrame,
        label_column: str,
        current_column: str,
        expected_column: str,
        title: str,
        xlabel: str,
    ) -> None:
        """Compara valor atual e esperado em barras horizontais.

        Exemplo:
            reporter.plot_comparison_bar(
                df=projection,
                label_column="department",
                current_column="current_value",
                expected_column="expected_value",
                title="Consumo atual vs esperado",
                xlabel="Valor estimado",
            )
        """
        chart = df.sort_values(current_column, ascending=True)
        plt = self._get_pyplot()

        ax = chart.plot.barh(
            x=label_column,
            y=[current_column, expected_column],
            figsize=(10, max(4, len(chart) * 0.45)),
            color=["#64748b", "#16a34a"],
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("")
        ax.legend(["Atual", "Esperado"])
        plt.tight_layout()
        self._show_or_close(plt)

    def plot_ranking_bar(
        self,
        df: pd.DataFrame,
        label_column: str,
        value_column: str,
        title: str,
        xlabel: str,
    ) -> None:
        """Exibe um ranking simples em barras horizontais.

        Exemplo:
            reporter.plot_ranking_bar(
                df=class_metrics,
                label_column="class_name",
                value_column="f1",
                title="F1 por classe",
                xlabel="F1",
            )
        """
        chart = df.sort_values(value_column, ascending=True)
        plt = self._get_pyplot()

        ax = chart.plot.barh(
            x=label_column,
            y=value_column,
            figsize=(10, max(4, len(chart) * 0.45)),
            color="#2563eb",
            legend=False,
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("")
        plt.tight_layout()
        self._show_or_close(plt)

    def plot_timeline(
        self,
        df: pd.DataFrame,
        x_column: str,
        current_column: str,
        expected_column: str,
        title: str,
        ylabel: str,
    ) -> None:
        """Compara valor atual e esperado ao longo do tempo.

        Exemplo:
            reporter.plot_timeline(
                df=timeline,
                x_column="period",
                current_column="current_value",
                expected_column="expected_value",
                title="Evolução do consumo",
                ylabel="Valor estimado",
            )
        """
        plt = self._get_pyplot()

        ax = df.plot(
            x=x_column,
            y=[current_column, expected_column],
            marker="o",
            figsize=(10, 4),
            color=["#64748b", "#16a34a"],
        )

        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.legend(["Atual", "Esperado"])
        plt.tight_layout()
        self._show_or_close(plt)

    def plot_heatmap(
        self,
        df: pd.DataFrame,
        row_column: str,
        column_column: str,
        value_column: str,
        title: str,
        colorbar_label: str,
    ) -> None:
        """Exibe uma matriz de intensidade entre duas dimensões.

        Exemplo:
            reporter.plot_heatmap(
                df=associations,
                row_column="source",
                column_column="target",
                value_column="orders",
                title="Categorias compradas juntas",
                colorbar_label="Pedidos",
            )
        """
        matrix = df.pivot(
            index=row_column,
            columns=column_column,
            values=value_column,
        ).fillna(0)

        plt = self._get_pyplot()
        fig, ax = plt.subplots(
            figsize=(
                max(8, len(matrix.columns) * 0.8),
                max(5, len(matrix.index) * 0.6),
            )
        )
        image = ax.imshow(matrix.to_numpy(), cmap="Blues")

        ax.set_title(title)
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_yticks(range(len(matrix.index)))
        ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
        ax.set_yticklabels(matrix.index)

        for row_index, row_name in enumerate(matrix.index):
            for column_index, column_name in enumerate(matrix.columns):
                value = matrix.loc[row_name, column_name]
                if value <= 0:
                    continue

                ax.text(
                    column_index,
                    row_index,
                    f"{value:,.0f}",
                    ha="center",
                    va="center",
                    color="#111827",
                    fontsize=8,
                )

        fig.colorbar(image, ax=ax, label=colorbar_label)
        plt.tight_layout()
        self._show_or_close(plt)

    def _get_pyplot(self):
        """Importa matplotlib somente quando um gráfico é necessário.

        Exemplo:
            plt = reporter._get_pyplot()
            figure, axes = plt.subplots()
        """
        from matplotlib import pyplot as plt

        return plt

    def _show_or_close(self, plt) -> None:
        """Exibe o gráfico no notebook ou fecha em backend não interativo.

        Exemplo:
            plt = reporter._get_pyplot()
            plt.plot([0, 1], [0.2, 0.8])
            reporter._show_or_close(plt)
        """
        if plt.get_backend().lower() == "agg":
            plt.close()
            return

        plt.show()

    def plot_target_counts(self, target):
        """Mostra o desbalanceamento do alvo.

        Exemplo:
            figure = reporter.plot_target_counts(train["TARGET"])
        """
        plt = self._get_pyplot()
        fig, ax = plt.subplots(figsize=(6, 4))
        target.value_counts().sort_index().plot.bar(ax=ax, rot=0)
        ax.set(title="Distribuição do alvo", xlabel="Classe", ylabel="Solicitações")
        fig.tight_layout()
        self._show_or_close(plt)
        return fig

    def plot_histograms(self, frame, columns, log_columns=()):
        """Distribuições; escala log muda apenas a visualização.

        Exemplo:
            figure = reporter.plot_histograms(
                train, columns=["AGE_YEARS", "AMT_INCOME_TOTAL"],
                log_columns=["AMT_INCOME_TOTAL"],
            )
        """
        import seaborn as sns

        plt = self._get_pyplot()
        fig, axes = plt.subplots(
            1, len(columns), figsize=(5 * len(columns), 4), squeeze=False
        )
        for ax, column in zip(axes[0], columns, strict=True):
            sns.histplot(
                frame, x=column, bins=30, log_scale=column in log_columns, ax=ax
            )
            ax.set(title=column, ylabel="Solicitações")
        fig.tight_layout()
        self._show_or_close(plt)
        return fig

    def plot_thresholds(self, table, threshold):
        """Compara métricas na validação e destaca o limiar escolhido.

        Exemplo:
            threshold, table = evaluator.select_threshold(
                y_validation, validation_scores
            )
            figure = reporter.plot_thresholds(table, threshold)
        """
        plt = self._get_pyplot()
        fig, ax = plt.subplots(figsize=(9, 4))
        table.sort_values("threshold").plot(
            x="threshold", y=["precision", "recall", "f1", "specificity"], ax=ax
        )
        ax.axvline(threshold, color="black", linestyle="--", label="Limiar escolhido")
        ax.set(
            title="Sensibilidade ao limiar na validação",
            xlabel="Limiar",
            ylabel="Métrica",
            ylim=(0, 1),
        )
        ax.legend()
        fig.tight_layout()
        self._show_or_close(plt)
        return fig

    def plot_metrics(self, summary, columns):
        """Compara métricas entre cenários em um gráfico de barras.

        Exemplo:
            figure = reporter.plot_metrics(
                summary, columns=["roc_auc", "recall", "drift_share"]
            )
        """
        plt = self._get_pyplot()
        ax = summary[columns].plot.bar(figsize=(10, 4), rot=0)
        ax.set(title="Comparação dos cenários", ylabel="Métrica", ylim=(0, 1))
        ax.figure.tight_layout()
        self._show_or_close(plt)
        return ax.figure

    def plot_confusion_matrices(self, reports, split="teste"):
        """Visualização das saídas do ModelEvaluator, sem recalcular métricas.

        Exemplo:
            figure = reporter.plot_confusion_matrices(
                evaluation["reports"], split="teste"
            )
        """
        import numpy as np
        from sklearn.metrics import ConfusionMatrixDisplay

        plt = self._get_pyplot()
        evaluation_reports = reports
        models = reports
        fig, axes = plt.subplots(
            len(models), 2, figsize=(11, 4 * len(models)), squeeze=False
        )
        for row, name in enumerate(models):
            matrix = evaluation_reports[name][split]["confusion_matrix"].to_numpy()
            normalized = np.divide(
                matrix,
                matrix.sum(axis=1, keepdims=True),
                out=np.zeros_like(matrix, dtype=float),
                where=matrix.sum(axis=1, keepdims=True) != 0,
            )
            for column, (values, title) in enumerate(
                [(matrix, "Contagens"), (normalized, "Proporção por classe real")]
            ):
                ConfusionMatrixDisplay(values, display_labels=[0, 1]).plot(
                    ax=axes[row, column],
                    colorbar=False,
                    cmap="Blues",
                    values_format="d" if column == 0 else ".2f",
                )
                axes[row, column].set(
                    title=f"{name} — {title}",
                    xlabel="Classe prevista",
                    ylabel="Classe real",
                )
        fig.tight_layout()
        self._show_or_close(plt)
        return fig

    def plot_roc_precision_recall(self, reports, target, split="teste"):
        """Visualização das saídas do ModelEvaluator, sem recalcular métricas.

        Exemplo:
            figure = reporter.plot_roc_precision_recall(
                evaluation["reports"], y_test, split="teste"
            )
        """
        plt = self._get_pyplot()
        evaluation_reports = reports
        models = reports
        y_test = target
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for name in models:
            report = evaluation_reports[name][split]
            metrics = report["metrics"]
            roc = report["roc_curve"]
            pr = report["precision_recall_curve"]
            axes[0].plot(
                roc["false_positive_rate"],
                roc["true_positive_rate"],
                label=f"{name}: AUC={metrics['roc_auc']:.3f}",
            )
            axes[1].step(
                pr["recall"],
                pr["precision"],
                where="post",
                label=f"{name}: AP={metrics['average_precision']:.3f}",
            )
        axes[0].plot([0, 1], [0, 1], "k--", label="Ordenação aleatória")
        axes[1].axhline(
            float(y_test.mean()),
            color="black",
            linestyle="--",
            label="Prevalência do alvo",
        )
        axes[0].set(
            title="ROC no teste", xlabel="Taxa de falsos positivos", ylabel="Recall"
        )
        axes[1].set(
            title="Precisão recall no teste", xlabel="Recall", ylabel="Precisão"
        )
        for ax in axes:
            ax.set(xlim=(0, 1), ylim=(0, 1.02))
            ax.legend(fontsize=8)
        fig.tight_layout()
        self._show_or_close(plt)
        return fig

    def plot_calibration_scores(
        self, evaluation, target, model_name, threshold, split="teste"
    ):
        """Visualização das saídas do ModelEvaluator, sem recalcular métricas.

        Exemplo:
            figure = reporter.plot_calibration_scores(
                evaluation, y_test, model_name="logistic_regression",
                threshold=selected_threshold, split="teste",
            )
        """
        import numpy as np

        plt = self._get_pyplot()
        evaluation_reports = evaluation["reports"]
        evaluation_scores = evaluation["scores"]
        models = evaluation_reports
        y_test = target
        selected_threshold = threshold
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for name in models:
            calibration = evaluation_reports[name][split]["calibration"]
            axes[0].plot(
                calibration["mean_probability"],
                calibration["observed_positive_rate"],
                marker="o",
                label=name,
            )
        axes[0].plot([0, 1], [0, 1], "k--", label="Calibração ideal")
        axes[0].set(
            title="Calibração no teste",
            xlabel="Probabilidade média",
            ylabel="Frequência positiva observada",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axes[0].legend(fontsize=8)
        scores = evaluation_scores[model_name][split]
        for label in [0, 1]:
            axes[1].hist(
                scores[y_test.to_numpy() == label],
                bins=np.linspace(0, 1, 21),
                density=True,
                alpha=0.55,
                label=f"Classe real {label}",
            )
        axes[1].axvline(
            selected_threshold, color="black", linestyle="--", label="Limiar"
        )
        axes[1].set(
            title="Scores da regressão por classe real",
            xlabel="Score",
            ylabel="Densidade",
        )
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        self._show_or_close(plt)
        return fig
