"""Métricas da API e resultados persistidos pelo DriftMonitor."""

import json
import math
import os
from collections.abc import Iterator
from pathlib import Path

from prometheus_client import REGISTRY, Counter, Gauge, Histogram, Summary
from prometheus_client.core import GaugeMetricFamily

PREDICTIONS_TOTAL = Counter(
    "total_de_predicoes",
    "Total de predições realizadas",
)
ERRORS_TOTAL = Counter(
    "total_de_erros_na_api",
    "Total de respostas HTTP com erro e exceções não tratadas",
)
PREDICTION_DURATION = Histogram(
    "duracao_da_predicao_segundos",
    "Duração do processamento da predição em segundos",
)
PREDICTION_CONFIDENCE = Histogram(
    "confianca_da_predicao",
    "Maior probabilidade retornada em cada predição",
    buckets=(0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0),
)
MODEL_INFO = Gauge(
    "modelo_info",
    "Identifica a versão do modelo carregado pela API",
    labelnames=("version",),
)

REFERENCE_MODEL_AMT_INCOME_TOTAL = Summary(
    "referencia_de_renda",
    "Summary de valores de renda do modelo em produção",
    labelnames=["ref_amt_income"],
)
REFERENCE_MODEL_AMT_CREDIT = Summary(
    "referencia_de_credito",
    "Summary de valores de credito do modelo em produção",
    labelnames=["ref_amt_credit"],
)
REFERENCE_MODEL_AGE_YEARS = Summary(
    "referencia_de_idades_em_anos",
    "Summary de valores de idade em anos do modelo em produção",
    labelnames=["ref_age_years"],
)

DRIFT_METRICS = {
    "drift_share": "Proporção de variáveis com drift estatístico",
    "drifted_features": "Quantidade de variáveis com drift estatístico",
    "drift_alert": "Alerta de drift definido pelo contrato de monitoramento",
    "quality_passed": "Contrato dos dados aprovado: 1; rejeitado: 0",
    "performance_alert": "Alerta de queda de desempenho com rótulos disponíveis",
    "roc_auc": "ROC AUC observado no lote com rótulos disponíveis",
}
SUMMARY_METRICS = {
    "summary_available": "Sumário válido disponível",
    "summary_read_error": "Falha ao ler o último sumário",
    "last_evaluation_timestamp_seconds": "Atualização do sumário em segundos Unix",
}


def summary_modified_at(path: Path) -> float:
    """Obtém a atualização do sumário para selecionar a avaliação mais recente.

    Exemplo:
        timestamp = summary_modified_at(Path("summary.json"))
    """
    return path.stat().st_mtime


def validate_summary(summary: dict) -> None:
    """Recusa estruturas inválidas e valores não numéricos nas métricas conhecidas.

    Exemplo:
        validate_summary({"controle": {"quality_passed": 1}})
    """
    if not isinstance(summary, dict) or not summary:
        raise ValueError("Sumário sem cenários válidos.")
    for scenario, metrics in summary.items():
        if not scenario or not isinstance(metrics, dict):
            raise ValueError("Cenário inválido.")
        if metrics.get("quality_passed") not in (0, 1):
            raise ValueError("Resultado do contrato ausente ou inválido.")
        for name in DRIFT_METRICS:
            value = metrics.get(name)
            if value is not None and (
                not isinstance(value, (int, float)) or not math.isfinite(value)
            ):
                raise ValueError(f"Métrica inválida: {name}.")


class MonitoringCollector:
    """Expõe o último sumário do framework sem executar detecção no scrape.

    Exemplo:
        collector = MonitoringCollector(Path("data/processed"))
        metrics = list(collector.collect())
    """

    def __init__(self, processed_dir: Path) -> None:
        """Define a raiz compartilhada entre notebooks, workflow e API.

        Exemplo:
            collector = MonitoringCollector(Path("data/processed"))
        """
        self.processed_dir = processed_dir

    def latest_summary(self) -> Path | None:
        """Seleciona somente sumários originais, excluindo cópias dos artefatos.

        Exemplo:
            path = collector.latest_summary()
        """
        patterns = (
            "credit_monitoring/*/drift/summary.json",
            "monitoring/*/summary.json",
        )
        paths = [
            path for pattern in patterns for path in self.processed_dir.glob(pattern)
        ]
        return max(paths, key=summary_modified_at, default=None)

    def read_summary(self) -> tuple[dict, float | None, int]:
        """Distingue ausência de relatório e falha de leitura, sem usar valores antigos.

        Exemplo:
            summary, timestamp, error = collector.read_summary()
        """
        try:
            path = self.latest_summary()
            if path is None:
                return {}, None, 0
            summary = json.loads(path.read_text(encoding="utf-8"))
            validate_summary(summary)
            return summary, summary_modified_at(path), 0
        except (OSError, ValueError):
            return {}, None, 1

    def scenario_metrics(self, summary: dict) -> Iterator[GaugeMetricFamily]:
        """Omite métricas indisponíveis, inclusive desempenho sem rótulos.

        Exemplo:
            metrics = list(collector.scenario_metrics({"lote": {"quality_passed": 1}}))
        """
        for name, description in DRIFT_METRICS.items():
            metric = GaugeMetricFamily(
                f"credit_monitoring_{name}", description, labels=["scenario"]
            )
            for scenario, values in summary.items():
                value = values.get(name)
                if value is not None:
                    metric.add_metric([scenario], value)
            yield metric

    def summary_metrics(self, summary, timestamp, error) -> Iterator[GaugeMetricFamily]:
        """Publica disponibilidade e atualização do arquivo da avaliação.

        Exemplo:
            metrics = list(collector.summary_metrics({}, None, 0))
        """
        values = {
            "summary_available": bool(summary),
            "summary_read_error": error,
            "last_evaluation_timestamp_seconds": timestamp,
        }
        for name, value in values.items():
            if value is not None:
                yield GaugeMetricFamily(
                    f"credit_monitoring_{name}", SUMMARY_METRICS[name], value=value
                )

    def collect(self) -> Iterator[GaugeMetricFamily]:
        """Publica métricas do último relatório e a disponibilidade da evidência.

        Exemplo:
            metrics = list(collector.collect())
        """
        summary, timestamp, error = self.read_summary()
        yield from self.summary_metrics(summary, timestamp, error)
        yield from self.scenario_metrics(summary)


# O registro único acompanha as métricas existentes, inclusive ao criar várias APIs.
MONITORING_COLLECTOR = MonitoringCollector(
    Path(os.getenv("MONITORING_OUTPUT_DIR", "data/processed"))
)
REGISTRY.register(MONITORING_COLLECTOR)
