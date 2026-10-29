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

API_READY = Gauge("credit_monitoring_api_ready", "API com modelo aprovado carregado")
MODEL_RELOAD_ERROR = Gauge(
    "credit_monitoring_model_reload_error", "Última tentativa de atualização falhou"
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
FEATURE_DRIFT_PREFIX = "feature_drift__"
SUMMARY_METRICS = {
    "summary_available": "Sumário válido disponível",
    "summary_read_error": "Falha ao ler o último sumário",
    "last_evaluation_timestamp_seconds": "Horário da avaliação em segundos Unix",
}

REPORT_PATTERNS = {
    "summary": ("credit_monitoring/*/drift/summary.json", "monitoring/*/summary.json"),
    "workflow": ("credit_monitoring/*/result.json",),
}


def finite_number(value):
    """Confere se um valor pode ser publicado como métrica finita.

    Exemplo:
        valid = finite_number(0.3)
    """
    return isinstance(value, (int, float)) and math.isfinite(value)


def validate_workflow(result):
    """Valida estado, horário e durações antes de publicar o último workflow.

    Exemplo:
        validate_workflow({"status": "falha", "step_durations_seconds": {}})
    """
    if not isinstance(result, dict) or not isinstance(result.get("status"), str):
        raise ValueError("Resultado do workflow inválido.")
    running = result.get("workflow_running", False)
    if not result["status"] or not isinstance(running, bool):
        raise ValueError("Estado do workflow inválido.")
    timestamp = result.get("workflow_updated_timestamp_seconds", 1)
    if not finite_number(timestamp) or timestamp <= 0:
        raise ValueError("Horário do workflow inválido.")
    durations = result.get("step_durations_seconds", {})
    if not isinstance(durations, dict):
        raise ValueError("Durações inválidas.")
    for name, value in durations.items():
        if not name or not finite_number(value) or value < 0:
            raise ValueError("Etapa ou duração inválida.")


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
        validate_scenario_metrics(metrics)
    timestamps = {
        metrics.get("evaluation_timestamp_seconds") for metrics in summary.values()
    }
    if len(timestamps) != 1:
        raise ValueError("Os cenários devem compartilhar o horário da avaliação.")


def validate_scenario_metrics(metrics):
    """Valida valores numéricos, horário e indicadores individuais de drift.

    Exemplo:
        validate_scenario_metrics({"drift_share": 0.2})
    """
    names = set(DRIFT_METRICS) | {"evaluation_timestamp_seconds"}
    names.update(name for name in metrics if name.startswith(FEATURE_DRIFT_PREFIX))
    for name in names:
        value = metrics.get(name)
        if value is not None and not finite_number(value):
            raise ValueError(f"Métrica inválida: {name}.")
        if name.startswith(FEATURE_DRIFT_PREFIX):
            if not name.removeprefix(FEATURE_DRIFT_PREFIX) or value not in (None, 0, 1):
                raise ValueError(f"Indicador de drift inválido: {name}.")
    timestamp = metrics.get("evaluation_timestamp_seconds")
    if timestamp is not None and timestamp <= 0:
        raise ValueError("Horário de avaliação inválido.")


def evaluation_timestamp(summary, path):
    """Usa o horário persistido; relatórios legados conservam a data do arquivo.

    Exemplo:
        timestamp = evaluation_timestamp(summary, Path("summary.json"))
    """
    values = next(iter(summary.values()))
    return values.get("evaluation_timestamp_seconds") or summary_modified_at(path)


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

    def read_report(self, kind):
        """Lê somente o último relatório original, sem recuar para evidência antiga.

        Exemplo:
            summary, path, error = collector.read_report("summary")
        """
        path = None
        try:
            root = self.processed_dir
            paths = [p for mask in REPORT_PATTERNS[kind] for p in root.glob(mask)]
            path = max(paths, key=summary_modified_at, default=None)
            if path is None:
                return {}, None, 0
            report = json.loads(path.read_text(encoding="utf-8"))
            validator = validate_summary if kind == "summary" else validate_workflow
            validator(report)
            return report, path, 0
        except (OSError, ValueError):
            return {}, path, 1

    def workflow_blocks_summary(self, workflow_path, summary_path):
        """Oculta avaliação antiga quando um workflow novo ainda não gerou a sua.

        Exemplo:
            blocked = collector.workflow_blocks_summary(workflow_path, summary_path)
        """
        if workflow_path is None:
            return False
        try:
            if (workflow_path.parent / "drift" / "summary.json").exists():
                return False
            return summary_path is None or (
                summary_modified_at(workflow_path) >= summary_modified_at(summary_path)
            )
        except OSError:
            return True

    def contract_failure(self, result):
        """Publica somente o bloqueio contratual registrado, sem fabricar drift.

        Exemplo:
            evidence = collector.contract_failure({"status": "falha"})
        """
        failed = result.get("status") == "falha"
        if failed and result.get("failed_stage") == "limpeza_e_contrato":
            return {"data_drift": {"quality_passed": 0}}
        return {}

    def workflow_metrics(self, result, error):
        """Publica estado e durações da última execução; ausência não vira sucesso.

        Exemplo:
            metrics = list(collector.workflow_metrics(result, 0))
        """
        names = ("workflow_running", "workflow_updated_timestamp_seconds")
        values = {name: result.get(name) for name in names}
        values["workflow_read_error"] = error
        values["workflow_failed"] = result["status"] == "falha" if result else None
        for name, value in values.items():
            if value is not None:
                yield GaugeMetricFamily(f"credit_monitoring_{name}", name, value=value)
        name = "credit_monitoring_stage_duration_seconds"
        metric = GaugeMetricFamily(name, "Duração da etapa", labels=["stage"])
        for name, duration in result.get("step_durations_seconds", {}).items():
            metric.add_metric([name], duration)
        yield metric

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

    def feature_metrics(self, summary) -> Iterator[GaugeMetricFamily]:
        """Expõe cada variável avaliada; ausência de evidência não vira zero.

        Exemplo:
            metrics = list(collector.feature_metrics(summary))
        """
        metric = GaugeMetricFamily(
            "credit_monitoring_feature_drift",
            "Drift por variável: 1; estável: 0",
            labels=["scenario", "feature"],
        )
        for scenario, values in summary.items():
            for name, value in values.items():
                if name.startswith(FEATURE_DRIFT_PREFIX) and value is not None:
                    feature = name.removeprefix(FEATURE_DRIFT_PREFIX)
                    metric.add_metric([scenario, feature], value)
        yield metric

    def summary_metrics(self, summary, path, error) -> Iterator[GaugeMetricFamily]:
        """Publica disponibilidade e horário persistido da avaliação.

        Exemplo:
            metrics = list(collector.summary_metrics({}, None, 0))
        """
        timestamp = evaluation_timestamp(summary, path) if summary else None
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
        result, workflow_path, workflow_error = self.read_report("workflow")
        summary, path, error = self.read_report("summary")
        blocked = self.workflow_blocks_summary(workflow_path, path)
        if blocked:
            summary, path = {}, None
        yield from self.summary_metrics(summary, path, error)
        evidence = self.contract_failure(result) if blocked else summary
        yield from self.scenario_metrics(evidence)
        yield from self.feature_metrics(summary)
        yield from self.workflow_metrics(result, workflow_error)


# O registro único acompanha as métricas existentes, inclusive ao criar várias APIs.
MONITORING_COLLECTOR = MonitoringCollector(
    Path(os.getenv("MONITORING_OUTPUT_DIR", "data/processed"))
)
REGISTRY.register(MONITORING_COLLECTOR)
