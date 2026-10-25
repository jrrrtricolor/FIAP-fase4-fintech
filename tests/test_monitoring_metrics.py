"""Verifica a exportação das evidências do framework para o Prometheus."""

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry, generate_latest

from credit_monitoring.api import create_app
from credit_monitoring.prometheus.metrics import (
    MONITORING_COLLECTOR,
    MonitoringCollector,
)

VALID_METRICS = {
    "quality_passed": 1,
    "drift_share": 0.2,
    "drifted_features": 2,
    "drift_alert": 1,
    "performance_alert": 0,
    "roc_auc": 0.62,
}


@pytest.fixture
def registry(tmp_path):
    """Isola a coleta dos artefatos e das métricas globais da API.

    Exemplo:
        def test_metrics(registry): assert registry is not None
    """
    metrics = CollectorRegistry()
    metrics.register(MonitoringCollector(tmp_path))
    return metrics


def write_summary(root: Path, metrics: dict, execution="monitoring/first", timestamp=1):
    """Grava um resultado no formato usado pelo DriftMonitor.

    Exemplo:
        path = write_summary(tmp_path, {"quality_passed": 1})
    """
    path = root / execution / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"data_drift": metrics}), encoding="utf-8")
    os.utime(path, (timestamp, timestamp))
    return path


def test_missing_summary_is_not_reported_as_no_drift(registry):
    """Ausência de avaliação não significa ausência de drift.

    Exemplo:
        pytest tests/test_monitoring_metrics.py -k missing
    """
    assert registry.get_sample_value("credit_monitoring_summary_available") == 0
    assert registry.get_sample_value("credit_monitoring_summary_read_error") == 0
    assert registry.get_sample_value("credit_monitoring_drift_alert") is None
    assert (
        registry.get_sample_value("credit_monitoring_last_evaluation_timestamp_seconds")
        is None
    )


def test_exports_framework_metrics_and_timestamp(tmp_path, registry):
    """Exporta valores calculados pelo framework com o cenário correspondente.

    Exemplo:
        pytest tests/test_monitoring_metrics.py -k exports
    """
    write_summary(tmp_path, VALID_METRICS, timestamp=100)
    assert registry.get_sample_value("credit_monitoring_summary_available") == 1
    for name, expected in VALID_METRICS.items():
        assert (
            registry.get_sample_value(
                f"credit_monitoring_{name}", {"scenario": "data_drift"}
            )
            == expected
        )
    assert (
        registry.get_sample_value("credit_monitoring_last_evaluation_timestamp_seconds")
        == 100
    )


def test_new_workflow_result_replaces_notebook_result(tmp_path, registry):
    """Cada coleta lê a avaliação mais recente sem reiniciar a API.

    Exemplo:
        pytest tests/test_monitoring_metrics.py -k replaces
    """
    write_summary(tmp_path, {"quality_passed": 1, "drift_alert": 0})
    metric = "credit_monitoring_drift_alert"
    labels = {"scenario": "data_drift"}
    assert registry.get_sample_value(metric, labels) == 0
    write_summary(
        tmp_path,
        {"quality_passed": 1, "drift_alert": 1},
        "credit_monitoring/new/drift",
        timestamp=2,
    )
    assert registry.get_sample_value(metric, labels) == 1


@pytest.mark.parametrize(
    "content",
    [
        "{",
        "[]",
        "{}",
        '{"lote": {"quality_passed": null}}',
        '{"lote": {"quality_passed": 1, "drift_share": "invalid"}}',
        '{"lote": {"quality_passed": 1, "roc_auc": NaN}}',
    ],
)
def test_invalid_latest_report_never_falls_back(tmp_path, registry, content):
    """Relatório inválido sinaliza falha e não reutiliza uma avaliação antiga.

    Exemplo:
        pytest tests/test_monitoring_metrics.py -k invalid
    """
    write_summary(tmp_path, {"quality_passed": 1, "drift_alert": 0})
    path = write_summary(tmp_path, {}, "credit_monitoring/new/drift", timestamp=2)
    path.write_text(content, encoding="utf-8")
    assert registry.get_sample_value("credit_monitoring_summary_available") == 0
    assert registry.get_sample_value("credit_monitoring_summary_read_error") == 1
    assert (
        registry.get_sample_value(
            "credit_monitoring_drift_alert", {"scenario": "data_drift"}
        )
        is None
    )


def test_unlabeled_batch_omits_performance_metrics(tmp_path, registry):
    """Lote sem TARGET não publica desempenho zero nem alerta artificial.

    Exemplo:
        pytest tests/test_monitoring_metrics.py -k unlabeled
    """
    values = {**VALID_METRICS, "roc_auc": None, "performance_alert": None}
    write_summary(tmp_path, values)
    labels = {"scenario": "data_drift"}
    assert registry.get_sample_value("credit_monitoring_drift_alert", labels) == 1
    assert registry.get_sample_value("credit_monitoring_roc_auc", labels) is None
    assert (
        registry.get_sample_value("credit_monitoring_performance_alert", labels) is None
    )


def test_rejected_batch_exports_only_available_evidence(tmp_path, registry):
    """Rejeição contratual não produz evidência estatística inexistente.

    Exemplo:
        pytest tests/test_monitoring_metrics.py -k rejected
    """
    write_summary(tmp_path, {"quality_passed": 0, "drift_alert": None})
    text = generate_latest(registry).decode()
    assert 'credit_monitoring_quality_passed{scenario="data_drift"} 0.0' in text
    assert 'credit_monitoring_drift_alert{scenario="data_drift"}' not in text


def test_api_scrape_reads_new_evidence_without_restart(tmp_path, monkeypatch):
    """O endpoint disponibiliza o resultado gravado após iniciar a API.

    Exemplo:
        pytest tests/test_monitoring_metrics.py -k api_scrape
    """
    monkeypatch.setattr(MONITORING_COLLECTOR, "processed_dir", tmp_path)
    with TestClient(create_app(model_loader=None)) as client:
        assert "credit_monitoring_summary_available 0.0" in client.get("/metrics").text
        write_summary(tmp_path, VALID_METRICS)
        response = client.get("/metrics")
        assert response.status_code == 200
        assert (
            'credit_monitoring_drift_alert{scenario="data_drift"} 1.0' in response.text
        )
