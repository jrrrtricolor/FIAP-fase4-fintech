"""Verifica a exportação das evidências do framework para o Prometheus."""

import json
import os
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry, generate_latest

from credit_monitoring.api import create_app
from credit_monitoring.prometheus.metrics import (
    MONITORING_COLLECTOR,
    MonitoringCollector,
)
from ml_prep_kit import DriftMonitor

VALID_METRICS = {
    "quality_passed": 1,
    "drift_share": 0.2,
    "drifted_features": 2,
    "drift_alert": 1,
    "performance_alert": 0,
    "roc_auc": 0.62,
}


def test_framework_summary_exports_drift_and_time(tmp_path, monkeypatch):
    """Conecta o relatório real do framework às métricas exportadas para o Grafana.

    Exemplo:
        pytest tests/test_monitoring_metrics.py -k framework_summary
    """
    monkeypatch.setattr("ml_prep_kit.drift_monitor.time", lambda: 123.0)
    table = pd.DataFrame(
        {"feature": ["AMT_CREDIT", "AGE_YEARS"], "drift": [True, False]}
    )
    results = {"data_drift": {"metrics": VALID_METRICS, "drift_table": table}}
    path = write_summary(tmp_path, VALID_METRICS)
    DriftMonitor.summarize(results, path)
    summary, _, error = MonitoringCollector(tmp_path).read_report("summary")
    values = summary["data_drift"]
    assert values["evaluation_timestamp_seconds"] == 123.0 and error == 0
    assert values["feature_drift__AMT_CREDIT"] == 1
    assert values["feature_drift__AGE_YEARS"] == 0


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
    assert registry.get_sample_value("credit_monitoring_workflow_read_error") == 0
    assert registry.get_sample_value("credit_monitoring_workflow_failed") is None
    assert registry.get_sample_value("credit_monitoring_workflow_running") is None
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
        '{"lote": {"quality_passed": 1, "evaluation_timestamp_seconds": 0}}',
        '{"lote": {"quality_passed": 1, "evaluation_timestamp_seconds": NaN}}',
        '{"lote": {"quality_passed": 1, "evaluation_timestamp_seconds": "recent"}}',
        '{"lote": {"quality_passed": 1, "feature_drift__AMT_CREDIT": 2}}',
        '{"lote": {"quality_passed": 1, "feature_drift__AMT_CREDIT": Infinity}}',
        '{"lote": {"quality_passed": 1, "feature_drift__": 1}}',
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
    assert registry.get_sample_value("credit_monitoring_feature_drift") is None
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


@pytest.mark.parametrize(
    "field,name,labels",
    [
        ("drift_alert", "drift_alert", 'scenario="data_drift"'),
        (
            "feature_drift__AMT_CREDIT",
            "feature_drift",
            'feature="AMT_CREDIT",scenario="data_drift"',
        ),
    ],
)
def test_api_scrape_reads_new_evidence_without_restart(
    tmp_path, monkeypatch, field, name, labels
):
    """O scrape lê a evidência nova e suas mudanças sem reiniciar a API.

    Exemplo:
        pytest tests/test_monitoring_metrics.py -k api_scrape
    """
    monkeypatch.setattr(MONITORING_COLLECTOR, "processed_dir", tmp_path)
    sample = f"credit_monitoring_{name}{{{labels}}}"
    with TestClient(create_app(model_loader=None)) as client:
        assert "credit_monitoring_summary_available 0.0" in client.get("/metrics").text
        for value in (1, 0):
            write_summary(tmp_path, {**VALID_METRICS, field: value})
            response = client.get("/metrics")
            assert response.status_code == 200
            assert f"{sample} {value}.0" in response.text


def test_persisted_evaluation_timestamp_survives_file_copy(tmp_path, registry):
    """O horário da execução não muda quando o arquivo recebe outro mtime.

    Exemplo:
        pytest tests/test_monitoring_metrics.py -k persisted_evaluation
    """
    values = {**VALID_METRICS, "evaluation_timestamp_seconds": 100}
    path = write_summary(tmp_path, values, timestamp=900)
    os.utime(path, (1000, 1000))
    metric = "credit_monitoring_last_evaluation_timestamp_seconds"
    assert registry.get_sample_value(metric) == 100


@pytest.mark.parametrize("expected", [1, 0, None])
def test_exports_affected_and_stable_features(tmp_path, registry, expected):
    """Distingue variável alterada, estável e variável sem avaliação disponível.

    Exemplo:
        pytest tests/test_monitoring_metrics.py -k affected_and_stable
    """
    values = {**VALID_METRICS, "feature_drift__AMT_CREDIT": expected}
    write_summary(tmp_path, values)
    labels = {"scenario": "data_drift", "feature": "AMT_CREDIT"}
    assert (
        registry.get_sample_value("credit_monitoring_feature_drift", labels) == expected
    )


@pytest.mark.parametrize("other_timestamp", [None, 200])
def test_inconsistent_evaluation_timestamps_are_rejected(
    tmp_path, registry, other_timestamp
):
    """Cenários do mesmo resumo não podem ter horários divergentes ou incompletos.

    Exemplo:
        pytest tests/test_monitoring_metrics.py -k inconsistent_evaluation
    """
    values = {**VALID_METRICS, "evaluation_timestamp_seconds": 100}
    path = write_summary(tmp_path, values)
    summary = {
        "controle": values,
        "data_drift": {**values, "evaluation_timestamp_seconds": other_timestamp},
    }
    path.write_text(json.dumps(summary), encoding="utf-8")
    assert registry.get_sample_value("credit_monitoring_summary_read_error") == 1


VALID_WORKFLOW = {
    "status": "falha",
    "workflow_running": False,
    "workflow_updated_timestamp_seconds": 100,
    "step_durations_seconds": {"carga": 0.4},
    "failed_stage": "carga",
}


def write_workflow(root, result, execution="new", timestamp=2):
    """Grava um resultado original com horário de arquivo controlado.

    Exemplo:
        path = write_workflow(tmp_path, VALID_WORKFLOW)
    """
    path = root / "credit_monitoring" / execution / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result), encoding="utf-8")
    os.utime(path, (timestamp, timestamp))
    return path


def test_workflow_exports_persisted_state_and_stage_duration(tmp_path, registry):
    """Expõe falha, duração e horário persistido, sem depender do mtime exibido.

    Exemplo:
        pytest tests/test_monitoring_metrics.py -k persisted_state
    """
    write_workflow(tmp_path, VALID_WORKFLOW, timestamp=900)
    expected = {
        "workflow_failed": 1,
        "workflow_running": 0,
        "workflow_updated_timestamp_seconds": 100,
    }
    for name, value in expected.items():
        assert registry.get_sample_value(f"credit_monitoring_{name}") == value
    assert (
        registry.get_sample_value(
            "credit_monitoring_stage_duration_seconds", {"stage": "carga"}
        )
        == 0.4
    )


@pytest.mark.parametrize(
    "running,stage,quality",
    [(False, "carga", None), (True, "carga", None), (False, "limpeza_e_contrato", 0)],
)
def test_workflow_without_summary(tmp_path, registry, running, stage, quality):
    """Ausência de sumário novo expõe só o estado real e eventual bloqueio contratual.

    Exemplo:
        pytest tests/test_monitoring_metrics.py -k workflow_without_summary
    """
    write_summary(tmp_path, VALID_METRICS)
    result = {**VALID_WORKFLOW, "workflow_running": running, "failed_stage": stage}
    result["status"] = "iniciado" if running else "falha"
    write_workflow(tmp_path, result)
    labels = {"scenario": "data_drift"}
    assert registry.get_sample_value("credit_monitoring_summary_available") == 0
    assert (
        registry.get_sample_value("credit_monitoring_quality_passed", labels) == quality
    )
    assert registry.get_sample_value("credit_monitoring_drift_alert", labels) is None
    assert registry.get_sample_value("credit_monitoring_drift_share", labels) is None
    assert registry.get_sample_value("credit_monitoring_feature_drift") is None
    assert registry.get_sample_value("credit_monitoring_workflow_running") == running


@pytest.mark.parametrize(
    "execution,summary_time,workflow_time",
    [("credit_monitoring/new/drift", 1, 2), ("monitoring/first", 2, 1)],
)
def test_workflow_keeps_current_summary(
    tmp_path, registry, execution, summary_time, workflow_time
):
    """Mantém o sumário da própria execução ou uma avaliação posterior à falha antiga.

    Exemplo:
        pytest tests/test_monitoring_metrics.py -k keeps_current_summary
    """
    write_summary(tmp_path, VALID_METRICS, execution, timestamp=summary_time)
    write_workflow(tmp_path, VALID_WORKFLOW, timestamp=workflow_time)
    assert registry.get_sample_value("credit_monitoring_summary_available") == 1
    assert registry.get_sample_value("credit_monitoring_workflow_failed") == 1


@pytest.mark.parametrize(
    "patch",
    [
        {"status": None},
        {"workflow_running": "yes"},
        {"workflow_updated_timestamp_seconds": None},
        {"workflow_updated_timestamp_seconds": float("nan")},
        {"step_durations_seconds": []},
        {"step_durations_seconds": {"carga": -1}},
        {"step_durations_seconds": {"carga": float("inf")}},
    ],
)
def test_invalid_workflow_hides_stale_summary(tmp_path, registry, patch):
    """Metadados inválidos expõem erro de leitura e não deixam resultado antigo verde.

    Exemplo:
        pytest tests/test_monitoring_metrics.py -k invalid_workflow
    """
    write_summary(tmp_path, VALID_METRICS)
    write_workflow(tmp_path, {**VALID_WORKFLOW, **patch})
    assert registry.get_sample_value("credit_monitoring_workflow_read_error") == 1
    assert registry.get_sample_value("credit_monitoring_summary_available") == 0
    assert registry.get_sample_value("credit_monitoring_workflow_failed") is None


def test_latest_workflow_recovery_clears_failure_and_old_stages(tmp_path, registry):
    """O novo resultado substitui a falha e remove etapas ausentes da execução atual.

    Exemplo:
        pytest tests/test_monitoring_metrics.py -k clears_failure
    """
    write_workflow(tmp_path, VALID_WORKFLOW, execution="failed")
    result = {
        **VALID_WORKFLOW,
        "status": "sem_alerta",
        "step_durations_seconds": {"deteccao_de_drift": 3},
    }
    write_workflow(tmp_path, result, execution="recovered", timestamp=3)
    write_summary(
        tmp_path, VALID_METRICS, "credit_monitoring/recovered/drift", timestamp=4
    )
    assert registry.get_sample_value("credit_monitoring_workflow_failed") == 0
    metric = "credit_monitoring_stage_duration_seconds"
    assert registry.get_sample_value(metric, {"stage": "carga"}) is None
    assert registry.get_sample_value("credit_monitoring_summary_available") == 1
