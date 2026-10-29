"""Confere snapshots operacionais e duração das etapas no MLflow real."""

import json
from functools import partial
from pathlib import Path
from unittest.mock import Mock

import pytest
from mlflow.tracking import MlflowClient

from ml_prep_kit import ExperimentTracker, ModelEvaluator


@pytest.fixture
def observed_workflow(tmp_path):
    """Disponibiliza um workflow com SQLite e artefatos isolados.

    Exemplo:
        def test_workflow(observed_workflow): ...
    """
    tracker = ExperimentTracker.from_directory("observability", tmp_path / "tracking")
    result = {"status": "iniciado", "output_dir": str(tmp_path / "output")}
    return tracker, result


def read_snapshot(result):
    """Lê o mesmo arquivo consultado pelo coletor de métricas.

    Exemplo:
        snapshot = read_snapshot(result)
    """
    return json.loads((Path(result["output_dir"]) / "result.json").read_text())


def capture_snapshot(result, snapshots):
    """Observa o estado persistido durante a execução da ação.

    Exemplo:
        value = capture_snapshot(result, snapshots)
    """
    snapshots.append(read_snapshot(result))
    return "carregado"


def interrupt_report(report, path):
    """Simula interrupção enquanto o novo JSON ainda está incompleto.

    Exemplo:
        interrupt_report(result, temporary_path)
    """
    Path(path).write_text('{"status":')
    raise OSError("interrupted write")


def test_snapshot_exists_during_action_and_finishes(observed_workflow):
    """O coletor enxerga o início da etapa e o encerramento do workflow.

    Exemplo:
        pytest tests/test_workflow_observability.py -k during_action
    """
    tracker, result = observed_workflow
    snapshots = []
    with tracker.workflow("observed", result):
        initial = read_snapshot(result)
        value = tracker.step(
            "carga", partial(capture_snapshot, result, snapshots), result
        )
        finished_step = read_snapshot(result)
        result["status"] = "sem_drift"
    final = read_snapshot(result)
    assert value == "carregado"
    assert_snapshot_sequence(initial, snapshots[0], finished_step, final)


def assert_snapshot_sequence(initial, during, after, final):
    """Valida que os estados publicados acompanham a execução observada.

    Exemplo:
        assert_snapshot_sequence(initial, during, after, final)
    """
    assert initial["workflow_running"] and during["workflow_running"]
    assert during["stage"] == "carga"
    assert "carga" not in during["step_durations_seconds"]
    assert after["step_durations_seconds"]["carga"] >= 0
    assert final["status"] == "sem_drift" and not final["workflow_running"]
    started = initial["workflow_updated_timestamp_seconds"]
    assert 0 < started <= final["workflow_updated_timestamp_seconds"]


def test_failure_before_summary_is_observable(observed_workflow):
    """Uma falha precoce persiste etapa, duração e falha no MLflow.

    Exemplo:
        pytest tests/test_workflow_observability.py -k before_summary
    """
    tracker, result = observed_workflow
    with pytest.raises(ValueError, match="invalid contract"):
        with tracker.workflow("observed", result):
            tracker.step(
                "contrato", Mock(side_effect=ValueError("invalid contract")), result
            )
    assert_failed_workflow(result)


def assert_failed_workflow(result):
    """Confere evidência de falha publicada antes de qualquer relatório de drift.

    Exemplo:
        assert_failed_workflow(result)
    """
    final = read_snapshot(result)
    run = MlflowClient().get_run(result["workflow_run_id"])
    assert final["status"] == "falha" and final["failed_stage"] == "contrato"
    assert final["error_type"] == "ValueError" and not final["workflow_running"]
    assert not (Path(result["output_dir"]) / "drift/summary.json").exists()
    duration = final["step_durations_seconds"]["contrato"]
    assert duration >= 0
    assert run.info.status == "FAILED" and run.data.tags["status"] == "falha"
    assert run.data.metrics["step_contrato_duration_seconds"] == duration


@pytest.mark.parametrize("failed", [False, True])
def test_nested_step_duration_reaches_parent(observed_workflow, failed):
    """O pai consolida a duração registrada no treino filho, inclusive se falhar.

    Exemplo:
        pytest tests/test_workflow_observability.py -k reaches_parent
    """
    tracker, result = observed_workflow
    action = Mock(side_effect=RuntimeError("fit failed") if failed else None)
    execute_nested_workflow(tracker, result, action, failed)
    parent = MlflowClient().get_run(result["workflow_run_id"])
    child = MlflowClient().get_run(result["training_run_id"])
    metric = "step_treinamento_duration_seconds"
    duration = result["step_durations_seconds"]["treinamento"]
    assert parent.data.metrics[metric] == child.data.metrics[metric] == duration
    assert child.data.tags["mlflow.parentRunId"] == parent.info.run_id
    assert (
        parent.info.status == child.info.status == ("FAILED" if failed else "FINISHED")
    )
    assert not read_snapshot(result)["workflow_running"]


def execute_nested_workflow(tracker, result, action, failed):
    """Executa uma etapa dentro do run candidato e preserva o seu erro esperado.

    Exemplo:
        execute_nested_workflow(tracker, result, action, failed=False)
    """
    try:
        with tracker.workflow("observed", result):
            with tracker.run("candidate") as child:
                result["training_run_id"] = child.info.run_id
                tracker.step("treinamento", action, result)
            result["status"] = "treinado"
    except RuntimeError as error:
        assert failed and str(error) == "fit failed"
    else:
        assert not failed


def test_interrupted_snapshot_preserves_last_complete_json(
    observed_workflow, monkeypatch
):
    """Uma escrita incompleta não substitui o JSON já disponível ao coletor.

    Exemplo:
        pytest tests/test_workflow_observability.py -k interrupted_snapshot
    """
    tracker, result = observed_workflow
    with tracker.workflow("observed", result):
        previous = read_snapshot(result)
        with monkeypatch.context() as patch:
            patch.setattr(ModelEvaluator, "save_report", interrupt_report)
            with pytest.raises(OSError, match="interrupted write"):
                tracker._save_workflow_snapshot(result)
        assert read_snapshot(result) == previous
        result["status"] = "sem_drift"
    assert read_snapshot(result)["status"] == "sem_drift"
    assert not (Path(result["output_dir"]) / "result.json.tmp").exists()
