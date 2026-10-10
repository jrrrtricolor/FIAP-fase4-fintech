"""Validação, predição e comparação estatística de um lote de produção."""

from pathlib import Path
from time import perf_counter

import pandas as pd
import pandera.pandas as pa
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from .model_evaluator import ModelEvaluator
from .structured_logging_configurator import StructuredLoggingConfigurator


class DriftMonitor:
    """Valida lotes e registra comparações estatísticas e cenários de monitoramento.

    Exemplo:
        data = HomeCreditData()
        monitor = DriftMonitor(
            data.numeric_features,
            data.categorical_features + data.binary_features,
            "data/processed/monitoring",
        )
        snapshot, drift = monitor.compare(train, production)
    """

    def __init__(
        self,
        numeric_columns,
        categorical_columns,
        output_dir,
        ks_threshold=0.01,
        psi_threshold=0.20,
        drift_share=0.20,
        auc_drop_threshold=0.05,
    ):
        """Configura colunas, limiares estatísticos e diretório dos relatórios.

        Exemplo:
            monitor = DriftMonitor(
                numeric_columns=["AMT_INCOME_TOTAL", "AMT_CREDIT"],
                categorical_columns=["NAME_INCOME_TYPE"],
                output_dir="data/processed/monitoring",
                ks_threshold=0.01,
                psi_threshold=0.20,
            )
        """
        if not 0 < ks_threshold < 1 or psi_threshold <= 0:
            raise ValueError("Limiares KS/PSI inválidos.")
        if not 0 < drift_share <= 1 or not 0 < auc_drop_threshold <= 1:
            raise ValueError("Limiares de alerta inválidos.")
        self.features = list(numeric_columns) + list(categorical_columns)
        self.definition = DataDefinition(
            numerical_columns=numeric_columns, categorical_columns=categorical_columns
        )
        self.parameters = {
            "ks_threshold": ks_threshold,
            "psi_threshold": psi_threshold,
            "drift_share": drift_share,
            "auc_drop_threshold": auc_drop_threshold,
        }
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = StructuredLoggingConfigurator.file_logger(
            "ml_prep_kit.monitoring", self.output_dir / "pipeline.jsonl"
        )

    def compare(self, reference, current):
        """Executa testes nas features configuradas, excluindo ID e alvo.

        Exemplo:
            snapshot, drift = monitor.compare(train, production)
            drifted_features = drift.loc[drift["drift"], "feature"]
        """
        snapshot = Report(
            [
                DataDriftPreset(
                    columns=self.features,
                    num_method="ks",
                    cat_method="psi",
                    num_threshold=self.parameters["ks_threshold"],
                    cat_threshold=self.parameters["psi_threshold"],
                    drift_share=self.parameters["drift_share"],
                )
            ]
        ).run(
            reference_data=Dataset.from_pandas(
                reference[self.features], self.definition
            ),
            current_data=Dataset.from_pandas(current[self.features], self.definition),
        )
        rows = []
        for metric in snapshot.dict()["metrics"]:
            config = metric["config"]
            if config["type"].endswith(":ValueDrift"):
                score, limit = float(metric["value"]), float(config["threshold"])
                detected = score < limit if config["method"] == "ks" else score >= limit
                rows.append(
                    {
                        "feature": config["column"],
                        "method": config["method"],
                        "score": score,
                        "threshold": limit,
                        "drift": bool(detected),
                    }
                )
        table = pd.DataFrame(rows)
        if set(table["feature"]) != set(self.features):
            raise ValueError("Relatório Evidently incompleto.")
        return snapshot, table

    def evaluate_batch(
        self,
        name,
        frame,
        reference,
        predictor,
        contract,
        target,
        threshold=0.5,
        control_auc=None,
    ):
        """Rejeita dados inválidos antes de chamar o modelo; salva relatórios.

        Rótulos são opcionais: sem eles, apenas Data Drift pode ser avaliado.
        O retorno expõe métricas e artefatos para registro pelo ExperimentTracker.

        Exemplo:
            result = monitor.evaluate_batch(
                name="controle",
                frame=partitions["producao"],
                reference=partitions["treino"],
                predictor=predictor,
                contract=data.contract,
                target=data.target,
                threshold=threshold,
            )
            status = result["status"]
        """
        if target in self.features:
            raise ValueError("O alvo não pode participar dos testes de drift.")
        if Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError("Nome de cenário inválido.")
        started = perf_counter()
        metrics = {"input_rows": len(frame), "predicted_rows": 0, "quality_passed": 0}
        result = {
            "status": "rejeitado",
            "metrics": metrics,
            "evaluation": None,
            "drift_table": None,
            "scores": None,
            "artifacts": [],
        }
        self.logger.info(
            "Lote iniciado.", extra={"cenario": name, "status": "iniciado"}
        )
        try:
            try:
                validated = contract.validate(frame, lazy=True)
            except pa.errors.SchemaErrors as error:
                failures = (
                    error.failure_cases.groupby(["column", "check"], dropna=False)
                    .size()
                    .reset_index(name="violations")
                )
                path = self.output_dir / f"{name}_quality.json"
                failures.to_json(path, orient="records", indent=2, force_ascii=False)
                metrics["validation_failures"] = len(error.failure_cases)
                result.update(quality_failures=failures, artifacts=[path])
                return result

            contract.validate(reference, lazy=True)
            scores = predictor.positive_scores(validated[self.features])
            evaluator = ModelEvaluator()
            if target in validated:
                evaluation = evaluator.evaluate_binary_classification(
                    validated[target], scores, threshold
                )
                metrics.update(evaluation["metrics"])
                metrics["target_positive_rate"] = metrics["positive_rate"]
                path = self.output_dir / f"classification_{name}.json"
                evaluator.save_report(evaluation, path)
                result.update(evaluation=evaluation)
                result["artifacts"].append(path)
            current_auc = metrics.get("roc_auc")
            drop = (
                control_auc - current_auc
                if control_auc is not None and current_auc is not None
                else None
            )
            snapshot, table = self.compare(reference, validated)
            html, report = (
                self.output_dir / f"{name}.html",
                self.output_dir / f"{name}.json",
            )
            snapshot.save_html(str(html))
            snapshot.save_json(str(report))
            count = int(table["drift"].sum())
            share = count / len(self.features)
            metrics.update(
                {
                    "quality_passed": 1,
                    "predicted_rows": len(scores),
                    "predicted_positive_rate": float((scores >= threshold).mean()),
                    "drifted_features": count,
                    "drift_share": share,
                    "drift_alert": int(share >= self.parameters["drift_share"]),
                    "auc_drop": drop,
                    "performance_alert": int(
                        drop >= self.parameters["auc_drop_threshold"]
                    )
                    if drop is not None
                    else None,
                }
            )
            result.update(status="concluido", scores=scores, drift_table=table)
            result["artifacts"].extend([html, report])
            return result
        except Exception:
            result["status"] = "falha"
            raise
        finally:
            metrics["duration_seconds"] = perf_counter() - started
            self.logger.info(
                "Lote encerrado.",
                extra={"cenario": name, "status": result["status"], **metrics},
            )

    def evaluate_scenarios(
        self,
        scenarios,
        reference,
        predictor,
        contract,
        target,
        tracker,
        threshold=0.5,
        control_name="controle",
        parameters=None,
        tags=None,
        figures=None,
    ):
        """Executa o controle primeiro e registra cenários e artefatos no MLflow.

        Lotes rejeitados não chegam ao modelo. Sem rótulos, o alerta de
        desempenho permanece indisponível. Retorna resultados, resumo e run_id.

        Exemplo:
            monitoring = monitor.evaluate_scenarios(
                scenarios=scenarios,
                reference=train,
                predictor=predictor,
                contract=data.contract,
                target=data.target,
                tracker=tracker,
                threshold=0.5,
                tags={"training_run_id": training_run_id},
                figures={"histogramas": figure},
            )
            summary = monitoring["summary"]
        """
        if control_name not in scenarios:
            raise ValueError("O cenário de controle é obrigatório.")
        results = {}
        control_auc = None
        names = [control_name, *[name for name in scenarios if name != control_name]]
        with tracker.run(
            self.output_dir.name,
            parameters={
                **(parameters or {}),
                **self.parameters,
                "threshold": threshold,
            },
            tags=tags,
        ) as run:
            for name in names:
                with tracker.run(name, nested=True):
                    result = self.evaluate_batch(
                        name,
                        scenarios[name],
                        reference,
                        predictor,
                        contract,
                        target,
                        threshold=threshold,
                        control_auc=control_auc,
                    )
                    tracker.log(
                        metrics=result["metrics"],
                        tags={"status": result["status"]},
                        artifacts=result["artifacts"],
                    )
                    results[name] = result
                    if name == control_name:
                        control_auc = result["metrics"].get("roc_auc")
            summary = self.summarize(results, self.output_dir / "summary.json")
            artifacts = [
                self.output_dir / "summary.json",
                self.output_dir / "pipeline.jsonl",
            ]
            for name, figure in (figures or {}).items():
                if Path(name).name != name or name in {"", ".", ".."}:
                    raise ValueError("Nome de figura inválido.")
                path = self.output_dir / f"{name}.png"
                figure.savefig(path, dpi=150, bbox_inches="tight")
                artifacts.append(path)
            tracker.log(artifacts=artifacts)
        return {"results": results, "summary": summary, "run_id": run.info.run_id}

    @staticmethod
    def summarize(results, path):
        """Consolida métricas por cenário, salva JSON e retorna a tabela.

        Exemplo:
            summary = DriftMonitor.summarize(
                {"controle": control_result, "data_drift": drift_result},
                "data/processed/monitoring/summary.json",
            )
        """
        summary = pd.DataFrame(
            {name: result["metrics"] for name, result in results.items()}
        ).T
        summary.to_json(path, orient="index", indent=2, force_ascii=False)
        return summary
