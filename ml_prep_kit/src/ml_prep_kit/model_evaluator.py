"""Métricas reutilizáveis para avaliação de classificadores."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class ModelEvaluator:
    """Calcula métricas para classificação binária ou multiclasse.

    Exemplo:
        evaluator = ModelEvaluator()
        report = evaluator.evaluate_binary_classification(
            [0, 1, 0, 1], [0.1, 0.8, 0.3, 0.6], threshold=0.5
        )
    """

    @classmethod
    def save_report(cls, report, path):
        """Salva métricas e tabelas agregadas; não persiste dados individuais.

        Exemplo:
            evaluator.save_report(
                report, "data/processed/classification_report.json"
            )
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                cls._report_values(report),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def _report_values(cls, value):
        """Converte DataFrames em estruturas JSON, preservando índice e colunas.

        Exemplo:
            payload = ModelEvaluator._report_values(
                {"metrics": pd.DataFrame({"roc_auc": [0.63]})}
            )
        """
        if isinstance(value, pd.DataFrame):
            return json.loads(value.to_json(orient="split"))
        if isinstance(value, dict):
            return {key: cls._report_values(item) for key, item in value.items()}
        return value

    def save_evaluation(self, evaluation, intervals, thresholds, figures, directory):
        """Salva relatórios, tabelas e figuras; intervals=None omite intervalos.

        Exemplo:
            directory = evaluator.save_evaluation(
                evaluation, confidence_intervals, threshold_metrics,
                {"confusion_matrix": confusion_figure}, "data/processed/evaluation",
            )
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.save_report(
            {"reports": evaluation["reports"], "confidence_intervals": intervals},
            directory / "classification_report.json",
        )
        self._save_evaluation_tables(evaluation, intervals, thresholds, directory)
        for name, figure in (figures or {}).items():
            figure.savefig(directory / f"{name}.png", dpi=160, bbox_inches="tight")
        return directory

    @staticmethod
    def _save_evaluation_tables(evaluation, intervals, thresholds, directory):
        """Exporta somente as tabelas calculadas, sem simular evidências ausentes.

        Exemplo:
            evaluator._save_evaluation_tables(evaluation, None, thresholds, directory)
        """
        tables = {
            "global_metrics": evaluation["global_metrics"],
            "metrics_by_class": evaluation["class_metrics"],
            "confidence_intervals": intervals,
            "threshold_metrics": thresholds,
        }
        for name, frame in tables.items():
            path = directory / f"{name}.csv"
            if frame is not None:
                frame.to_csv(path)
            else:
                path.unlink(missing_ok=True)

    def select_threshold(self, target, scores, thresholds=None) -> tuple:
        """Escolhe maior F1 na validação; desempata por recall e menor limiar.

        Exemplo:
            threshold, table = evaluator.select_threshold(
                y_validation, validation_scores, thresholds=[0.3, 0.5, 0.7]
            )
        """
        if thresholds is None:
            thresholds = np.arange(0.05, 1.0, 0.05)
        rows = [
            self._binary_metrics(
                *self._validate_binary_inputs(target, scores, float(t)), float(t)
            )
            for t in thresholds
        ]
        if not rows:
            raise ValueError("Informe pelo menos um limiar.")
        table = pd.DataFrame(rows).sort_values(
            ["f1", "recall", "threshold"], ascending=[False, False, True]
        )
        if table["f1"].isna().all():
            raise ValueError("Não há F1 definido para escolher o limiar.")
        return float(table.iloc[0]["threshold"]), table

    def evaluate_models(self, models, datasets, thresholds) -> dict:
        """Avalia modelos congelados em partições, sem ajuste ou seleção.

        Exemplo:
            evaluation = evaluator.evaluate_models(
                models={"logistic_regression": trained_pipeline},
                datasets={"teste": (X_test, y_test)},
                thresholds={"logistic_regression": selected_threshold},
            )
        """
        from .model_predictor import ModelPredictor

        reports, scores, rows, classes = {}, {}, [], []
        for name, model in models.items():
            reports[name], scores[name] = {}, {}
            predictor = ModelPredictor(model)
            for split, (features, target) in datasets.items():
                score = predictor.positive_scores(features)
                report = self.evaluate_binary_classification(
                    target, score, threshold=thresholds[name]
                )
                reports[name][split] = report
                scores[name][split] = score
                rows.append({"model": name, "split": split, **report["metrics"]})
                classes.append(report["by_class"].assign(model=name, split=split))
        return {
            "reports": reports,
            "scores": scores,
            "global_metrics": pd.DataFrame(rows).set_index(["model", "split"]),
            "class_metrics": pd.concat(classes, ignore_index=True),
        }

    def evaluate_classification(
        self,
        y_true: pd.Series | np.ndarray | list,
        y_pred: pd.Series | np.ndarray | list,
        y_score: pd.Series | pd.DataFrame | np.ndarray | None = None,
    ) -> dict[str, float]:
        """Retorna métricas globais de classificação.

        As métricas macro e weighted estão sempre presentes. Para manter
        compatibilidade com fluxos binários, ``precision``, ``recall`` e ``f1``
        representam a classe positiva em problemas 0/1 e a média macro em
        problemas multiclasse.

        Exemplo:
            metrics = evaluator.evaluate_classification(
                y_true=[0, 1, 0, 1],
                y_pred=[0, 1, 0, 0],
                y_score=[0.1, 0.8, 0.2, 0.4],
            )
        """
        labels = np.unique(np.asarray(y_true))
        is_binary = len(labels) == 2
        default_average = "binary" if is_binary else "macro"
        average_options = {
            "average": default_average,
            "zero_division": 0,
        }
        if is_binary:
            average_options["pos_label"] = labels[-1]

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true,
                y_pred,
                **average_options,
            ),
            "recall": recall_score(
                y_true,
                y_pred,
                **average_options,
            ),
            "f1": f1_score(
                y_true,
                y_pred,
                **average_options,
            ),
            "precision_macro": precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            "recall_macro": recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            "f1_macro": f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            "precision_weighted": precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "recall_weighted": recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "f1_weighted": f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
        }

        if y_score is not None and len(labels) > 1:
            score_array = np.asarray(y_score)

            if is_binary:
                if score_array.ndim == 2:
                    score_array = score_array[:, -1]
                metrics["roc_auc"] = roc_auc_score(y_true, score_array)
            else:
                metrics["roc_auc"] = roc_auc_score(
                    y_true,
                    score_array,
                    labels=labels,
                    multi_class="ovr",
                    average="macro",
                )

        return {name: float(value) for name, value in metrics.items()}

    def evaluate_classification_by_class(
        self,
        y_true: pd.Series | np.ndarray | list,
        y_pred: pd.Series | np.ndarray | list,
        labels: list | np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Retorna métricas por classe, incluindo rótulos explícitos sem suporte.

        Exemplo:
            by_class = evaluator.evaluate_classification_by_class(
                y_true=[0, 1, 0, 1], y_pred=[0, 1, 0, 0], labels=[0, 1]
            )
        """
        labels = np.unique(np.asarray(y_true)) if labels is None else np.asarray(labels)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )

        return pd.DataFrame(
            {
                "class": labels,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    def evaluate_binary_classification(
        self,
        y_true: pd.Series | np.ndarray | list,
        y_score: pd.Series | np.ndarray | list,
        threshold: float = 0.5,
        calibration_bins: int = 10,
    ) -> dict[str, object]:
        """Avalia alvos 0/1 usando scores da classe 1 e um limiar fixo.

        Retorna métricas, tabela por classe, matriz de confusão, curvas ROC e
        precisão-recall e calibração em intervalos uniformes. A matriz usa linhas
        reais e colunas previstas, ambas na ordem [0, 1]. Taxas sem denominador
        e métricas de ordenação sem as duas classes recebem None. Nas tabelas
        por classe, a convenção zero_division=0 é mantida.

        Os scores devem ser um vetor de probabilidades em [0, 1]. A avaliação
        não escolhe o limiar e não ajusta nem calibra o modelo.

        Exemplo:
            report = evaluator.evaluate_binary_classification(
                y_true=[0, 1, 0, 1],
                y_score=[0.1, 0.8, 0.2, 0.4],
                threshold=0.5,
            )
            matrix = report["confusion_matrix"]
        """
        target, scores = self._validate_binary_inputs(y_true, y_score, threshold)
        if not isinstance(calibration_bins, int) or calibration_bins < 2:
            raise ValueError("calibration_bins deve ser inteiro maior ou igual a 2.")

        predictions = (scores >= threshold).astype(int)
        metrics = self._binary_metrics(target, scores, threshold)
        by_class = self.evaluate_classification_by_class(
            target, predictions, labels=[0, 1]
        )
        matrix = pd.DataFrame(
            confusion_matrix(target, predictions, labels=[0, 1]),
            index=pd.Index([0, 1], name="actual_class"),
            columns=pd.Index([0, 1], name="predicted_class"),
        )
        roc = pd.DataFrame(columns=["false_positive_rate", "true_positive_rate"])
        pr = pd.DataFrame(columns=["recall", "precision"])
        notes = []
        if len(np.unique(target)) == 2:
            fpr, tpr, _ = roc_curve(target, scores)
            precision, recall, _ = precision_recall_curve(target, scores)
            roc = pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr})
            pr = pd.DataFrame({"recall": recall, "precision": precision})
        else:
            notes.append("ROC AUC, Average Precision e curvas exigem as duas classes.")

        bin_ids = np.minimum(
            (scores * calibration_bins).astype(int), calibration_bins - 1
        )
        calibration = (
            pd.DataFrame({"bin": bin_ids, "score": scores, "target": target})
            .groupby("bin", sort=True)
            .agg(
                mean_probability=("score", "mean"),
                observed_positive_rate=("target", "mean"),
                support=("target", "size"),
            )
            .reset_index()
        )
        return {
            "metrics": metrics,
            "by_class": by_class,
            "confusion_matrix": matrix,
            "roc_curve": roc,
            "precision_recall_curve": pr,
            "calibration": calibration,
            "notes": notes,
        }

    def bootstrap_binary_metrics(
        self,
        y_true: pd.Series | np.ndarray | list,
        y_score: pd.Series | np.ndarray | list,
        threshold: float = 0.5,
        n_resamples: int = 500,
        confidence_level: float = 0.95,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Intervalos percentis por bootstrap estratificado, sem reajustar modelo.

        Mantém a proporção de classes da amostra. Os intervalos refletem a
        variação de amostragem condicionada ao modelo, limiar e prevalência
        observados; não incluem a incerteza do treinamento ou da seleção.
        valid_resamples informa quantas reamostragens tiveram a métrica definida;
        intervalos de métricas indefinidas em todas as reamostragens ficam ausentes.

        Exemplo:
            intervals = evaluator.bootstrap_binary_metrics(
                y_test, test_scores, threshold=0.5,
                n_resamples=500, confidence_level=0.95, random_state=42,
            )
        """
        target, scores = self._validate_binary_inputs(y_true, y_score, threshold)
        if len(np.unique(target)) != 2:
            raise ValueError("O bootstrap estratificado exige as duas classes.")
        if not isinstance(n_resamples, int) or n_resamples < 2:
            raise ValueError("n_resamples deve ser inteiro maior ou igual a 2.")
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level deve estar entre 0 e 1.")

        metric_names = ["roc_auc", "average_precision", "precision", "recall", "f1"]
        estimate = self._binary_metrics(target, scores, threshold)
        groups = [np.flatnonzero(target == label) for label in [0, 1]]
        rng = np.random.default_rng(random_state)
        samples = []
        for _ in range(n_resamples):
            indices = np.concatenate(
                [rng.choice(group, size=len(group), replace=True) for group in groups]
            )
            metrics = self._binary_metrics(target[indices], scores[indices], threshold)
            samples.append({name: metrics[name] for name in metric_names})

        sample_metrics = pd.DataFrame(samples, dtype=float)
        tail = (1 - confidence_level) / 2
        return pd.DataFrame(
            [
                {
                    "metric": name,
                    "estimate": estimate[name],
                    "lower": sample_metrics[name].quantile(tail),
                    "upper": sample_metrics[name].quantile(1 - tail),
                    "valid_resamples": int(sample_metrics[name].notna().sum()),
                    "confidence_level": confidence_level,
                }
                for name in metric_names
            ]
        )

    def _validate_binary_inputs(self, y_true, y_score, threshold):
        """Valida alvos, probabilidades e limiar; retorna os vetores NumPy.

        Exemplo:
            target, scores = evaluator._validate_binary_inputs(
                [0, 1], [0.2, 0.8], threshold=0.5
            )
        """
        target = np.asarray(y_true)
        scores = np.asarray(y_score, dtype=float)
        if target.ndim != 1 or scores.ndim != 1 or len(target) == 0:
            raise ValueError(
                "Alvo e scores devem ser vetores unidimensionais não vazios."
            )
        if len(target) != len(scores):
            raise ValueError("Alvo e scores devem ter o mesmo tamanho.")
        if not np.isin(target, [0, 1]).all():
            raise ValueError("O alvo binário deve conter somente 0 e 1.")
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("Os scores devem ser finitos e estar entre 0 e 1.")
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("O limiar deve ser finito e estar entre 0 e 1.")
        return target.astype(int), scores

    def _binary_metrics(self, target, scores, threshold):
        """Calcula métricas binárias a partir de vetores previamente validados.

        Exemplo:
            metrics = evaluator._binary_metrics(
                np.array([0, 1]), np.array([0.2, 0.8]), threshold=0.5
            )
        """
        predictions = (scores >= threshold).astype(int)
        metrics = self.evaluate_classification(target, predictions)
        tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()

        def rate(numerator, denominator):
            """Calcula uma taxa ou retorna None quando o denominador é zero.

            Exemplo:
                recall = rate(3, 4)
                undefined = rate(0, 0)
            """
            return float(numerator / denominator) if denominator else None

        both_classes = len(np.unique(target)) == 2
        sensitivity = rate(tp, tp + fn)
        specificity = rate(tn, tn + fp)
        metrics.update(
            {
                "threshold": float(threshold),
                "sample_size": int(len(target)),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "precision": rate(tp, tp + fp),
                "recall": sensitivity,
                "f1": rate(2 * tp, 2 * tp + fp + fn),
                "specificity": specificity,
                "false_positive_rate": rate(fp, fp + tn),
                "false_negative_rate": rate(fn, fn + tp),
                "negative_predictive_value": rate(tn, tn + fn),
                "balanced_accuracy": (
                    (sensitivity + specificity) / 2 if both_classes else None
                ),
                "matthews_correlation": (
                    float(matthews_corrcoef(target, predictions))
                    if both_classes
                    else None
                ),
                "roc_auc": float(roc_auc_score(target, scores))
                if both_classes
                else None,
                "average_precision": (
                    float(average_precision_score(target, scores))
                    if both_classes
                    else None
                ),
                "brier_score": float(brier_score_loss(target, scores)),
                "log_loss": float(log_loss(target, scores, labels=[0, 1])),
                "positive_rate": float(target.mean()),
                "predicted_positive_rate": float(predictions.mean()),
            }
        )
        return metrics
