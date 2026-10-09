"""Métricas reutilizáveis para avaliação de classificadores."""

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
    """Calcula métricas para classificação binária ou multiclasse."""

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
        """Retorna métricas por classe, incluindo rótulos explícitos sem suporte."""
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
        predictions = (scores >= threshold).astype(int)
        metrics = self.evaluate_classification(target, predictions)
        tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()

        def rate(numerator, denominator):
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
