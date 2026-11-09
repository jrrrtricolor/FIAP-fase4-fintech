"""Regras de aprovação do candidato com métricas do mesmo teste reservado."""

import json

import numpy as np
import pytest

from ml_prep_kit import ModelEvaluator


@pytest.fixture
def reference():
    """Fornece métricas válidas com amostra e duas classes suficientes.

    Exemplo:
        def test_policy(reference): ...
    """
    return dict(
        sample_size=500,
        true_negatives=200,
        false_positives=50,
        false_negatives=75,
        true_positives=175,
        roc_auc=0.70,
        average_precision=0.30,
        recall=0.70,
        brier_score=0.20,
    )


@pytest.fixture
def candidate(reference):
    """Cria melhora suficiente sem alterar a população avaliada.

    Exemplo:
        def test_policy(candidate): ...
    """
    return dict(reference, roc_auc=0.75, average_precision=0.40, brier_score=0.18)


def test_improvement_is_approved_without_mutating_reports(reference, candidate):
    """Aprova o ganho e entrega evidência serializável sem alterar as métricas.

    Exemplo:
        pytest ml_prep_kit/tests/test_candidate_acceptance.py -k improvement
    """
    original = dict(reference)
    decision = ModelEvaluator.evaluate_candidate(reference, candidate)
    assert decision["approved"] and decision["reasons"] == []
    assert all(decision["checks"].values())
    assert decision["thresholds"]["minimum_auc_gain"] == 0.02
    assert reference == original
    json.dumps(decision, allow_nan=False)


def test_identical_model_is_rejected(reference):
    """Rejeita um modelo equivalente mesmo que suas métricas absolutas sejam boas.

    Exemplo:
        pytest ml_prep_kit/tests/test_candidate_acceptance.py -k identical
    """
    decision = ModelEvaluator.evaluate_candidate(reference, reference)
    assert not decision["approved"]
    assert decision["reasons"] == ["auc_improvement"]


@pytest.mark.parametrize(
    "metric,value,reason",
    [
        ("roc_auc", 0.719, "auc_improvement"),
        ("roc_auc", 0.59, "minimum_auc"),
        ("average_precision", 0.299, "average_precision_preserved"),
        ("recall", 0.649, "recall_preserved"),
        ("brier_score", 0.211, "brier_preserved"),
    ],
)
def test_regressions_are_rejected(reference, candidate, metric, value, reason):
    """Cada limite bloqueia aprovação mesmo quando os demais são atendidos.

    Exemplo:
        pytest ml_prep_kit/tests/test_candidate_acceptance.py -k regressions
    """
    candidate[metric] = value
    decision = ModelEvaluator.evaluate_candidate(reference, candidate)
    assert not decision["approved"]
    assert reason in decision["reasons"]


def test_exact_boundaries_are_accepted(reference, candidate):
    """Aceita os limites inclusivos sem rejeição por arredondamento binário.

    Exemplo:
        pytest ml_prep_kit/tests/test_candidate_acceptance.py -k boundaries
    """
    reference["roc_auc"] = 0.67
    candidate.update(
        roc_auc=0.69, average_precision=0.30, recall=0.65, brier_score=0.21
    )
    assert ModelEvaluator.evaluate_candidate(reference, candidate)["approved"]
    reference["roc_auc"], candidate["roc_auc"] = 0.58, 0.60
    assert ModelEvaluator.evaluate_candidate(reference, candidate)["approved"]


@pytest.mark.parametrize("value", [None, np.nan, np.inf, -np.inf, "0.9", True, 1.1])
@pytest.mark.parametrize("side", ["reference", "candidate"])
def test_invalid_metrics_fail_closed(reference, candidate, value, side):
    """Recusa valores ausentes, não finitos, textuais ou fora da faixa válida.

    Exemplo:
        pytest ml_prep_kit/tests/test_candidate_acceptance.py -k invalid_metrics
    """
    metrics = reference if side == "reference" else candidate
    metrics["roc_auc"] = value
    decision = ModelEvaluator.evaluate_candidate(reference, candidate)
    assert decision["approved"] is False
    assert decision["reasons"] == ["valid_metrics"]


def test_missing_metric_is_rejected(reference, candidate):
    """Uma avaliação incompleta não satisfaz silenciosamente os critérios.

    Exemplo:
        pytest ml_prep_kit/tests/test_candidate_acceptance.py -k missing_metric
    """
    del candidate["average_precision"]
    assert ModelEvaluator.evaluate_candidate(reference, candidate)["reasons"] == [
        "valid_metrics"
    ]


@pytest.mark.parametrize("size,positive", [(499, 250), (500, 29), (500, 471)])
def test_small_sample_or_class_support_is_rejected(
    reference, candidate, size, positive
):
    """Exige tamanho total e suporte mínimo nas duas classes de ambos os modelos.

    Exemplo:
        pytest ml_prep_kit/tests/test_candidate_acceptance.py -k small_sample
    """
    for metrics in (reference, candidate):
        metrics.update(
            sample_size=size,
            true_positives=positive,
            false_negatives=0,
            true_negatives=size - positive,
            false_positives=0,
        )
    decision = ModelEvaluator.evaluate_candidate(reference, candidate)
    expected = "minimum_sample_size" if size < 500 else "minimum_class_support"
    assert not decision["approved"] and expected in decision["reasons"]


@pytest.mark.parametrize("size,positive", [(500, 30), (500, 470)])
def test_minimum_class_support_is_inclusive(reference, candidate, size, positive):
    """Trinta exemplos de qualquer classe atendem ao limite documentado.

    Exemplo:
        pytest ml_prep_kit/tests/test_candidate_acceptance.py -k inclusive
    """
    for metrics in (reference, candidate):
        metrics.update(
            sample_size=size,
            true_positives=positive,
            false_negatives=0,
            true_negatives=size - positive,
            false_positives=0,
        )
    assert ModelEvaluator.evaluate_candidate(reference, candidate)["approved"]


@pytest.mark.parametrize("metric", ["sample_size", "true_positives", "true_negatives"])
def test_divergent_populations_are_rejected(reference, candidate, metric):
    """Impede comparar amostras com tamanhos ou suportes de classes diferentes.

    Exemplo:
        pytest ml_prep_kit/tests/test_candidate_acceptance.py -k divergent
    """
    candidate[metric] += 1
    decision = ModelEvaluator.evaluate_candidate(reference, candidate)
    assert not decision["approved"]
    assert "matching_sample" in decision["reasons"]


def test_inconsistent_counts_are_rejected_even_when_equal(reference, candidate):
    """Matrizes igualmente inconsistentes não comprovam a população de teste.

    Exemplo:
        pytest ml_prep_kit/tests/test_candidate_acceptance.py -k inconsistent
    """
    reference["sample_size"] = candidate["sample_size"] = 501
    decision = ModelEvaluator.evaluate_candidate(reference, candidate)
    assert not decision["approved"]
    assert "consistent_counts" in decision["reasons"]


def test_policy_accepts_reports_from_binary_evaluation():
    """Integra a política aos relatórios reais gerados pelo avaliador existente.

    Exemplo:
        pytest ml_prep_kit/tests/test_candidate_acceptance.py -k binary_evaluation
    """
    evaluator = ModelEvaluator()
    target = np.tile([0, 0, 1, 1], 125)
    original = evaluator.evaluate_binary_classification(
        target, np.tile([0.3, 0.7], 250)
    )
    improved = evaluator.evaluate_binary_classification(
        target, np.where(target, 0.9, 0.1)
    )
    assert evaluator.evaluate_candidate(original["metrics"], improved["metrics"])[
        "approved"
    ]
