"""Testes do classificador textual PyTorch com interface Scikit-Learn."""

import unittest

import numpy as np
import pytest

pytest.importorskip("torch", reason="Opcional: execute make install-ml-prep.")

from ml_prep_kit import SklearnTorchTextClassifier  # noqa: E402


class SklearnTorchTextClassifierTest(unittest.TestCase):
    """Valida treino, transformação e predição multiclasse.

    Exemplo:
        pytest ml_prep_kit/tests/test_sklearn_torch_text_classifier.py
    """

    def test_fit_predict_and_predict_proba(self) -> None:
        """Verifica treino, probabilidades e codificação dos textos.

        Exemplo:
            pytest -k "SklearnTorchTextClassifierTest and fit_predict"
        """
        texts = [
            "amostra alfa para classificacao",
            "registro alfa para treinamento",
            "amostra beta para classificacao",
            "registro beta para treinamento",
            "amostra gama para classificacao",
            "registro gama para treinamento",
        ]
        targets = [
            "classe_a",
            "classe_a",
            "classe_b",
            "classe_b",
            "classe_c",
            "classe_c",
        ]
        classifier = SklearnTorchTextClassifier(
            embedding_dim=8,
            epochs=2,
            batch_size=2,
            max_vocab_size=100,
            max_sequence_length=12,
            random_seed=42,
        )

        classifier.fit(texts, targets)
        predictions = classifier.predict(texts[:3])
        probabilities = classifier.predict_proba(texts[:3])
        encoded_texts = classifier.transform_texts(texts[:3])

        self.assertEqual(predictions.shape, (3,))
        self.assertEqual(probabilities.shape, (3, 3))
        self.assertEqual(encoded_texts.shape, (3, 12))
        np.testing.assert_allclose(
            probabilities.sum(axis=1),
            np.ones(3),
            rtol=1e-5,
        )
        self.assertEqual(set(classifier.classes_), set(targets))

    def test_fit_rejects_empty_text(self) -> None:
        """Verifica a rejeição de textos vazios no treinamento.

        Exemplo:
            pytest -k test_fit_rejects_empty_text
        """
        classifier = SklearnTorchTextClassifier(epochs=1)

        with self.assertRaisesRegex(ValueError, "Textos vazios"):
            classifier.fit(["texto válido", ""], ["classe_a", "classe_b"])


if __name__ == "__main__":
    unittest.main()
