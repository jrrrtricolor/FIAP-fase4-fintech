"""Testes da rede neural textual multiclasse."""

import unittest

import pytest

torch = pytest.importorskip("torch", reason="Opcional: execute make install-ml-prep.")

from ml_prep_kit import TextMulticlassClassifier  # noqa: E402


class TextMulticlassClassifierTest(unittest.TestCase):
    """Valida o contrato de entrada e saída da rede textual.

    Exemplo:
        pytest ml_prep_kit/tests/test_text_multiclass_classifier.py
    """

    def test_forward_returns_one_logit_per_class(self) -> None:
        """Verifica um logit por classe para cada texto.

        Exemplo:
            pytest -k test_forward_returns_one_logit_per_class
        """
        model = TextMulticlassClassifier(
            vocab_size=20,
            num_classes=3,
            embedding_dim=8,
        )
        token_ids = torch.tensor(
            [
                [2, 3, 4, 0],
                [5, 6, 0, 0],
            ],
            dtype=torch.long,
        )

        logits = model(token_ids)

        self.assertEqual(logits.shape, torch.Size([2, 3]))

    def test_constructor_rejects_single_class(self) -> None:
        """Verifica que a rede exige pelo menos duas classes.

        Exemplo:
            pytest -k test_constructor_rejects_single_class
        """
        with self.assertRaisesRegex(ValueError, "pelo menos 2 classes"):
            TextMulticlassClassifier(vocab_size=20, num_classes=1)


if __name__ == "__main__":
    unittest.main()
