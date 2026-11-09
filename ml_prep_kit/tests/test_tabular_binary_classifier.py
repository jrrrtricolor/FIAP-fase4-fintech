"""Testes da rede neural para classificação binária tabular."""

import logging
import unittest

import pytest

torch = pytest.importorskip("torch", reason="Opcional: execute make install-ml-prep.")

from ml_prep_kit import (  # noqa: E402
    StructuredLoggingConfigurator,
    TabularBinaryClassifier,
)

logger = logging.getLogger(__name__)


class TabularBinaryClassifierTest(unittest.TestCase):
    """Valida o comportamento básico da rede neural.

    Exemplo:
        modelo = TabularBinaryClassifier(input_size=4, hidden_size=8)
        saida = modelo(torch.ones(3, 4))
    """

    def test_forward_returns_one_score_per_row(self) -> None:
        """A saída deve ter um valor bruto para cada exemplo de entrada.

        Exemplo:
            saida = modelo(torch.ones(3, 4))
        """
        StructuredLoggingConfigurator.configure()
        logger.info(
            "Validando saída da rede neural tabular.",
            extra={
                "evento": "teste_classificador_tabular_iniciado",
                "linhas": 3,
                "features": 4,
            },
        )

        modelo = TabularBinaryClassifier(
            input_size=4,
            hidden_size=8,
        )

        saida = modelo(torch.ones(3, 4))

        self.assertEqual(saida.shape, torch.Size([3]))
        logger.info(
            "Saída da rede neural tabular validada com sucesso.",
            extra={
                "evento": "teste_classificador_tabular_concluido",
                "formato_saida": list(saida.shape),
            },
        )


if __name__ == "__main__":
    unittest.main()
