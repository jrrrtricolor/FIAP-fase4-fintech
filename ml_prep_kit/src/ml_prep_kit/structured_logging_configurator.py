"""Configuração reutilizável para logging estruturado."""

import logging
import sys
from pathlib import Path

from ml_prep_kit.structured_json_formatter import StructuredJsonFormatter


class StructuredLoggingConfigurator:
    """Configura logs estruturados para scripts e pipelines.

    Use esta classe quando quiser que a aplicação escreva logs em JSON usando
    o formatador padrão do ``ml_prep_kit``.

    Exemplo:
        StructuredLoggingConfigurator.configure()

        logger = logging.getLogger(__name__)
        logger.info(
            "Treino iniciado.",
            extra={"evento": "treino_iniciado"},
        )
    """

    @classmethod
    def configure(cls, nivel: int = logging.INFO) -> None:
        """Configura o logger raiz para emitir logs estruturados.

        Exemplo:
            StructuredLoggingConfigurator.configure(logging.INFO)
        """
        manipulador = logging.StreamHandler(sys.stdout)
        manipulador.setFormatter(StructuredJsonFormatter())

        logger_raiz = logging.getLogger()
        logger_raiz.handlers.clear()
        logger_raiz.addHandler(manipulador)
        logger_raiz.setLevel(nivel)

    @classmethod
    def file_logger(cls, name: str, path: str | Path) -> logging.Logger:
        """Cria logger isolado e fecha handlers anteriores em reexecuções.

        Exemplo:
            logger = StructuredLoggingConfigurator.file_logger(
                "fintech.training", "data/processed/training.jsonl"
            )
            logger.info("Treino iniciado.", extra={"evento": "treino_iniciado"})
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(name)
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(StructuredJsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger
