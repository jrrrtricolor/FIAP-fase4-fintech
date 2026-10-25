"""Configuração de logs da API usando o formatador do framework."""

import logging

from ml_prep_kit.structured_logging_configurator import StructuredLoggingConfigurator


def setup_api_logger() -> logging.Logger:
    """Configura logs estruturados durante o startup da aplicação.

    Exemplo:
        logger = setup_api_logger()
        logger.info("api_iniciada")
    """
    StructuredLoggingConfigurator.configure()
    return logging.getLogger("credit_monitoring_api")
