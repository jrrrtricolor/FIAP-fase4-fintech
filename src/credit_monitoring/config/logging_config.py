import logging

from ml_prep_kit.structured_logging_configurator import StructuredLoggingConfigurator


def setup_api_logger() -> logging.Logger:
    """Configura o logger estruturado uma única vez por processo."""
    StructuredLoggingConfigurator.configure()

    logger = logging.getLogger("credit_monitoring_api")

    return logger
