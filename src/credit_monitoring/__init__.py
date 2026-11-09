"""Preserva os imports públicos do workflow após sua organização em pacote."""

from importlib import import_module

__all__ = ["EXPERIMENT_NAME", "CreditMonitoring", "run_workflow"]


def __getattr__(name):
    """Carrega o workflow somente quando seu consumidor o solicita.

    Exemplo:
        from credit_monitoring import run_workflow
    """
    if name not in __all__:
        raise AttributeError(f"Atributo desconhecido: {name}.")
    module = import_module(f"{__name__}.credit_monitoring")
    value = getattr(module, name)
    globals()[name] = value
    return value
