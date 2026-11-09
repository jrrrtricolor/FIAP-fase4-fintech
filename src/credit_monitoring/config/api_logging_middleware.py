"""Logs estruturados e correlação de requisições da API."""

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ..prometheus.metrics import ERRORS_TOTAL

LOGGER = logging.getLogger("credit_monitoring_api")


class LoggingMiddleware(BaseHTTPMiddleware):
    """Registra duração e erros, sem armazenar os perfis dos clientes.

    Exemplo:
        application.add_middleware(LoggingMiddleware)
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Correlaciona o início e o resultado de cada requisição.

        Exemplo:
            response = await middleware.dispatch(request, call_next)
        """
        request.state.request_id = str(uuid.uuid4())
        started_at = time.perf_counter()
        self.log_request(request, "inicio_de_requisicao")
        try:
            response = await call_next(request)
        except Exception:
            self.log_request(request, "erro_de_requisicao", started_at, 500)
            LOGGER.exception("falha_na_requisicao")
            raise
        self.log_request(request, "fim_de_requisicao", started_at, response.status_code)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @staticmethod
    def log_request(request, event, started_at=None, status_code=200):
        """Publica duração e status com o mesmo identificador de correlação.

        Exemplo:
            LoggingMiddleware.log_request(request, "fim_de_requisicao", start, 200)
        """
        if status_code >= 400:
            ERRORS_TOTAL.inc()
        LOGGER.log(
            logging.ERROR if status_code >= 500 else logging.INFO,
            event,
            extra={
                "request_id": request.state.request_id,
                "metodo_http": request.method,
                "rota": request.url.path,
                "status_code": status_code,
                "duracao": time.perf_counter() - started_at if started_at else 0,
            },
        )
