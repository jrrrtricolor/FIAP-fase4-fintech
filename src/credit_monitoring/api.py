"""API REST para inferência de score de crédito."""

import os
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, cast

from fastapi import Depends, FastAPI, HTTPException, Request, status
from pandera.typing import DataFrame
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from ml_prep_kit import HomeCreditDataV1Schema, ModelPredictor

from .api_constants import MLFLOW_MODEL_NAME
from .config.api_logging_middleware import LoggingMiddleware
from .config.logging_config import setup_api_logger
from .prometheus.metrics import (
    MODEL_INFO,
    PREDICTION_CONFIDENCE,
    PREDICTION_DURATION,
    PREDICTIONS_TOTAL,
    REFERENCE_MODEL_AGE_YEARS,
    REFERENCE_MODEL_AMT_CREDIT,
    REFERENCE_MODEL_AMT_INCOME_TOTAL,
)


class PredictionRequest(BaseModel):
    prediction_data: DataFrame[HomeCreditDataV1Schema]

class PredictionResponse(BaseModel):
    """Classificação e probabilidades retornadas ao cliente."""

    credit_scores: list[float]
    model_version: str
    inference_time_ms: float


ModelLoader = Callable[[], tuple[ModelPredictor, str]]

LOGGER = setup_api_logger()
DEFAULT_TRACKING_FILE = Path.joinpath(
    Path.cwd(), "data", "processed", "monitoring", "mlflow.db"
)
DEFAULT_TRACKING_URI = f"sqlite:///{DEFAULT_TRACKING_FILE}"
DEFAULT_MODEL_URI = f"models:/{MLFLOW_MODEL_NAME}@production"

def load_model() -> tuple[ModelPredictor, str]:
    """Carrega uma única vez o modelo."""
    version = os.getenv("MODEL_VERSION", "")
    model_uri = os.getenv("MODEL_URI", DEFAULT_MODEL_URI)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    
    predictor = ModelPredictor.from_mlflow(
        model_uri=model_uri,
        tracking_uri=tracking_uri,
    )

    return predictor, version

def create_app(
    predictor: ModelPredictor | None = None,
    model_version: str = "unavailable",
    model_loader: ModelLoader | None = load_model,
) -> FastAPI:
    """Cria a aplicação e permite injetar o modelo nos testes."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        # Em produção, o modelo é carregado somente durante o startup.
        if application.state.predictor is None and model_loader is not None:
            loaded_predictor, loaded_version = model_loader()
            application.state.predictor = loaded_predictor
            application.state.model_version = loaded_version

        MODEL_INFO.labels(version=application.state.model_version).set(1)
        yield

    application = FastAPI(
        title="Credit Monitoring API",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.predictor = predictor
    application.state.model_version = model_version
    if predictor is not None:
        MODEL_INFO.labels(version=model_version).set(1)

    application.add_middleware(LoggingMiddleware)
    Instrumentator().instrument(application).expose(
        application,
        endpoint="/metrics",
        summary="Métricas de desempenho da API para Prometheus",
    )

    def require_predictor(request: Request) -> ModelPredictor:
        """Recusa inferências enquanto o modelo não estiver disponível."""
        loaded_predictor = request.app.state.predictor
        if loaded_predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Modelo indisponível.",
            )
        return loaded_predictor

    @application.get("/health")
    def health() -> dict[str, str]:
        """Confirma que o processo da API está ativo."""
        return {"status": "ok"}

    @application.get("/ready")
    def ready(request: Request) -> dict[str, str | bool]:
        """Informa se o modelo já está carregado em memória."""
        is_ready = request.app.state.predictor is not None
        return {
            "ready": is_ready,
            "model_version": request.app.state.model_version,
        }

    @application.post("/predict", response_model=PredictionResponse)
    def predict(
        payload: PredictionRequest,
        loaded_predictor: Annotated[
            ModelPredictor,
            Depends(require_predictor),
        ],
        request: Request,
    ) -> PredictionResponse:
        """Retorna o score para um perfil"""
        started_at = time.perf_counter()
        requests = payload.prediction_data

        scores = loaded_predictor.positive_scores(requests)
        probabilities = loaded_predictor.predict_proba(requests)

        inference_time_ms = (time.perf_counter() - started_at) * 1_000

        LOGGER.info(
            "predicao_gerada",
            extra={
                "duracao_ms": inference_time_ms,
                "confiancas": list(map(max, probabilities)),
                "scores": scores,
            },
        )

        PREDICTION_DURATION.observe(inference_time_ms / 1_000)
        PREDICTIONS_TOTAL.inc()

        for probas in probabilities:
            PREDICTION_CONFIDENCE.observe(max(probas))

        for data in requests.itertuples():
            (
                REFERENCE_MODEL_AGE_YEARS
                .labels("AGE_YEARS")
                .observe(cast(float, data.AGE_YEARS))
            )

            (
                REFERENCE_MODEL_AMT_CREDIT
                .labels("AMT_CREDIT")
                .observe(cast(float, data.AMT_CREDIT))
            )

            (
                REFERENCE_MODEL_AMT_INCOME_TOTAL
                .labels("AMT_INCOME_TOTAL")
                .observe(cast(float, data.AMT_INCOME_TOTAL))
            )

        return PredictionResponse(
            credit_scores=scores,
            model_version=request.app.state.model_version,
            inference_time_ms=inference_time_ms,
        )

    @application.get("/income-types")
    def income_types(
        loaded_predictor: Annotated[
            ModelPredictor,
            Depends(require_predictor),
        ],
        request: Request,
    ) -> list[str]:
        return loaded_predictor.get_categorical_column_values("NAME_INCOME_TYPE")

    @application.get("/education-types")
    def education_types(
        loaded_predictor: Annotated[
            ModelPredictor,
            Depends(require_predictor),
        ],
        request: Request,
    ) -> list[str]:
        return loaded_predictor.get_categorical_column_values("NAME_EDUCATION_TYPE")

    @application.get("/housing-types")
    def housing_types(
        loaded_predictor: Annotated[
            ModelPredictor,
            Depends(require_predictor),
        ],
        request: Request,
    ) -> list[str]:
        return loaded_predictor.get_categorical_column_values("NAME_HOUSING_TYPE")
    
    return application


app = create_app()
