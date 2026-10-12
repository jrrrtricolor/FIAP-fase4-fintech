"""API REST para inferência de score de crédito."""

import os
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
import pandas as pd

from fastapi import Depends, FastAPI, HTTPException, Request, status
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from ml_prep_kit import ModelPredictor

from .config.api_logging_middleware import LoggingMiddleware
from .config.logging_config import setup_api_logger
from .prometheus.metrics import (
    MODEL_INFO,
    PREDICTION_CONFIDENCE,
    PREDICTION_DURATION,
    PREDICTIONS_TOTAL,
)
from .api_constants import (
    DEFAULT_MODEL_FILE,
    DEFAULT_MODEL_PATH
)

class PredictionRequest(BaseModel):
    AGE_YEARS: int
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    CNT_CHILDREN: int
    
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_HOUSING_TYPE: str
    
    FLAG_OWN_CAR: bool
    FLAG_OWN_REALTY: bool

    def to_dict(self):
        return {
            "AGE_YEARS": self.AGE_YEARS,
            "AMT_INCOME_TOTAL": self.AMT_INCOME_TOTAL,
            "AMT_CREDIT": self.AMT_CREDIT,
            "AMT_ANNUITY": self.AMT_ANNUITY,
            "CNT_CHILDREN": self.CNT_CHILDREN,
            "NAME_INCOME_TYPE": self.NAME_INCOME_TYPE,
            "NAME_EDUCATION_TYPE": self.NAME_EDUCATION_TYPE,
            "NAME_HOUSING_TYPE": self.NAME_HOUSING_TYPE,
            "FLAG_OWN_CAR": self.FLAG_OWN_CAR,
            "FLAG_OWN_REALTY": self.FLAG_OWN_REALTY,
        }

class PredictionResponse(BaseModel):
    """Classificação e probabilidades retornadas ao cliente."""

    credit_score: float
    model_version: str
    inference_time_ms: float


ModelLoader = Callable[[], tuple[ModelPredictor, str]]
LOGGER = setup_api_logger()

def load_model() -> tuple[ModelPredictor, str]:
    """Carrega uma única vez o modelo."""
    model_path = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_FILE))

    if not model_path.is_file():
        raise FileNotFoundError(f"Artefato de inferência ausente: {model_path}.")
    
    if model_path.suffix != ".pkl":
        raise ValueError("MODEL_PATH deve apontar para um artefato .pkl.")
    
    predictor = ModelPredictor.from_pkl(
        model_path,
    )

    version = None

    with open(Path(DEFAULT_MODEL_PATH).joinpath("version.txt"), "r", encoding="utf-8") as file:
        version = file.read().strip()
    
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
        requests = pd.DataFrame([payload.to_dict()])
        score = loaded_predictor.predict(requests)[0]
        probabilities = loaded_predictor.predict_proba(requests)[0]

        inference_time_ms = (time.perf_counter() - started_at) * 1_000

        LOGGER.info(
            "predicao_gerada",
            extra={
                "duracao_ms": inference_time_ms,
                "confianca": max(probabilities),
                "score": score,
            },
        )

        PREDICTION_DURATION.observe(inference_time_ms / 1_000)
        PREDICTIONS_TOTAL.inc()
        PREDICTION_CONFIDENCE.observe(max(probabilities))

        return PredictionResponse(
            credit_score=score,
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
