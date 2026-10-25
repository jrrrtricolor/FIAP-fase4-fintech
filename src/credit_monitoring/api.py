"""API REST para inferência com o Pipeline aprovado no MLflow."""

import logging
import os
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from pandera.typing import DataFrame
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, field_validator

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

LOGGER = logging.getLogger("credit_monitoring_api")
DEFAULT_TRACKING_FILE = Path.cwd() / "data/processed/monitoring/mlflow.db"
DEFAULT_TRACKING_URI = f"sqlite:///{DEFAULT_TRACKING_FILE}"
DEFAULT_MODEL_URI = f"models:/{MLFLOW_MODEL_NAME}@production"
ModelLoader = Callable[[], tuple[ModelPredictor, str]]
router = APIRouter()


class PredictionRequest(BaseModel):
    """Valida os perfis com o contrato de inferência do framework.

    Exemplo:
        request = PredictionRequest(prediction_data=perfis)
    """

    prediction_data: DataFrame[HomeCreditDataV1Schema]

    @field_validator("prediction_data")
    @classmethod
    def require_rows(cls, frame):
        """Recusa lotes vazios antes de executar o modelo.

        Exemplo:
            PredictionRequest.require_rows(perfis)
        """
        if frame.empty:
            raise ValueError("Informe pelo menos um perfil.")
        return frame


class PredictionResponse(BaseModel):
    """Retorna scores, versão efetivamente carregada e tempo de inferência.

    Exemplo:
        result = PredictionResponse(
            credit_scores=[0.2], model_version="4", inference_time_ms=3.0
        )
    """

    credit_scores: list[float]
    model_version: str
    inference_time_ms: float


def load_model() -> tuple[ModelPredictor, str]:
    """Resolve o alias e carrega uma versão imutável do Registry no startup.

    Exemplo:
        predictor, version = load_model()
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    model_uri, version = ModelPredictor.resolve_mlflow_uri(
        os.getenv("MODEL_URI", DEFAULT_MODEL_URI), tracking_uri
    )
    predictor = ModelPredictor.from_mlflow(
        model_uri=model_uri, tracking_uri=tracking_uri
    )
    return predictor, version


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Carrega o modelo; falhas ficam nos logs e na prontidão HTTP.

    Exemplo:
        application = FastAPI(lifespan=lifespan)
    """
    setup_api_logger()
    MODEL_INFO.clear()
    try:
        if application.state.predictor is None and application.state.model_loader:
            predictor, version = application.state.model_loader()
            application.state.predictor = predictor
            application.state.model_version = version
        if application.state.predictor is not None:
            MODEL_INFO.labels(version=application.state.model_version).set(1)
    except Exception:
        LOGGER.exception("falha_ao_carregar_modelo")
    yield


def require_predictor(request: Request) -> ModelPredictor:
    """Recusa inferências enquanto o modelo não estiver disponível.

    Exemplo:
        predictor = require_predictor(request)
    """
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo indisponível.")
    return predictor


PredictorDependency = Annotated[ModelPredictor, Depends(require_predictor)]


@router.get("/health")
def health() -> dict[str, str]:
    """Confirma que o processo está ativo, mesmo sem modelo disponível.

    Exemplo:
        status = health()
    """
    return {"status": "ok"}


@router.get("/ready")
def ready(request: Request, response: Response) -> dict[str, str | bool]:
    """Retorna HTTP 503 enquanto não houver um modelo carregado.

    Exemplo:
        status = ready(request, response)
    """
    is_ready = request.app.state.predictor is not None
    response.status_code = 200 if is_ready else 503
    return {"ready": is_ready, "model_version": request.app.state.model_version}


@router.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest, loaded_predictor: PredictorDependency, request: Request
) -> PredictionResponse:
    """Calcula uma vez os scores de inadimplência dos perfis validados.

    Exemplo:
        result = predict(payload, predictor, request)
    """
    started_at = time.perf_counter()
    scores = loaded_predictor.positive_scores(payload.prediction_data)
    duration = time.perf_counter() - started_at
    record_predictions(payload.prediction_data, scores, duration)
    return PredictionResponse(
        credit_scores=scores,
        model_version=request.app.state.model_version,
        inference_time_ms=duration * 1_000,
    )


def record_predictions(frame, scores, duration: float) -> None:
    """Observa a inferência binária sem executar novamente o Pipeline.

    Exemplo:
        record_predictions(perfis, [0.2], 0.01)
    """
    PREDICTION_DURATION.observe(duration)
    PREDICTIONS_TOTAL.inc(len(scores))
    for score in scores:
        PREDICTION_CONFIDENCE.observe(max(float(score), 1 - float(score)))
    record_feature_metrics(frame)
    LOGGER.info("predicao_gerada", extra={"duracao_ms": duration * 1_000})


def record_feature_metrics(frame) -> None:
    """Mantém as distribuições operacionais; os testes de drift são separados.

    Exemplo:
        record_feature_metrics(perfis)
    """
    metrics = {
        "AGE_YEARS": REFERENCE_MODEL_AGE_YEARS,
        "AMT_CREDIT": REFERENCE_MODEL_AMT_CREDIT,
        "AMT_INCOME_TOTAL": REFERENCE_MODEL_AMT_INCOME_TOTAL,
    }
    for column, metric in metrics.items():
        for value in frame[column]:
            metric.labels(column).observe(float(value))


@router.get("/income-types")
def income_types(loaded_predictor: PredictorDependency) -> list[str]:
    """Lista tipos de renda conhecidos pelo encoder do modelo carregado.

    Exemplo:
        values = income_types(predictor)
    """
    return loaded_predictor.get_categorical_column_values("NAME_INCOME_TYPE")


@router.get("/education-types")
def education_types(loaded_predictor: PredictorDependency) -> list[str]:
    """Lista escolaridades conhecidas pelo encoder do modelo carregado.

    Exemplo:
        values = education_types(predictor)
    """
    return loaded_predictor.get_categorical_column_values("NAME_EDUCATION_TYPE")


@router.get("/housing-types")
def housing_types(loaded_predictor: PredictorDependency) -> list[str]:
    """Lista tipos de moradia conhecidos pelo encoder do modelo carregado.

    Exemplo:
        values = housing_types(predictor)
    """
    return loaded_predictor.get_categorical_column_values("NAME_HOUSING_TYPE")


def create_app(
    predictor: ModelPredictor | None = None,
    model_version: str = "unavailable",
    model_loader: ModelLoader | None = load_model,
) -> FastAPI:
    """Monta rotas e métricas sem carregar modelos durante a importação.

    Exemplo:
        application = create_app(predictor=predictor, model_version="4")
    """
    application = FastAPI(title="Credit Monitoring API", lifespan=lifespan)
    application.state.predictor = predictor
    application.state.model_version = model_version
    application.state.model_loader = model_loader
    application.include_router(router)
    application.add_middleware(LoggingMiddleware)
    Instrumentator().instrument(application).expose(application, endpoint="/metrics")
    return application


app = create_app()
