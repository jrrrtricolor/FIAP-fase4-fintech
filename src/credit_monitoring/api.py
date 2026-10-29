"""API REST para inferência com o Pipeline aprovado no MLflow."""

import logging
import os
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import monotonic
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
    API_READY,
    MODEL_INFO,
    MODEL_RELOAD_ERROR,
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
ModelSnapshot = tuple[ModelPredictor, str]
ModelLoader = Callable[[], ModelSnapshot]
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


def load_model() -> ModelSnapshot:
    """Carrega a versão aprovada e reutiliza seu artefato imutável.

    Exemplo:
        predictor, version = load_model()
    """
    model_uri, version = resolve_approved_model()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    return cached_predictor(model_uri, tracking_uri), version


def resolve_approved_model(version=None):
    """Confere somente metadados, usando o alias ou a versão numérica informada.

    Exemplo:
        uri, version = resolve_approved_model(version="1")
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    model_uri = os.getenv("MODEL_URI", DEFAULT_MODEL_URI)
    if version is not None:
        separator = "@" if "@" in model_uri else "/"
        model_uri = f"{model_uri.rsplit(separator, 1)[0]}/{version}"
    return ModelPredictor.resolve_approved_mlflow_uri(model_uri, tracking_uri)


@lru_cache(maxsize=2)
def cached_predictor(model_uri, tracking_uri):
    """Mantém no máximo dois artefatos, identificados por versão e backend.

    Exemplo:
        predictor = cached_predictor("models:/credito/2", "sqlite:///mlflow.db")
    """
    return ModelPredictor.from_mlflow(model_uri=model_uri, tracking_uri=tracking_uri)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Carrega o modelo; falhas ficam nos logs e na prontidão HTTP.

    Exemplo:
        application = FastAPI(lifespan=lifespan)
    """
    setup_api_logger()
    refresh_model(application)
    publish_model_metrics(application)
    yield


def refresh_model(application):
    """Tenta atualizar periodicamente sem bloquear requisições com modelo válido.

    Exemplo:
        refresh_model(application)
    """
    state = application.state
    now = monotonic()
    if state.model_loader is None or now < state.next_refresh:
        return
    if not state.model_lock.acquire(blocking=False):
        return
    try:
        if now >= state.next_refresh:
            state.next_refresh = now + state.refresh_seconds
            replace_model(application)
    finally:
        state.model_lock.release()


def replace_model(application):
    """Troca após a carga completa e preserva somente um snapshot ainda aprovado.

    Exemplo:
        replace_model(application)
    """
    state = application.state
    try:
        state.active_model = state.model_loader()
        state.reload_error = False
    except Exception:
        state.reload_error = True
        LOGGER.exception("falha_ao_carregar_modelo")
        discard_revoked_model(application)
    publish_model_metrics(application)


def discard_revoked_model(application):
    """Interrompe o uso de uma versão cuja aprovação foi comprovadamente retirada.

    Exemplo:
        discard_revoked_model(application)
    """
    state = application.state
    if state.model_loader is not load_model or state.active_model[0] is None:
        return
    try:
        resolve_approved_model(state.active_model[1])
    except Exception as error:
        revoked = isinstance(error, ValueError)
        missing = getattr(error, "error_code", None) == "RESOURCE_DOES_NOT_EXIST"
        if revoked or missing:
            state.active_model = (None, "unavailable")
            LOGGER.error("modelo_ativo_sem_aprovacao")


def publish_model_metrics(application):
    """Publica versão ativa, prontidão e erro de recarga a partir do mesmo estado.

    Exemplo:
        publish_model_metrics(application)
    """
    state = application.state
    predictor, version = state.active_model
    MODEL_INFO.clear()
    if predictor is not None:
        MODEL_INFO.labels(version=version).set(1)
    API_READY.set(predictor is not None)
    MODEL_RELOAD_ERROR.set(state.reload_error)


def require_predictor(request: Request) -> ModelSnapshot:
    """Captura modelo e versão juntos, recusando inferência sem snapshot válido.

    Exemplo:
        predictor, version = require_predictor(request)
    """
    refresh_model(request.app)
    snapshot = request.app.state.active_model
    if snapshot[0] is None:
        raise HTTPException(status_code=503, detail="Modelo indisponível.")
    return snapshot


PredictorDependency = Annotated[ModelSnapshot, Depends(require_predictor)]


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
    refresh_model(request.app)
    predictor, version = request.app.state.active_model
    is_ready = predictor is not None
    response.status_code = 200 if is_ready else 503
    return {"ready": is_ready, "model_version": version}


@router.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest, loaded_model: PredictorDependency
) -> PredictionResponse:
    """Calcula uma vez os scores de inadimplência dos perfis validados.

    Exemplo:
        result = predict(payload, (predictor, "2"))
    """
    started_at = time.perf_counter()
    loaded_predictor, version = loaded_model
    scores = loaded_predictor.positive_scores(payload.prediction_data)
    duration = time.perf_counter() - started_at
    record_predictions(payload.prediction_data, scores, duration)
    return PredictionResponse(
        credit_scores=scores,
        model_version=version,
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
        values = income_types((predictor, "2"))
    """
    return loaded_predictor[0].get_categorical_column_values("NAME_INCOME_TYPE")


@router.get("/education-types")
def education_types(loaded_predictor: PredictorDependency) -> list[str]:
    """Lista escolaridades conhecidas pelo encoder do modelo carregado.

    Exemplo:
        values = education_types((predictor, "2"))
    """
    return loaded_predictor[0].get_categorical_column_values("NAME_EDUCATION_TYPE")


@router.get("/housing-types")
def housing_types(loaded_predictor: PredictorDependency) -> list[str]:
    """Lista tipos de moradia conhecidos pelo encoder do modelo carregado.

    Exemplo:
        values = housing_types((predictor, "2"))
    """
    return loaded_predictor[0].get_categorical_column_values("NAME_HOUSING_TYPE")


def configure_model_state(application, predictor, version, loader):
    """Prepara snapshot e sincronização sem carregar artefatos na importação.

    Exemplo:
        configure_model_state(application, None, "unavailable", load_model)
    """
    state = application.state
    state.active_model = (predictor, version)
    state.model_loader = loader if predictor is None else None
    state.model_lock = Lock()
    state.next_refresh = 0.0
    state.refresh_seconds = 30.0
    state.reload_error = False


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
    configure_model_state(application, predictor, model_version, model_loader)
    application.include_router(router)
    application.add_middleware(LoggingMiddleware)
    Instrumentator().instrument(application).expose(application, endpoint="/metrics")
    return application


app = create_app()
