from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.models.schemas import (
    HealthResponse,
    ModelPerformanceResponse,
    PredictionRequest,
    PredictionResponse,
    PredictionSnapshot,
    ValidationReportResponse,
    WeeklyRunSummary,
)
from app.services.pipeline import (
    get_model_performance,
    get_upcoming_predictions,
    get_validation_report,
    predict_fixtures,
    run_weekly_update,
    _current_model_version,
    _current_prediction_mode,
)
from app.services.tier1_scope import whitelisted_scope

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        app_env=settings.app_env,
        prediction_mode=_current_prediction_mode(),
        model_version=_current_model_version(),
    )


@router.get("/scope")
def scope() -> dict[str, tuple[str, ...]]:
    data = whitelisted_scope()
    return {
        "regions": tuple(data["regions"]),
        "events": tuple(data["events"]),
    }


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        return predict_fixtures(payload.fixtures)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/predictions/upcoming", response_model=PredictionSnapshot)
def upcoming_predictions() -> PredictionSnapshot:
    return get_upcoming_predictions()


@router.get("/model/performance", response_model=ModelPerformanceResponse)
def model_performance() -> ModelPerformanceResponse:
    return get_model_performance()


@router.get("/data/validation", response_model=ValidationReportResponse)
def data_validation() -> ValidationReportResponse:
    return get_validation_report()


@router.post("/pipeline/weekly", response_model=WeeklyRunSummary)
def weekly_pipeline() -> WeeklyRunSummary:
    return run_weekly_update()
