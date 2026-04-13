from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class MatchFixture(BaseModel):
    match_id: str
    region: str
    event_name: str
    event_stage: Optional[str] = None
    team_a: str
    team_b: str
    match_date: date
    best_of: int = Field(default=3, ge=1, le=5)


class PredictionRequest(BaseModel):
    fixtures: list[MatchFixture]


class MapPrediction(BaseModel):
    map_name: str
    team_a_win_probability: float = Field(ge=0.0, le=1.0)
    projected_score: str
    picked_by: Optional[str] = None


class PlayerProjection(BaseModel):
    player_name: str
    team_name: str
    map_name: Optional[str] = None
    agent_name: Optional[str] = None
    projected_kills: float
    projected_deaths: float


class MatchPrediction(BaseModel):
    match_id: str
    team_a: str
    team_b: str
    region: Optional[str] = None
    event_name: Optional[str] = None
    event_stage: Optional[str] = None
    match_date: Optional[date] = None
    team_a_match_win_probability: float = Field(ge=0.0, le=1.0)
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    prediction_generated_at: Optional[str] = None
    sample_size: Optional[int] = Field(default=None, ge=0)
    map_predictions: list[MapPrediction]
    player_projections: list[PlayerProjection]
    model_version: str


class PredictionResponse(BaseModel):
    predictions: list[MatchPrediction]


class HealthResponse(BaseModel):
    status: str
    app_env: str
    prediction_mode: str
    model_version: str


class WeeklyRunSummary(BaseModel):
    run_at: str
    status: str
    prediction_mode: str
    model_version: str
    records_fetched: int
    tier1_records: int
    compared_matches: int
    winner_accuracy: float = Field(ge=0.0, le=1.0)
    artifact_path: str


class PredictionSnapshot(BaseModel):
    generated_at: str
    source: str
    prediction_mode: str
    model_version: str
    predictions: list[MatchPrediction]


class ModelPerformanceResponse(BaseModel):
    last_run_at: Optional[str] = None
    prediction_mode: str
    model_version: str
    upcoming_prediction_count: int = Field(ge=0)
    compared_matches: int = Field(ge=0)
    winner_accuracy: float = Field(ge=0.0, le=1.0)
    map_accuracy: float = Field(ge=0.0, le=1.0)
    player_kd_mae: float = Field(ge=0.0)
    tier1_results_count: int = Field(ge=0)
    training_samples: int = Field(ge=0)
    map_training_samples: int = Field(ge=0)
    player_training_samples: int = Field(ge=0)
    cv_accuracy: float = Field(ge=0.0, le=1.0)
    map_cv_accuracy: float = Field(ge=0.0, le=1.0)
    player_cv_score: float = Field(ge=0.0)
    rolling_winner_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    rolling_map_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    rolling_player_kd_mae: float = Field(default=0.0, ge=0.0)
    match_estimator: Optional[str] = None
    map_estimator: Optional[str] = None
    player_kills_estimator: Optional[str] = None
    player_deaths_estimator: Optional[str] = None
    match_calibration: Optional[str] = None
    map_calibration: Optional[str] = None
    excluded_match_rows: int = Field(default=0, ge=0)
    excluded_map_rows: int = Field(default=0, ge=0)
    excluded_player_rows: int = Field(default=0, ge=0)
    candidate_leaderboards: dict[str, list[dict]] = Field(default_factory=dict)
    source: str


class ValidationReportResponse(BaseModel):
    generated_at: str
    source: str
    compared_matches: int = Field(ge=0)
    winner_accuracy: float = Field(ge=0.0, le=1.0)
    compared_maps: int = Field(ge=0)
    map_accuracy: float = Field(ge=0.0, le=1.0)
    player_rows: int = Field(ge=0)
    player_kd_mae: float = Field(ge=0.0)
    rolling_windows_evaluated: int = Field(default=0, ge=0)
    integrity_issue_count: int = Field(default=0, ge=0)
    calibration_summary: dict[str, str] = Field(default_factory=dict)
    rolling_winner_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    rolling_map_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    rolling_player_kd_mae: float = Field(default=0.0, ge=0.0)
    status: str
