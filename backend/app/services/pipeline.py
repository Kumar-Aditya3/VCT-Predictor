from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import re

from app.core.config import get_settings
from app.models.schemas import (
    MatchFixture,
    MatchPrediction,
    MapPrediction,
    ModelPerformanceResponse,
    PlayerProjection,
    PredictionResponse,
    PredictionSnapshot,
    ValidationReportResponse,
    WeeklyRunSummary,
)
from app.services.modeling import (
    load_model_bundle,
    predict_map_probability,
    predict_match_probability,
    predict_player_stat_lines,
    save_model_bundle,
    select_maps_for_fixture,
    train_prediction_bundle,
)
from app.services.integrity import validate_match_details
from app.services.storage import SQLiteStore
from app.services.tier1_scope import assert_tier1_scope
from app.services.vlr_client import VLRClient, filter_tier1_fixtures, filter_tier1_records
from app.services.vlr_validation import validation_report_from_bundle


MODEL_VERSION = "vlr-core-unavailable"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _artifacts_dir() -> Path:
    settings = get_settings()
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return settings.artifacts_dir


def _run_artifact_path(run_at: datetime) -> Path:
    return _artifacts_dir() / f"pipeline_run_{run_at.strftime('%Y%m%d_%H%M%S')}.json"


def _default_upcoming_fixtures(today: date | None = None) -> list[MatchFixture]:
    fixture_date = today or date.today()
    return [
        MatchFixture(match_id="bootstrap-pac-001", region="Pacific", event_name="Masters", event_stage="Group Stage", team_a="Paper Rex", team_b="Gen.G", match_date=fixture_date, best_of=3),
        MatchFixture(match_id="bootstrap-emea-014", region="EMEA", event_name="Split 1", event_stage="Week 2", team_a="FNATIC", team_b="Team Heretics", match_date=fixture_date + timedelta(days=1), best_of=3),
    ]


def _clamp_probability(raw_probability: float) -> float:
    return max(0.08, min(0.92, raw_probability))


def _normalize_team_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _build_team_alias_index(model_bundle: dict | None) -> dict[str, str]:
    if model_bundle is None:
        return {}
    context = model_bundle.get("context", {})
    team_names = set(context.get("team_states", {}).keys())
    for player_state in context.get("player_states", {}).values():
        team_name = getattr(player_state, "team_name", None)
        if team_name:
            team_names.add(team_name)

    alias_index: dict[str, str] = {}
    collisions: set[str] = set()
    for team_name in team_names:
        token = _normalize_team_token(team_name)
        if not token:
            continue
        existing = alias_index.get(token)
        if existing is None:
            alias_index[token] = team_name
            continue
        if existing != team_name:
            collisions.add(token)

    for token in collisions:
        alias_index.pop(token, None)

    return alias_index


def _canonicalize_team_name(team_name: str, alias_index: dict[str, str]) -> str:
    cleaned = team_name.strip()
    if not cleaned:
        return team_name
    token = _normalize_team_token(cleaned)
    if not token:
        return cleaned
    return alias_index.get(token, cleaned)


def _canonicalize_fixture_teams(fixture: MatchFixture, alias_index: dict[str, str]) -> MatchFixture:
    team_a = _canonicalize_team_name(fixture.team_a, alias_index)
    team_b = _canonicalize_team_name(fixture.team_b, alias_index)
    if team_a == fixture.team_a and team_b == fixture.team_b:
        return fixture
    return fixture.model_copy(update={"team_a": team_a, "team_b": team_b})


def _projected_score(probability: float, map_index: int) -> str:
    margin = abs(probability - 0.5) * 2.0
    loser_rounds = int(round(12 - (margin * 5)))
    loser_rounds = max(7, min(12, loser_rounds))
    return f"13-{loser_rounds}" if probability >= 0.5 else f"{loser_rounds}-13"


def _projected_maps_played(team_a_match_win_probability: float, best_of: int) -> int:
    edge = abs(team_a_match_win_probability - 0.5)
    if best_of <= 1:
        return 1
    if best_of == 3:
        # Strongly favored sides are more likely to close in two maps.
        return 2 if edge >= 0.14 else 3
    if best_of == 5:
        if edge >= 0.18:
            return 3
        if edge >= 0.1:
            return 4
        return 5
    return min(best_of, 3)


def _current_model_version() -> str:
    model_bundle = load_model_bundle()
    return model_bundle["model_version"] if model_bundle is not None else MODEL_VERSION


def _current_prediction_mode() -> str:
    return "trained_ml" if load_model_bundle() is not None else "bootstrap"


def _prediction_metadata(model_bundle: dict | None, team_a_match_win_probability: float) -> dict[str, object | None]:
    if model_bundle is None:
        return {
            "confidence_score": None,
            "prediction_generated_at": None,
            "sample_size": None,
        }

    metrics = model_bundle.get("metrics", {})
    accuracy = float(metrics.get("winner_accuracy", 0.0))
    margin = abs(team_a_match_win_probability - 0.5) * 2.0
    confidence_score = round(max(0.0, min(1.0, (accuracy + margin) / 2.0)), 3)
    return {
        "confidence_score": confidence_score,
        "prediction_generated_at": model_bundle.get("trained_at"),
        "sample_size": int(metrics.get("training_samples", 0) or 0),
    }


def _build_match_prediction(fixture: MatchFixture, model_bundle: dict | None) -> MatchPrediction:
    assert_tier1_scope(fixture.region, fixture.event_name)
    if model_bundle is not None:
        team_a_match_win_probability = round(_clamp_probability(predict_match_probability(fixture, model_bundle)), 3)
        selected_maps = select_maps_for_fixture(fixture, model_bundle)
        projected_maps = _projected_maps_played(team_a_match_win_probability, fixture.best_of)
        map_predictions = []
        for index, (map_name, picked_by) in enumerate(selected_maps[:projected_maps]):
            map_probability = round(_clamp_probability(predict_map_probability(fixture, map_name, picked_by, model_bundle)), 3)
            map_predictions.append(
                MapPrediction(
                    map_name=map_name,
                    team_a_win_probability=map_probability,
                    projected_score=_projected_score(map_probability, index),
                    picked_by=picked_by,
                )
            )
        player_projection_rows = predict_player_stat_lines(fixture, selected_maps[:projected_maps], model_bundle)
        player_projections = [
            PlayerProjection(
                player_name=row["player_name"],
                team_name=row["team_name"],
                map_name=row.get("map_name"),
                agent_name=row["agent_name"],
                projected_kills=row["projected_kills"],
                projected_deaths=row["projected_deaths"],
            )
            for row in player_projection_rows
        ]
        model_version = model_bundle["model_version"]
    else:
        seed = sum(ord(char) for char in (fixture.team_a + fixture.team_b + fixture.region + fixture.event_name) if char.isalnum())
        team_a_match_win_probability = round(_clamp_probability(0.4 + (seed % 25) / 50), 3)
        map_predictions = [
            MapPrediction(map_name=name, team_a_win_probability=round(_clamp_probability(team_a_match_win_probability + ((index + 1) * 0.02) - 0.02), 3), projected_score=_projected_score(team_a_match_win_probability, index), picked_by=None)
            for index, name in enumerate(("Ascent", "Bind", "Lotus"))
        ]
        player_projections = []
        model_version = MODEL_VERSION

    prediction_metadata = _prediction_metadata(model_bundle, team_a_match_win_probability)

    return MatchPrediction(
        match_id=fixture.match_id,
        team_a=fixture.team_a,
        team_b=fixture.team_b,
        region=fixture.region,
        event_name=fixture.event_name,
        event_stage=fixture.event_stage,
        match_date=fixture.match_date,
        team_a_match_win_probability=team_a_match_win_probability,
        confidence_score=prediction_metadata["confidence_score"],
        prediction_generated_at=prediction_metadata["prediction_generated_at"],
        sample_size=prediction_metadata["sample_size"],
        map_predictions=map_predictions,
        player_projections=player_projections,
        model_version=model_version,
    )


def predict_fixtures(fixtures: list[MatchFixture]) -> PredictionResponse:
    model_bundle = load_model_bundle()
    alias_index = _build_team_alias_index(model_bundle)
    predictions = [_build_match_prediction(_canonicalize_fixture_teams(fixture, alias_index), model_bundle) for fixture in fixtures]
    return PredictionResponse(predictions=predictions)


def get_upcoming_predictions() -> PredictionSnapshot:
    client = VLRClient()
    today = date.today()
    try:
        fixtures = filter_tier1_fixtures(client.fetch_upcoming_fixtures(from_date=today, to_date=today + timedelta(days=7)))
    except Exception:
        fixtures = []
    if not fixtures:
        latest = load_latest_snapshot()
        if latest is not None:
            return latest
        fixtures = _default_upcoming_fixtures(today)
        source = "bootstrap_fallback"
    else:
        source = "vlr_schedule"
    predictions = predict_fixtures(fixtures).predictions
    model_version = predictions[0].model_version if predictions else _current_model_version()
    return PredictionSnapshot(
        generated_at=_utc_now().isoformat(),
        source=source,
        prediction_mode=_current_prediction_mode(),
        model_version=model_version,
        predictions=predictions,
    )


def run_weekly_update() -> WeeklyRunSummary:
    now = _utc_now()
    settings = get_settings()
    store = SQLiteStore()
    client = VLRClient()
    history_start = now.date() - timedelta(days=settings.training_history_days)

    fetched_records = filter_tier1_records(client.fetch_matches(from_date=history_start, to_date=now.date()))
    store.upsert_match_records(fetched_records)
    details = client.fetch_match_details(fetched_records)
    integrity_result = validate_match_details(details)
    clean_details = integrity_result["details"]
    store.record_scrape_issues(now.isoformat(), integrity_result["scrape_issues"])
    store.record_training_exclusions(now.isoformat(), integrity_result["training_exclusions"])
    store.upsert_match_details(clean_details)

    matches = store.load_matches()
    maps = store.load_maps()
    player_stats = store.load_player_stats()
    previous_bundle = load_model_bundle()
    model_bundle = train_prediction_bundle(matches, maps, player_stats)
    publish_note = "trained_new_bundle"
    if previous_bundle is not None and model_bundle is not None and not _should_publish_bundle(model_bundle, previous_bundle):
        publish_note = "trained_new_bundle_below_guardrail"
    model_path = save_model_bundle(model_bundle) if model_bundle is not None else None

    try:
        upcoming_fixtures = filter_tier1_fixtures(client.fetch_upcoming_fixtures(from_date=now.date(), to_date=now.date() + timedelta(days=7)))
    except Exception:
        upcoming_fixtures = []
    prediction_fixtures = upcoming_fixtures or _default_upcoming_fixtures(now.date())
    prediction_response = predict_fixtures(prediction_fixtures)
    model_version = prediction_response.predictions[0].model_version if prediction_response.predictions else _current_model_version()
    prediction_mode = _current_prediction_mode()
    validation = validation_report_from_bundle(model_bundle, now.isoformat(), "weekly_pipeline")
    validation = validation.model_copy(
        update={
            "integrity_issue_count": len(integrity_result["scrape_issues"]),
        }
    )

    snapshot = PredictionSnapshot(
        generated_at=now.isoformat(),
        source="weekly_pipeline",
        prediction_mode=prediction_mode,
        model_version=model_version,
        predictions=prediction_response.predictions,
    )
    summary = WeeklyRunSummary(
        run_at=now.isoformat(),
        status="ok" if model_bundle is not None else "degraded",
        prediction_mode=prediction_mode,
        model_version=model_version,
        records_fetched=len(fetched_records),
        tier1_records=len(matches),
        compared_matches=validation.compared_matches,
        winner_accuracy=validation.winner_accuracy,
        artifact_path="",
    )
    artifact_path = _run_artifact_path(now)
    summary.artifact_path = str(artifact_path)
    counts = store.counts()
    feature_summary = {
        "excluded_match_rows": integrity_result["exclusion_counts"]["excluded_matches"],
        "excluded_map_rows": integrity_result["exclusion_counts"]["excluded_maps"],
        "excluded_player_rows": integrity_result["exclusion_counts"]["excluded_player_rows"],
        "scrape_issue_count": len(integrity_result["scrape_issues"]),
    }
    store.record_feature_run(summary.run_at, feature_summary)
    artifact_path.write_text(
        json.dumps(
            {
                "summary": summary.model_dump(mode="json"),
                "snapshot": snapshot.model_dump(mode="json"),
                "validation": validation.model_dump(mode="json"),
                "training": {
                    **(model_bundle["metrics"] if model_bundle is not None else {}),
                    "history_start": history_start.isoformat(),
                    "history_end": now.date().isoformat(),
                    "model_artifact_path": str(model_path) if model_path is not None else None,
                },
                "search": model_bundle.get("search", {}) if model_bundle is not None else {},
                "backtests": model_bundle.get("backtests", {}) if model_bundle is not None else {},
                "calibration": model_bundle.get("calibration", {}) if model_bundle is not None else {},
                "candidate_results": model_bundle.get("search", {}) if model_bundle is not None else {},
                "residuals": model_bundle.get("residuals", {}) if model_bundle is not None else {},
                "exclusions": feature_summary,
                "publish": {"decision": publish_note},
                "storage": counts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    store.record_pipeline_run(
        run_at=summary.run_at,
        artifact_path=summary.artifact_path,
        model_version=summary.model_version,
        prediction_mode=summary.prediction_mode,
        winner_accuracy=validation.winner_accuracy,
        map_accuracy=validation.map_accuracy,
        player_kd_mae=validation.player_kd_mae,
    )
    return summary


def _should_publish_bundle(candidate_bundle: dict, current_bundle: dict) -> bool:
    candidate_match = float(candidate_bundle.get("metrics", {}).get("winner_accuracy", 0.0))
    current_match = float(current_bundle.get("metrics", {}).get("winner_accuracy", 0.0))
    candidate_rolling = float(candidate_bundle.get("metrics", {}).get("rolling_winner_accuracy", 0.0))
    current_rolling = float(current_bundle.get("metrics", {}).get("rolling_winner_accuracy", 0.0))
    candidate_map = float(candidate_bundle.get("metrics", {}).get("map_accuracy", 0.0))
    candidate_player_mae = float(candidate_bundle.get("metrics", {}).get("player_kd_mae", float("inf")))
    if candidate_match < 0.67:
        return False
    if candidate_rolling < 0.65:
        return False
    if candidate_map < 0.54:
        return False
    if candidate_player_mae > 3.2:
        return False
    if candidate_match > current_match:
        return True
    if candidate_match == current_match and candidate_rolling >= current_rolling:
        return True
    return False


def get_model_performance() -> ModelPerformanceResponse:
    payload = _load_latest_artifact()
    if payload is None:
        snapshot = get_upcoming_predictions()
        return ModelPerformanceResponse(
            prediction_mode=snapshot.prediction_mode,
            model_version=snapshot.model_version,
            upcoming_prediction_count=len(snapshot.predictions),
            compared_matches=0,
            winner_accuracy=0.0,
            map_accuracy=0.0,
            player_kd_mae=0.0,
            tier1_results_count=0,
            training_samples=0,
            map_training_samples=0,
            player_training_samples=0,
            cv_accuracy=0.0,
            map_cv_accuracy=0.0,
            player_cv_score=0.0,
            rolling_winner_accuracy=0.0,
            rolling_map_accuracy=0.0,
            rolling_player_kd_mae=0.0,
            excluded_match_rows=0,
            excluded_map_rows=0,
            excluded_player_rows=0,
            candidate_leaderboards={},
            source=snapshot.source,
        )
    summary = payload["summary"]
    snapshot = payload["snapshot"]
    training = payload.get("training", {})
    validation = payload.get("validation", {})
    storage = payload.get("storage", {})
    exclusions = payload.get("exclusions", {})
    search = payload.get("search", {})
    return ModelPerformanceResponse(
        last_run_at=summary["run_at"],
        prediction_mode=summary["prediction_mode"],
        model_version=summary["model_version"],
        upcoming_prediction_count=len(snapshot["predictions"]),
        compared_matches=validation.get("compared_matches", 0),
        winner_accuracy=validation.get("winner_accuracy", 0.0),
        map_accuracy=validation.get("map_accuracy", 0.0),
        player_kd_mae=validation.get("player_kd_mae", 0.0),
        tier1_results_count=storage.get("matches", summary["tier1_records"]),
        training_samples=training.get("training_samples", 0),
        map_training_samples=training.get("map_training_samples", 0),
        player_training_samples=training.get("player_training_samples", 0),
        cv_accuracy=training.get("cv_accuracy", 0.0),
        map_cv_accuracy=training.get("map_cv_accuracy", 0.0),
        player_cv_score=training.get("player_cv_score", 0.0),
        rolling_winner_accuracy=training.get("rolling_winner_accuracy", 0.0),
        rolling_map_accuracy=training.get("rolling_map_accuracy", 0.0),
        rolling_player_kd_mae=training.get("rolling_player_kd_mae", 0.0),
        match_estimator=training.get("match_estimator"),
        map_estimator=training.get("map_estimator"),
        player_kills_estimator=training.get("player_kills_estimator"),
        player_deaths_estimator=training.get("player_deaths_estimator"),
        match_calibration=training.get("match_calibration"),
        map_calibration=training.get("map_calibration"),
        excluded_match_rows=exclusions.get("excluded_match_rows", 0),
        excluded_map_rows=exclusions.get("excluded_map_rows", 0),
        excluded_player_rows=exclusions.get("excluded_player_rows", 0),
        candidate_leaderboards={
            "match": search.get("match_candidates", [])[:5],
            "map": search.get("map_candidates", [])[:5],
            "player_kills": search.get("player_kills_candidates", [])[:5],
            "player_deaths": search.get("player_deaths_candidates", [])[:5],
        },
        source=snapshot["source"],
    )


def get_validation_report() -> ValidationReportResponse:
    payload = _load_latest_artifact()
    if payload is not None and "validation" in payload:
        return ValidationReportResponse.model_validate(_normalize_validation_payload(payload["validation"]))
    now = _utc_now().isoformat()
    return ValidationReportResponse(
        generated_at=now,
        source="none",
        compared_matches=0,
        winner_accuracy=0.0,
        compared_maps=0,
        map_accuracy=0.0,
        player_rows=0,
        player_kd_mae=0.0,
        status="not_run",
    )


def load_latest_snapshot() -> PredictionSnapshot | None:
    payload = _load_latest_artifact()
    if payload is None or "snapshot" not in payload:
        return None
    return PredictionSnapshot.model_validate(payload["snapshot"])


def _load_latest_artifact() -> dict | None:
    latest_path = _latest_artifact_path()
    if latest_path is None:
        return None
    return json.loads(latest_path.read_text(encoding="utf-8"))


def _latest_artifact_path() -> Path | None:
    candidates = sorted(_artifacts_dir().glob("pipeline_run_*.json"), reverse=True)
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if {"summary", "snapshot", "validation"}.issubset(payload.keys()):
            return candidate
    return None


def _normalize_validation_payload(payload: dict) -> dict:
    return {
        "generated_at": payload.get("generated_at", _utc_now().isoformat()),
        "source": payload.get("source", "artifact"),
        "compared_matches": int(payload.get("compared_matches", 0)),
        "winner_accuracy": float(payload.get("winner_accuracy", 0.0)),
        "compared_maps": int(payload.get("compared_maps", 0)),
        "map_accuracy": float(payload.get("map_accuracy", 0.0)),
        "player_rows": int(payload.get("player_rows", 0)),
        "player_kd_mae": float(payload.get("player_kd_mae", 0.0)),
        "rolling_windows_evaluated": int(payload.get("rolling_windows_evaluated", 0)),
        "integrity_issue_count": int(payload.get("integrity_issue_count", 0)),
        "calibration_summary": payload.get("calibration_summary", {}),
        "rolling_winner_accuracy": float(payload.get("rolling_winner_accuracy", 0.0)),
        "rolling_map_accuracy": float(payload.get("rolling_map_accuracy", 0.0)),
        "rolling_player_kd_mae": float(payload.get("rolling_player_kd_mae", 0.0)),
        "status": payload.get("status", "warn"),
    }
