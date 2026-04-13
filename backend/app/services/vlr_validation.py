from __future__ import annotations

from dataclasses import dataclass

from app.models.schemas import MatchPrediction, ValidationReportResponse
from app.models.vlr import VLRMatchRecord


@dataclass
class VLRMetricBundle:
    compared_matches: int
    winner_accuracy: float
    compared_maps: int
    map_accuracy: float
    player_rows: int
    player_kd_mae: float


def compute_winner_accuracy(predictions: list[MatchPrediction], truth: list[VLRMatchRecord]) -> VLRMetricBundle:
    truth_by_id = {row.match_id: row for row in truth}
    compared = 0
    correct = 0
    for prediction in predictions:
        row = truth_by_id.get(prediction.match_id)
        if row is None:
            continue
        compared += 1
        predicted_a_wins = prediction.team_a_match_win_probability >= 0.5
        actual_a_wins = row.team_a_maps_won > row.team_b_maps_won
        if predicted_a_wins == actual_a_wins:
            correct += 1
    return VLRMetricBundle(
        compared_matches=compared,
        winner_accuracy=(correct / compared) if compared else 0.0,
        compared_maps=0,
        map_accuracy=0.0,
        player_rows=0,
        player_kd_mae=0.0,
    )


def validation_report_from_bundle(model_bundle: dict | None, generated_at: str, source: str) -> ValidationReportResponse:
    if model_bundle is None:
        return ValidationReportResponse(
            generated_at=generated_at,
            source=source,
            compared_matches=0,
            winner_accuracy=0.0,
            compared_maps=0,
            map_accuracy=0.0,
            player_rows=0,
            player_kd_mae=0.0,
            rolling_windows_evaluated=0,
            integrity_issue_count=0,
            calibration_summary={},
            rolling_winner_accuracy=0.0,
            rolling_map_accuracy=0.0,
            rolling_player_kd_mae=0.0,
            status="not_run",
        )

    metrics = model_bundle["metrics"]
    winner_accuracy = float(metrics.get("winner_accuracy", 0.0))
    map_accuracy = float(metrics.get("map_accuracy", 0.0))
    player_kd_mae = float(metrics.get("player_kd_mae", 0.0))
    status = "pass" if winner_accuracy >= 0.58 and map_accuracy >= 0.5 else "warn"
    return ValidationReportResponse(
        generated_at=generated_at,
        source=source,
        compared_matches=int(metrics.get("compared_matches", 0)),
        winner_accuracy=winner_accuracy,
        compared_maps=int(metrics.get("compared_maps", 0)),
        map_accuracy=map_accuracy,
        player_rows=int(metrics.get("player_rows", 0)),
        player_kd_mae=player_kd_mae,
        rolling_windows_evaluated=int(model_bundle.get("backtests", {}).get("rolling_windows_evaluated", 0)),
        integrity_issue_count=0,
        calibration_summary={
            "match": str(model_bundle.get("calibration", {}).get("match", "none")),
            "map": str(model_bundle.get("calibration", {}).get("map", "none")),
        },
        rolling_winner_accuracy=float(metrics.get("rolling_winner_accuracy", 0.0)),
        rolling_map_accuracy=float(metrics.get("rolling_map_accuracy", 0.0)),
        rolling_player_kd_mae=float(metrics.get("rolling_player_kd_mae", 0.0)),
        status=status,
    )
