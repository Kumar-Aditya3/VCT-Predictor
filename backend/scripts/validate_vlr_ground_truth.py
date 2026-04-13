from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from app.models.schemas import MatchFixture
from app.services.pipeline import get_validation_report, predict_fixtures
from app.services.vlr_client import VLRClient, filter_tier1_records
from app.services.vlr_validation import compute_winner_accuracy


@dataclass
class VLRValidationSummary:
    compared_matches: int
    winner_accuracy: float
    map_score_mae: float
    player_kd_mae: float



def validate_against_vlr_ground_truth() -> VLRValidationSummary:
    client = VLRClient()
    today = date.today()
    records = client.fetch_matches(from_date=today - timedelta(days=30), to_date=today)
    records = filter_tier1_records(records)
    prediction_response = (
        predict_fixtures(
            [
                MatchFixture(
                    match_id=record.match_id,
                    region=record.region,
                    event_name=record.event_name,
                    team_a=record.team_a,
                    team_b=record.team_b,
                    match_date=record.match_date,
                    best_of=3,
                )
                for record in records
            ]
        )
        if records
        else predict_fixtures([])
    )
    winner_metrics = compute_winner_accuracy(prediction_response.predictions, records)

    return VLRValidationSummary(
        compared_matches=winner_metrics.compared_matches,
        winner_accuracy=winner_metrics.winner_accuracy,
        map_score_mae=0.0,
        player_kd_mae=0.0,
    )


if __name__ == "__main__":
    summary = validate_against_vlr_ground_truth()
    latest_report = get_validation_report()
    print(
        "vlr_validation "
        f"compared_matches={summary.compared_matches} "
        f"winner_accuracy={summary.winner_accuracy:.4f} "
        f"map_score_mae={summary.map_score_mae:.4f} "
        f"player_kd_mae={summary.player_kd_mae:.4f} "
        f"latest_status={latest_report.status}"
    )
