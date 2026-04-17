import os
from collections import deque
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings
from app.models.schemas import MatchFixture, MatchPrediction, PredictionSnapshot
from app.models.vlr import VLRMatchDetails, VLRMatchRecord
from app.services.modeling import (
    PlayerState,
    TeamState,
    _likely_lineup,
    _plasticity_adjustment,
    load_model_bundle,
    predict_match_probability,
    train_prediction_bundle,
)
from app.services import pipeline as pipeline_service
from app.services.pipeline import get_model_performance, get_validation_report, predict_fixtures, run_weekly_update
from app.services.storage import SQLiteStore
from app.services.vlr_client import extract_tier1_metadata
from app.services.vlr_validation import compute_winner_accuracy
from tests.synthetic_data import generate_synthetic_vct_dataset


def build_training_records(count: int = 30) -> list[VLRMatchRecord]:
    return generate_synthetic_vct_dataset(match_count=count).records


def build_training_details(records: list[VLRMatchRecord]) -> list[VLRMatchDetails]:
    return generate_synthetic_vct_dataset(match_count=len(records)).details


class PipelineServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["ARTIFACTS_DIR"] = self.temp_dir.name
        os.environ["MODEL_ARTIFACTS_DIR"] = str(Path(self.temp_dir.name) / "models")
        os.environ["SQLITE_DB_PATH"] = str(Path(self.temp_dir.name) / "vct.sqlite3")
        os.environ["FAST_MODEL_SEARCH"] = "1"
        get_settings.cache_clear()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()
        for key in ("ARTIFACTS_DIR", "MODEL_ARTIFACTS_DIR", "SQLITE_DB_PATH", "FAST_MODEL_SEARCH"):
            os.environ.pop(key, None)
        get_settings.cache_clear()

    def test_train_prediction_bundle_builds_bundle(self) -> None:
        records = build_training_records()
        details = build_training_details(records)
        maps = [item for detail in details for item in detail.maps]
        player_stats = [item for detail in details for item in detail.player_stats]
        bundle = train_prediction_bundle(records, maps, player_stats)
        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertTrue(bundle["model_version"].startswith("vlr-core-"))
        self.assertGreater(bundle["metrics"]["map_training_samples"], 0)

    def test_predict_fixtures_uses_trained_model_when_available(self) -> None:
        records = build_training_records()
        details = build_training_details(records)
        bundle = train_prediction_bundle(records, [item for detail in details for item in detail.maps], [item for detail in details for item in detail.player_stats])
        assert bundle is not None
        with patch("app.services.pipeline.load_model_bundle", return_value=bundle):
            fixtures = [
                MatchFixture(match_id="sample-1", region="Pacific", event_name="Masters", event_stage="Group Stage", team_a="Alpha 1", team_b="Beta 1", match_date="2026-04-08", best_of=3)
            ]
            prediction = predict_fixtures(fixtures).predictions[0]
        self.assertTrue(prediction.model_version.startswith("vlr-core-"))
        self.assertGreater(len(prediction.player_projections), 0)
        self.assertIsNotNone(prediction.confidence_score)
        self.assertIsNotNone(prediction.prediction_generated_at)
        self.assertGreater(prediction.sample_size or 0, 0)

    def test_predict_fixtures_generates_player_rows_without_player_models(self) -> None:
        records = build_training_records()
        details = build_training_details(records)
        bundle = train_prediction_bundle(records, [item for detail in details for item in detail.maps], [item for detail in details for item in detail.player_stats])
        assert bundle is not None
        bundle["player_kills_model"] = None
        bundle["player_deaths_model"] = None

        with patch("app.services.pipeline.load_model_bundle", return_value=bundle):
            fixtures = [
                MatchFixture(match_id="sample-2", region="Pacific", event_name="Masters", event_stage="Group Stage", team_a="Alpha 1", team_b="Beta 1", match_date="2026-04-08", best_of=3)
            ]
            prediction = predict_fixtures(fixtures).predictions[0]

        self.assertGreater(len(prediction.player_projections), 0)
        self.assertTrue(all(player.map_name is not None for player in prediction.player_projections))

    def test_predict_match_probability_is_order_invariant(self) -> None:
        records = build_training_records()
        details = build_training_details(records)
        bundle = train_prediction_bundle(
            records,
            [item for detail in details for item in detail.maps],
            [item for detail in details for item in detail.player_stats],
        )
        assert bundle is not None
        fixture = MatchFixture(
            match_id="sample-1",
            region="Pacific",
            event_name="Masters",
            event_stage="Group Stage",
            team_a="Alpha 1",
            team_b="Beta 1",
            match_date="2026-04-08",
            best_of=3,
        )
        reverse_fixture = MatchFixture(
            match_id="sample-1",
            region="Pacific",
            event_name="Masters",
            event_stage="Group Stage",
            team_a="Beta 1",
            team_b="Alpha 1",
            match_date="2026-04-08",
            best_of=3,
        )
        probability = predict_match_probability(fixture, bundle)
        reverse_probability = predict_match_probability(reverse_fixture, bundle)
        self.assertAlmostEqual(probability, 1.0 - reverse_probability, places=6)

    def test_compute_winner_accuracy_handles_matches_and_misses(self) -> None:
        predictions = [
            MatchPrediction(match_id="match-1", team_a="Team Alpha", team_b="Team Beta", team_a_match_win_probability=0.6, map_predictions=[], player_projections=[], model_version="vlr-core-test"),
            MatchPrediction(match_id="missing", team_a="Team One", team_b="Team Two", team_a_match_win_probability=0.4, map_predictions=[], player_projections=[], model_version="vlr-core-test"),
        ]
        truth = [VLRMatchRecord(match_id="match-1", event_name="Masters", region="Pacific", match_date="2026-04-08", team_a="Team Alpha", team_b="Team Beta", team_a_maps_won=2, team_b_maps_won=1, best_of=3)]
        metrics = compute_winner_accuracy(predictions, truth)
        self.assertEqual(metrics.compared_matches, 1)
        self.assertEqual(metrics.winner_accuracy, 1.0)

    @patch("app.services.pipeline.VLRClient.fetch_match_details")
    @patch("app.services.pipeline.VLRClient.fetch_upcoming_fixtures")
    @patch("app.services.pipeline.VLRClient.fetch_matches")
    def test_weekly_update_persists_trained_model(self, mock_fetch_matches, mock_fetch_upcoming, mock_fetch_details) -> None:
        records = build_training_records()
        mock_fetch_matches.return_value = records
        mock_fetch_upcoming.return_value = []
        mock_fetch_details.return_value = build_training_details(records)
        summary = run_weekly_update()
        model_bundle = load_model_bundle()
        store = SQLiteStore()
        self.assertEqual(summary.status, "ok")
        self.assertIsNotNone(model_bundle)
        self.assertGreater(store.counts()["player_stats"], 0)
        self.assertGreater(get_model_performance().map_training_samples, 0)
        self.assertIn(get_validation_report().status, {"pass", "warn"})

    def test_extract_tier1_metadata_parses_vct_event_names(self) -> None:
        self.assertEqual(extract_tier1_metadata("Group Stage VCT 2026: Pacific Stage 1"), ("Split 1", "Pacific"))
        self.assertEqual(extract_tier1_metadata("Playoffs VCT 2026: China Stage 2"), ("Split 2", "China"))
        self.assertEqual(extract_tier1_metadata("VCT 2026 Masters - Grand Final"), ("Masters", "International"))
        self.assertEqual(extract_tier1_metadata("VCT 2026 Champions - Lower Final"), ("Champions", "International"))
        self.assertEqual(extract_tier1_metadata("VCT 2026 Kickoff - Showmatch"), (None, None))

    def test_synthetic_generator_produces_tier1_compatible_dataset(self) -> None:
        dataset = generate_synthetic_vct_dataset(match_count=12, seed=7)
        self.assertEqual(len(dataset.records), 12)
        self.assertEqual(len(dataset.details), 12)
        self.assertGreater(len(dataset.maps), 12)
        self.assertGreater(len(dataset.player_stats), 0)
        self.assertTrue(all(record.event_name in {"Kickoff", "Split 1", "Split 2", "Masters", "Champions"} for record in dataset.records))
        self.assertTrue(all(record.region in {"Americas", "EMEA", "Pacific", "China", "International"} for record in dataset.records))

    def test_publish_guardrails_require_thresholds(self) -> None:
        current_bundle = {
            "metrics": {
                "winner_accuracy": 0.7,
                "rolling_winner_accuracy": 0.66,
                "rolling_winner_log_loss": 0.34,
                "map_accuracy": 0.56,
                "rolling_map_log_loss": 0.29,
                "player_kd_mae": 3.1,
                "rolling_player_kd_mae": 3.1,
            }
        }
        passing_candidate = {
            "metrics": {
                "winner_accuracy": 0.71,
                "rolling_winner_accuracy": 0.66,
                "rolling_winner_log_loss": 0.32,
                "map_accuracy": 0.56,
                "rolling_map_log_loss": 0.285,
                "player_kd_mae": 3.0,
                "rolling_player_kd_mae": 3.0,
            }
        }
        failing_candidate = {
            "metrics": {
                "winner_accuracy": 0.66,
                "rolling_winner_accuracy": 0.66,
                "rolling_winner_log_loss": 0.36,
                "map_accuracy": 0.56,
                "rolling_map_log_loss": 0.30,
                "player_kd_mae": 3.0,
                "rolling_player_kd_mae": 3.0,
            }
        }
        self.assertTrue(pipeline_service._should_publish_bundle(passing_candidate, current_bundle))
        self.assertFalse(pipeline_service._should_publish_bundle(failing_candidate, current_bundle))

    def test_get_upcoming_predictions_refreshes_stale_snapshot_model_version(self) -> None:
        stale_snapshot = PredictionSnapshot(
            generated_at="2026-04-17T04:00:00+00:00",
            source="weekly_pipeline",
            prediction_mode="trained_ml",
            model_version="vlr-core-old",
            predictions=[
                MatchPrediction(
                    match_id="fixture-1",
                    team_a="Alpha 1",
                    team_b="Beta 1",
                    region="Pacific",
                    event_name="Masters",
                    event_stage="Group Stage",
                    match_date="2026-04-18",
                    team_a_match_win_probability=0.51,
                    map_predictions=[],
                    player_projections=[],
                    model_version="vlr-core-old",
                )
            ],
        )
        refreshed_prediction = MatchPrediction(
            match_id="fixture-1",
            team_a="Alpha 1",
            team_b="Beta 1",
            region="Pacific",
            event_name="Masters",
            event_stage="Group Stage",
            match_date="2026-04-18",
            team_a_match_win_probability=0.64,
            map_predictions=[],
            player_projections=[],
            model_version="vlr-core-new",
        )

        with (
            patch("app.services.pipeline.load_latest_snapshot", return_value=stale_snapshot),
            patch("app.services.pipeline._current_model_version", return_value="vlr-core-new"),
            patch("app.services.pipeline._current_prediction_mode", return_value="trained_ml"),
            patch("app.services.pipeline.predict_fixtures", return_value=pipeline_service.PredictionResponse(predictions=[refreshed_prediction])),
        ):
            snapshot = pipeline_service.get_upcoming_predictions()

        self.assertEqual(snapshot.model_version, "vlr-core-new")
        self.assertEqual(snapshot.source, "artifact_refresh")
        self.assertEqual(snapshot.predictions[0].team_a_match_win_probability, 0.64)

    def test_likely_lineup_excludes_players_on_other_current_team(self) -> None:
        context = {
            "team_recent_players": {
                "Paper Rex": {
                    "PatMen": 10.0,
                    "Jinggg": 9.0,
                    "d4v41": 8.0,
                    "something": 7.0,
                    "mindfreak": 6.0,
                    "f0rsakeN": 5.0,
                }
            },
            "player_states": {
                "PatMen": PlayerState(team_name="Global Esports"),
                "Jinggg": PlayerState(team_name="Paper Rex"),
                "d4v41": PlayerState(team_name="Paper Rex"),
                "something": PlayerState(team_name="Paper Rex"),
                "mindfreak": PlayerState(team_name="Paper Rex"),
                "f0rsakeN": PlayerState(team_name="Paper Rex"),
            },
        }

        lineup = _likely_lineup("Paper Rex", context)
        self.assertNotIn("PatMen", lineup)
        self.assertEqual(len(lineup), 5)

    def test_plasticity_adjustment_rewards_recent_improving_team(self) -> None:
        team_a_state = TeamState(matches=40, wins=18)
        team_a_state.recent_results = deque([1, 1, 1, 1, 1], maxlen=8)
        team_a_state.recent_map_margin = deque([6, 5, 4], maxlen=8)

        team_b_state = TeamState(matches=40, wins=24)
        team_b_state.recent_results = deque([0, 0, 1, 0, 0], maxlen=8)
        team_b_state.recent_map_margin = deque([-4, -5, -3], maxlen=8)

        context = {
            "team_states": {
                "Team A": team_a_state,
                "Team B": team_b_state,
            },
            "team_recent_players": {
                "Team A": {"A1": 5.0, "A2": 5.0, "A3": 5.0, "A4": 5.0, "A5": 5.0},
                "Team B": {"B1": 5.0, "B2": 5.0, "B3": 5.0, "B4": 5.0, "B5": 5.0},
            },
        }
        fixture = MatchFixture(
            match_id="plasticity-1",
            region="Pacific",
            event_name="Masters",
            event_stage="Playoffs",
            team_a="Team A",
            team_b="Team B",
            match_date="2026-04-17",
            best_of=3,
        )

        adjustment = _plasticity_adjustment(fixture, context, base_probability=0.46)
        self.assertGreater(adjustment, 0.0)

    def test_plasticity_adjustment_is_damped_for_large_base_edge(self) -> None:
        team_a_state = TeamState(matches=36, wins=15)
        team_a_state.recent_results = deque([1, 1, 1, 1, 1], maxlen=8)
        team_a_state.recent_map_margin = deque([5, 4, 6], maxlen=8)

        team_b_state = TeamState(matches=36, wins=23)
        team_b_state.recent_results = deque([0, 0, 1, 0, 0], maxlen=8)
        team_b_state.recent_map_margin = deque([-5, -4, -3], maxlen=8)

        context = {
            "team_states": {
                "Team A": team_a_state,
                "Team B": team_b_state,
            },
            "team_recent_players": {
                "Team A": {"A1": 6.0, "A2": 6.0, "A3": 6.0, "A4": 6.0, "A5": 6.0},
                "Team B": {"B1": 6.0, "B2": 6.0, "B3": 6.0, "B4": 6.0, "B5": 6.0},
            },
        }
        fixture = MatchFixture(
            match_id="plasticity-2",
            region="Pacific",
            event_name="Masters",
            event_stage="Playoffs",
            team_a="Team A",
            team_b="Team B",
            match_date="2026-04-17",
            best_of=3,
        )

        close_edge = abs(_plasticity_adjustment(fixture, context, base_probability=0.52))
        heavy_edge = abs(_plasticity_adjustment(fixture, context, base_probability=0.84))
        self.assertLess(heavy_edge, close_edge)


if __name__ == "__main__":
    unittest.main()
