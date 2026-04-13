import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings
from app.main import app
from app.models.schemas import MatchFixture
from app.models.vlr import VLRMatchDetails, VLRMatchRecord
from tests.synthetic_data import generate_synthetic_vct_dataset


def build_training_records(count: int = 24) -> list[VLRMatchRecord]:
    return generate_synthetic_vct_dataset(match_count=count).records


def build_training_details(records: list[VLRMatchRecord]) -> list[VLRMatchDetails]:
    return generate_synthetic_vct_dataset(match_count=len(records)).details


class ApiRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["ARTIFACTS_DIR"] = self.temp_dir.name
        os.environ["MODEL_ARTIFACTS_DIR"] = str(Path(self.temp_dir.name) / "models")
        os.environ["SQLITE_DB_PATH"] = str(Path(self.temp_dir.name) / "vct.sqlite3")
        get_settings.cache_clear()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()
        for key in ("ARTIFACTS_DIR", "MODEL_ARTIFACTS_DIR", "SQLITE_DB_PATH"):
            os.environ.pop(key, None)
        get_settings.cache_clear()

    def test_health_reports_fallback_without_model(self) -> None:
        response = self.client.get("/api/v1/health")
        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["prediction_mode"], "bootstrap")
        self.assertEqual(payload["model_version"], "vlr-core-unavailable")

    def test_predict_returns_prediction_shape(self) -> None:
        response = self.client.post(
            "/api/v1/predict",
            json={"fixtures": [{"match_id": "sample-1", "region": "Pacific", "event_name": "Masters", "team_a": "Team Alpha", "team_b": "Team Beta", "match_date": "2026-04-08", "best_of": 3}]},
        )
        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(payload["predictions"]), 1)
        self.assertEqual(len(payload["predictions"][0]["map_predictions"]), 3)
        self.assertIn("confidence_score", payload["predictions"][0])
        self.assertIn("prediction_generated_at", payload["predictions"][0])
        self.assertIn("sample_size", payload["predictions"][0])

    def test_predict_accepts_international_fixture(self) -> None:
        response = self.client.post(
            "/api/v1/predict",
            json={"fixtures": [{"match_id": "sample-intl", "region": "International", "event_name": "Champions", "team_a": "Paper Rex", "team_b": "Fnatic", "match_date": "2026-04-08", "best_of": 3}]},
        )
        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["predictions"][0]["region"], "International")

    def test_predict_rejects_out_of_scope_fixture(self) -> None:
        response = self.client.post(
            "/api/v1/predict",
            json={"fixtures": [{"match_id": "sample-2", "region": "Game Changers", "event_name": "Masters", "team_a": "Team Alpha", "team_b": "Team Beta", "match_date": "2026-04-08", "best_of": 3}]},
        )
        self.assertEqual(response.status_code, 400)

    @patch("app.services.pipeline.VLRClient.fetch_upcoming_fixtures")
    def test_upcoming_predictions_returns_snapshot(self, mock_fetch_upcoming) -> None:
        mock_fetch_upcoming.return_value = [
            MatchFixture(match_id="fixture-1", region="Pacific", event_name="Masters", event_stage="Group Stage", team_a="Paper Rex", team_b="Gen.G", match_date="2026-04-08", best_of=3)
        ]
        response = self.client.get("/api/v1/predictions/upcoming")
        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["source"], "vlr_schedule")

    @patch("app.services.pipeline.VLRClient.fetch_match_details")
    @patch("app.services.pipeline.VLRClient.fetch_upcoming_fixtures")
    @patch("app.services.pipeline.VLRClient.fetch_matches")
    def test_weekly_pipeline_trains_and_creates_artifact(self, mock_fetch_matches, mock_fetch_upcoming, mock_fetch_details) -> None:
        records = build_training_records()
        mock_fetch_matches.return_value = records
        mock_fetch_upcoming.return_value = []
        mock_fetch_details.return_value = build_training_details(records)
        response = self.client.post("/api/v1/pipeline/weekly")
        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["prediction_mode"], "trained_ml")
        self.assertTrue(os.path.exists(payload["artifact_path"]))

    @patch("app.services.pipeline.VLRClient.fetch_match_details")
    @patch("app.services.pipeline.VLRClient.fetch_upcoming_fixtures")
    @patch("app.services.pipeline.VLRClient.fetch_matches")
    def test_model_and_validation_endpoints_after_training(self, mock_fetch_matches, mock_fetch_upcoming, mock_fetch_details) -> None:
        records = build_training_records()
        mock_fetch_matches.return_value = records
        mock_fetch_upcoming.return_value = []
        mock_fetch_details.return_value = build_training_details(records)
        self.client.post("/api/v1/pipeline/weekly")

        health = self.client.get("/api/v1/health")
        performance = self.client.get("/api/v1/model/performance")
        validation = self.client.get("/api/v1/data/validation")
        self.assertEqual(health.json()["prediction_mode"], "trained_ml")
        self.assertGreater(performance.json()["training_samples"], 0)
        self.assertIn(validation.json()["status"], {"pass", "warn"})


if __name__ == "__main__":
    unittest.main()
