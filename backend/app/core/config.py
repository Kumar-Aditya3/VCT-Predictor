from functools import lru_cache
import os
from typing import Literal
from pathlib import Path


Region = Literal["Americas", "EMEA", "Pacific", "China", "International"]
EventName = Literal["Kickoff", "Split 1", "Split 2", "Masters", "Champions"]


class Settings:
    def __init__(self) -> None:
        backend_dir = Path(__file__).resolve().parents[2]
        project_dir = backend_dir.parent

        self.app_name: str = os.getenv("APP_NAME", "VCT Tier 1 Predictor API")
        self.app_env: str = os.getenv("APP_ENV", "dev")
        self.frontend_origin: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
        self.enable_international_events: bool = os.getenv("ENABLE_INTERNATIONAL_EVENTS", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        if self.enable_international_events:
            self.tier1_regions: tuple[Region, ...] = ("Americas", "EMEA", "Pacific", "China", "International")
        else:
            self.tier1_regions = ("Americas", "EMEA", "Pacific", "China")
        self.tier1_events: tuple[EventName, ...] = (
            "Kickoff",
            "Split 1",
            "Split 2",
            "Masters",
            "Champions",
        )

        self.recency_decay_lambda: float = float(os.getenv("RECENCY_DECAY_LAMBDA", "0.01"))
        self.min_history_maps: int = int(os.getenv("MIN_HISTORY_MAPS", "20"))

        self.project_dir: Path = project_dir
        self.raw_data_dir: Path = Path(os.getenv("RAW_DATA_DIR", str(project_dir / "data" / "raw"))).resolve()
        self.processed_data_dir: Path = Path(
            os.getenv("PROCESSED_DATA_DIR", str(project_dir / "data" / "processed"))
        ).resolve()
        self.sqlite_db_path: Path = Path(
            os.getenv("SQLITE_DB_PATH", str(project_dir / "data" / "processed" / "vct_tier1.sqlite3"))
        ).resolve()
        self.artifacts_dir: Path = Path(os.getenv("ARTIFACTS_DIR", str(project_dir / "artifacts"))).resolve()
        self.model_artifacts_dir: Path = Path(
            os.getenv("MODEL_ARTIFACTS_DIR", str(self.artifacts_dir / "models"))
        ).resolve()
        self.training_history_days: int = int(os.getenv("TRAINING_HISTORY_DAYS", "180"))
        self.training_results_pages: int = int(os.getenv("TRAINING_RESULTS_PAGES", "40"))
        self.upcoming_pages: int = int(os.getenv("UPCOMING_PAGES", "3"))
        self.detail_scrape_limit: int = int(os.getenv("DETAIL_SCRAPE_LIMIT", "600"))
        self.prediction_mode: str = "trained_ml"


@lru_cache
def get_settings() -> Settings:
    return Settings()
