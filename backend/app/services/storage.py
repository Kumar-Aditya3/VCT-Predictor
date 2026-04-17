from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from app.core.config import get_settings
from app.models.vlr import VLRMapRecord, VLRMatchDetails, VLRMatchRecord, VLRPlayerStatLine


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS matches (
        match_id TEXT PRIMARY KEY,
        event_name TEXT NOT NULL,
        region TEXT NOT NULL,
        match_date TEXT NOT NULL,
        team_a TEXT NOT NULL,
        team_b TEXT NOT NULL,
        event_stage TEXT,
        team_a_maps_won INTEGER NOT NULL,
        team_b_maps_won INTEGER NOT NULL,
        best_of INTEGER NOT NULL,
        status TEXT NOT NULL,
        source_url TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS maps (
        map_id TEXT PRIMARY KEY,
        match_id TEXT NOT NULL,
        map_name TEXT NOT NULL,
        team_a TEXT NOT NULL,
        team_b TEXT NOT NULL,
        team_a_rounds INTEGER NOT NULL,
        team_b_rounds INTEGER NOT NULL,
        picked_by TEXT,
        duration_text TEXT,
        winner_team TEXT,
        order_index INTEGER NOT NULL,
        FOREIGN KEY(match_id) REFERENCES matches(match_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS player_map_stats (
        match_id TEXT NOT NULL,
        map_id TEXT NOT NULL,
        map_name TEXT NOT NULL,
        team_name TEXT NOT NULL,
        opponent_team TEXT NOT NULL,
        player_name TEXT NOT NULL,
        agent_name TEXT,
        rating REAL,
        acs REAL,
        kills INTEGER NOT NULL,
        deaths INTEGER NOT NULL,
        assists INTEGER NOT NULL,
        PRIMARY KEY (map_id, player_name),
        FOREIGN KEY(match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
        FOREIGN KEY(map_id) REFERENCES maps(map_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS team_aliases (
        alias_name TEXT PRIMARY KEY,
        canonical_name TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS player_aliases (
        alias_name TEXT PRIMARY KEY,
        canonical_name TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pipeline_runs (
        run_at TEXT PRIMARY KEY,
        artifact_path TEXT NOT NULL,
        model_version TEXT NOT NULL,
        prediction_mode TEXT NOT NULL,
        winner_accuracy REAL NOT NULL,
        map_accuracy REAL NOT NULL,
        player_kd_mae REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS scrape_issues (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_at TEXT NOT NULL,
        scope TEXT NOT NULL,
        reference_id TEXT NOT NULL,
        issue_type TEXT NOT NULL,
        details_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS training_exclusions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_at TEXT NOT NULL,
        scope TEXT NOT NULL,
        reference_id TEXT NOT NULL,
        reason TEXT NOT NULL,
        details_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS feature_runs (
        run_at TEXT PRIMARY KEY,
        summary_json TEXT NOT NULL
    )
    """,
]


class SQLiteStore:
    def __init__(self) -> None:
        settings = get_settings()
        self.db_path = settings.sqlite_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _ensure_schema(self) -> None:
        with self.connect() as connection:
            for statement in SCHEMA_STATEMENTS:
                connection.execute(statement)

    def upsert_match_records(self, records: list[VLRMatchRecord]) -> None:
        if not records:
            return
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO matches (
                    match_id, event_name, region, match_date, team_a, team_b, event_stage,
                    team_a_maps_won, team_b_maps_won, best_of, status, source_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(match_id) DO UPDATE SET
                    event_name=excluded.event_name,
                    region=excluded.region,
                    match_date=excluded.match_date,
                    team_a=excluded.team_a,
                    team_b=excluded.team_b,
                    event_stage=excluded.event_stage,
                    team_a_maps_won=excluded.team_a_maps_won,
                    team_b_maps_won=excluded.team_b_maps_won,
                    best_of=excluded.best_of,
                    status=excluded.status,
                    source_url=excluded.source_url
                """,
                [
                    (
                        record.match_id,
                        record.event_name,
                        record.region,
                        record.match_date.isoformat(),
                        record.team_a,
                        record.team_b,
                        record.event_stage,
                        record.team_a_maps_won,
                        record.team_b_maps_won,
                        record.best_of,
                        record.status,
                        record.source_url,
                    )
                    for record in records
                ],
            )

    def upsert_match_details(self, details_list: list[VLRMatchDetails]) -> None:
        if not details_list:
            return
        with self.connect() as connection:
            for details in details_list:
                connection.execute("DELETE FROM player_map_stats WHERE match_id = ?", (details.match.match_id,))
                connection.execute("DELETE FROM maps WHERE match_id = ?", (details.match.match_id,))
                connection.execute(
                    """
                    INSERT INTO matches (
                        match_id, event_name, region, match_date, team_a, team_b, event_stage,
                        team_a_maps_won, team_b_maps_won, best_of, status, source_url
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(match_id) DO UPDATE SET
                        event_name=excluded.event_name,
                        region=excluded.region,
                        match_date=excluded.match_date,
                        team_a=excluded.team_a,
                        team_b=excluded.team_b,
                        event_stage=excluded.event_stage,
                        team_a_maps_won=excluded.team_a_maps_won,
                        team_b_maps_won=excluded.team_b_maps_won,
                        best_of=excluded.best_of,
                        status=excluded.status,
                        source_url=excluded.source_url
                    """,
                    (
                        details.match.match_id,
                        details.match.event_name,
                        details.match.region,
                        details.match.match_date.isoformat(),
                        details.match.team_a,
                        details.match.team_b,
                        details.match.event_stage,
                        details.match.team_a_maps_won,
                        details.match.team_b_maps_won,
                        details.match.best_of,
                        details.match.status,
                        details.match.source_url,
                    ),
                )
                self._upsert_alias(connection, "team_aliases", details.match.team_a)
                self._upsert_alias(connection, "team_aliases", details.match.team_b)
                for player_stat in details.player_stats:
                    self._upsert_alias(connection, "player_aliases", player_stat.player_name)
                    self._upsert_alias(connection, "team_aliases", player_stat.team_name)
                    self._upsert_alias(connection, "team_aliases", player_stat.opponent_team)
                connection.executemany(
                    """
                    INSERT INTO maps (
                        map_id, match_id, map_name, team_a, team_b, team_a_rounds, team_b_rounds,
                        picked_by, duration_text, winner_team, order_index
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(map_id) DO UPDATE SET
                        match_id=excluded.match_id,
                        map_name=excluded.map_name,
                        team_a=excluded.team_a,
                        team_b=excluded.team_b,
                        team_a_rounds=excluded.team_a_rounds,
                        team_b_rounds=excluded.team_b_rounds,
                        picked_by=excluded.picked_by,
                        duration_text=excluded.duration_text,
                        winner_team=excluded.winner_team,
                        order_index=excluded.order_index
                    """,
                    [
                        (
                            map_record.map_id,
                            map_record.match_id,
                            map_record.map_name,
                            map_record.team_a,
                            map_record.team_b,
                            map_record.team_a_rounds,
                            map_record.team_b_rounds,
                            map_record.picked_by,
                            map_record.duration_text,
                            map_record.winner_team,
                            map_record.order_index,
                        )
                        for map_record in details.maps
                    ],
                )
                connection.executemany(
                    """
                    INSERT INTO player_map_stats (
                        match_id, map_id, map_name, team_name, opponent_team, player_name,
                        agent_name, rating, acs, kills, deaths, assists
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(map_id, player_name) DO UPDATE SET
                        match_id=excluded.match_id,
                        map_name=excluded.map_name,
                        team_name=excluded.team_name,
                        opponent_team=excluded.opponent_team,
                        agent_name=excluded.agent_name,
                        rating=excluded.rating,
                        acs=excluded.acs,
                        kills=excluded.kills,
                        deaths=excluded.deaths,
                        assists=excluded.assists
                    """,
                    [
                        (
                            stat.match_id,
                            stat.map_id,
                            stat.map_name,
                            stat.team_name,
                            stat.opponent_team,
                            stat.player_name,
                            stat.agent_name,
                            stat.rating,
                            stat.acs,
                            stat.kills,
                            stat.deaths,
                            stat.assists,
                        )
                        for stat in details.player_stats
                    ],
                )

    def record_scrape_issues(self, run_at: str, issues: list[dict]) -> None:
        if not issues:
            return
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO scrape_issues (run_at, scope, reference_id, issue_type, details_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_at,
                        issue["scope"],
                        issue["reference_id"],
                        issue["issue_type"],
                        json.dumps(issue.get("details", {}), sort_keys=True),
                    )
                    for issue in issues
                ],
            )

    def record_training_exclusions(self, run_at: str, exclusions: list[dict]) -> None:
        if not exclusions:
            return
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO training_exclusions (run_at, scope, reference_id, reason, details_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_at,
                        exclusion["scope"],
                        exclusion["reference_id"],
                        exclusion["reason"],
                        json.dumps(exclusion.get("details", {}), sort_keys=True),
                    )
                    for exclusion in exclusions
                ],
            )

    def record_feature_run(self, run_at: str, summary: dict) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO feature_runs (run_at, summary_json)
                VALUES (?, ?)
                ON CONFLICT(run_at) DO UPDATE SET summary_json=excluded.summary_json
                """,
                (run_at, json.dumps(summary, sort_keys=True)),
            )

    def record_pipeline_run(
        self,
        run_at: str,
        artifact_path: str,
        model_version: str,
        prediction_mode: str,
        winner_accuracy: float,
        map_accuracy: float,
        player_kd_mae: float,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO pipeline_runs (
                    run_at, artifact_path, model_version, prediction_mode, winner_accuracy, map_accuracy, player_kd_mae
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_at) DO UPDATE SET
                    artifact_path=excluded.artifact_path,
                    model_version=excluded.model_version,
                    prediction_mode=excluded.prediction_mode,
                    winner_accuracy=excluded.winner_accuracy,
                    map_accuracy=excluded.map_accuracy,
                    player_kd_mae=excluded.player_kd_mae
                """,
                (run_at, artifact_path, model_version, prediction_mode, winner_accuracy, map_accuracy, player_kd_mae),
            )

    def load_matches(self) -> list[VLRMatchRecord]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM matches ORDER BY match_date, match_id").fetchall()
        return [self._row_to_match(row) for row in rows]

    def load_maps(self) -> list[VLRMapRecord]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM maps ORDER BY match_id, order_index").fetchall()
        return [self._row_to_map(row) for row in rows]

    def load_player_stats(self) -> list[VLRPlayerStatLine]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM player_map_stats ORDER BY match_id, map_id, team_name, player_name"
            ).fetchall()
        return [self._row_to_player_stat(row) for row in rows]

    def counts(self) -> dict[str, int]:
        with self.connect() as connection:
            match_count = connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
            map_count = connection.execute("SELECT COUNT(*) FROM maps").fetchone()[0]
            player_stat_count = connection.execute("SELECT COUNT(*) FROM player_map_stats").fetchone()[0]
            scrape_issue_count = connection.execute("SELECT COUNT(*) FROM scrape_issues").fetchone()[0]
            exclusion_count = connection.execute("SELECT COUNT(*) FROM training_exclusions").fetchone()[0]
        return {
            "matches": int(match_count),
            "maps": int(map_count),
            "player_stats": int(player_stat_count),
            "scrape_issues": int(scrape_issue_count),
            "training_exclusions": int(exclusion_count),
        }

    def _upsert_alias(self, connection: sqlite3.Connection, table_name: str, canonical_name: str) -> None:
        connection.execute(
            f"INSERT INTO {table_name} (alias_name, canonical_name) VALUES (?, ?) ON CONFLICT(alias_name) DO NOTHING",
            (canonical_name.lower(), canonical_name),
        )

    @staticmethod
    def _row_to_match(row: sqlite3.Row) -> VLRMatchRecord:
        return VLRMatchRecord(
            match_id=row["match_id"],
            event_name=row["event_name"],
            region=row["region"],
            match_date=row["match_date"],
            team_a=row["team_a"],
            team_b=row["team_b"],
            event_stage=row["event_stage"],
            team_a_maps_won=row["team_a_maps_won"],
            team_b_maps_won=row["team_b_maps_won"],
            best_of=row["best_of"],
            status=row["status"],
            source_url=row["source_url"],
        )

    @staticmethod
    def _row_to_map(row: sqlite3.Row) -> VLRMapRecord:
        return VLRMapRecord(
            map_id=row["map_id"],
            match_id=row["match_id"],
            map_name=row["map_name"],
            team_a=row["team_a"],
            team_b=row["team_b"],
            team_a_rounds=row["team_a_rounds"],
            team_b_rounds=row["team_b_rounds"],
            picked_by=row["picked_by"],
            duration_text=row["duration_text"],
            winner_team=row["winner_team"],
            order_index=row["order_index"],
        )

    @staticmethod
    def _row_to_player_stat(row: sqlite3.Row) -> VLRPlayerStatLine:
        return VLRPlayerStatLine(
            match_id=row["match_id"],
            map_id=row["map_id"],
            map_name=row["map_name"],
            team_name=row["team_name"],
            opponent_team=row["opponent_team"],
            player_name=row["player_name"],
            agent_name=row["agent_name"],
            rating=row["rating"],
            acs=row["acs"],
            kills=row["kills"],
            deaths=row["deaths"],
            assists=row["assists"],
        )
