from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.core.config import get_settings
from app.models.schemas import MatchFixture
from app.models.vlr import VLRMapRecord, VLRMatchRecord, VLRPlayerStatLine

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # pragma: no cover - optional dependency
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None
    XGBRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:  # pragma: no cover - optional dependency
    CatBoostClassifier = None
    CatBoostRegressor = None


MODEL_FILE_NAME = "prediction_bundle.pkl"
RECENT_WINDOW = 8
INITIAL_ELO = 1500.0
ELO_K = 24.0
COMMON_MAPS = ("Abyss", "Ascent", "Bind", "Haven", "Icebox", "Lotus", "Pearl", "Split", "Sunset")
SHORT_WINDOW = 3
MEDIUM_WINDOW = 5


@dataclass
class TeamState:
    matches: int = 0
    wins: int = 0
    maps_won: int = 0
    maps_lost: int = 0
    recent_results: deque[int] = field(default_factory=lambda: deque(maxlen=RECENT_WINDOW))
    recent_map_margin: deque[int] = field(default_factory=lambda: deque(maxlen=RECENT_WINDOW))
    elo: float = INITIAL_ELO
    last_played: date | None = None
    map_matches: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    map_wins: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    map_round_diff: dict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass
class PlayerState:
    team_name: str
    maps_played: int = 0
    recent_kills: deque[float] = field(default_factory=lambda: deque(maxlen=RECENT_WINDOW))
    recent_deaths: deque[float] = field(default_factory=lambda: deque(maxlen=RECENT_WINDOW))
    recent_acs: deque[float] = field(default_factory=lambda: deque(maxlen=RECENT_WINDOW))
    map_kills: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=RECENT_WINDOW)))
    map_deaths: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=RECENT_WINDOW)))
    agent_kills: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=RECENT_WINDOW)))
    agent_deaths: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=RECENT_WINDOW)))
    last_played: date | None = None
    last_agent: str | None = None


def train_prediction_bundle(
    matches: list[VLRMatchRecord],
    maps: list[VLRMapRecord],
    player_stats: list[VLRPlayerStatLine],
) -> dict | None:
    if len(matches) < 20 or len(maps) < 40:
        return None

    ordered_matches = sorted(matches, key=lambda item: (item.match_date, item.match_id))
    maps_by_match: dict[str, list[VLRMapRecord]] = defaultdict(list)
    stats_by_map: dict[str, list[VLRPlayerStatLine]] = defaultdict(list)
    for map_record in maps:
        maps_by_match[map_record.match_id].append(map_record)
    for map_list in maps_by_match.values():
        map_list.sort(key=lambda item: item.order_index)
    for stat in player_stats:
        stats_by_map[stat.map_id].append(stat)

    feature_store = _build_feature_store(ordered_matches, maps_by_match, stats_by_map)
    if len(feature_store["match_rows"]) < 20 or len(feature_store["map_rows"]) < 40:
        return None

    match_bundle = _train_classifier_bundle(
        feature_store["match_rows"],
        feature_store["match_labels"],
        feature_store["match_dates"],
        f"vlr-match-{ordered_matches[-1].match_date.isoformat()}",
        objective="holdout",
    )
    map_bundle = _train_classifier_bundle(
        feature_store["map_rows"],
        feature_store["map_labels"],
        feature_store["map_dates"],
        f"vlr-map-{ordered_matches[-1].match_date.isoformat()}",
        objective="rolling",
    )
    player_kills_bundle = None
    player_deaths_bundle = None
    if len(feature_store["player_rows"]) >= 100:
        player_kills_bundle = _train_regressor_bundle(
            feature_store["player_rows"],
            feature_store["player_kills"],
            feature_store["player_dates"],
            f"vlr-player-kills-{ordered_matches[-1].match_date.isoformat()}",
        )
        player_deaths_bundle = _train_regressor_bundle(
            feature_store["player_rows"],
            feature_store["player_deaths"],
            feature_store["player_dates"],
            f"vlr-player-deaths-{ordered_matches[-1].match_date.isoformat()}",
        )
    if match_bundle is None or map_bundle is None:
        return None

    player_mae = ((player_kills_bundle["mae"] + player_deaths_bundle["mae"]) / 2) if (player_kills_bundle and player_deaths_bundle) else 0.0
    player_cv_score = max(0.0, 1.0 - player_mae / 25.0) if (player_kills_bundle and player_deaths_bundle) else 0.0
    model_version = f"vlr-core-{ordered_matches[-1].match_date.isoformat()}"
    return {
        "model_version": model_version,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "history_start": ordered_matches[0].match_date.isoformat(),
        "history_end": ordered_matches[-1].match_date.isoformat(),
        "match_model": match_bundle,
        "map_model": map_bundle,
        "player_kills_model": player_kills_bundle,
        "player_deaths_model": player_deaths_bundle,
        "metrics": {
            "winner_accuracy": match_bundle["accuracy"],
            "map_accuracy": map_bundle["accuracy"],
            "player_kd_mae": player_mae,
            "training_samples": len(feature_store["match_rows"]),
            "map_training_samples": len(feature_store["map_rows"]),
            "player_training_samples": len(feature_store["player_rows"]),
            "compared_matches": match_bundle["validation_samples"],
            "compared_maps": map_bundle["validation_samples"],
            "player_rows": player_kills_bundle["validation_samples"] if player_kills_bundle is not None else 0,
            "cv_accuracy": match_bundle["accuracy"],
            "map_cv_accuracy": map_bundle["accuracy"],
            "player_cv_score": player_cv_score,
            "rolling_winner_accuracy": match_bundle["rolling_accuracy"],
            "rolling_map_accuracy": map_bundle["rolling_accuracy"],
            "rolling_player_kd_mae": ((player_kills_bundle["rolling_mae"] + player_deaths_bundle["rolling_mae"]) / 2) if (player_kills_bundle and player_deaths_bundle) else 0.0,
            "match_estimator": match_bundle["estimator_name"],
            "map_estimator": map_bundle["estimator_name"],
            "player_kills_estimator": player_kills_bundle["estimator_name"] if player_kills_bundle is not None else "unavailable",
            "player_deaths_estimator": player_deaths_bundle["estimator_name"] if player_deaths_bundle is not None else "unavailable",
            "match_calibration": match_bundle["calibration_method"],
            "map_calibration": map_bundle["calibration_method"],
        },
        "search": {
            "match_candidates": match_bundle["candidate_results"],
            "map_candidates": map_bundle["candidate_results"],
            "player_kills_candidates": player_kills_bundle["candidate_results"] if player_kills_bundle is not None else [],
            "player_deaths_candidates": player_deaths_bundle["candidate_results"] if player_deaths_bundle is not None else [],
        },
        "backtests": {
            "rolling_winner_accuracy": match_bundle["rolling_accuracy"],
            "rolling_map_accuracy": map_bundle["rolling_accuracy"],
            "rolling_player_kills_mae": player_kills_bundle["rolling_mae"] if player_kills_bundle is not None else 0.0,
            "rolling_player_deaths_mae": player_deaths_bundle["rolling_mae"] if player_deaths_bundle is not None else 0.0,
            "rolling_windows_evaluated": 3,
        },
        "calibration": {
            "match": match_bundle["calibration_method"],
            "map": map_bundle["calibration_method"],
        },
        "residuals": {
            "player_kills": player_kills_bundle["residual_quantiles"] if player_kills_bundle is not None else {},
            "player_deaths": player_deaths_bundle["residual_quantiles"] if player_deaths_bundle is not None else {},
        },
        "context": feature_store["context"],
    }


def save_model_bundle(bundle: dict) -> Path:
    settings = get_settings()
    settings.model_artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = settings.model_artifacts_dir / MODEL_FILE_NAME
    with model_path.open("wb") as file:
        pickle.dump(_serialize_bundle(bundle), file)
    return model_path


def load_model_bundle() -> dict | None:
    settings = get_settings()
    model_path = settings.model_artifacts_dir / MODEL_FILE_NAME
    if not model_path.exists():
        return None
    try:
        with model_path.open("rb") as file:
            return _deserialize_bundle(pickle.load(file))
    except (EOFError, pickle.PickleError, AttributeError, ValueError):
        return None


def train_match_winner_model(matches: list[VLRMatchRecord]) -> dict | None:
    return None if len(matches) < 20 else {"model_version": "compat-disabled"}


def predict_match_probability(fixture: MatchFixture, model_bundle: dict) -> float:
    forward_row = build_match_feature_row(fixture, model_bundle["context"])
    forward_frame = pd.DataFrame([forward_row], columns=model_bundle["match_model"]["feature_columns"])
    forward_probability = _predict_classifier_probability(model_bundle["match_model"], forward_frame)

    reverse_fixture = _swap_fixture_teams(fixture)
    reverse_row = build_match_feature_row(reverse_fixture, model_bundle["context"])
    reverse_frame = pd.DataFrame([reverse_row], columns=model_bundle["match_model"]["feature_columns"])
    reverse_probability = _predict_classifier_probability(model_bundle["match_model"], reverse_frame)

    # Make inference invariant to Team A / Team B ordering.
    return float(np.clip((forward_probability + (1.0 - reverse_probability)) / 2.0, 1e-6, 1 - 1e-6))


def predict_map_probability(fixture: MatchFixture, map_name: str, picked_by: str | None, model_bundle: dict) -> float:
    forward_row = build_map_feature_row(fixture, map_name, picked_by, model_bundle["context"])
    forward_frame = pd.DataFrame([forward_row], columns=model_bundle["map_model"]["feature_columns"])
    forward_probability = _predict_classifier_probability(model_bundle["map_model"], forward_frame)

    reverse_fixture = _swap_fixture_teams(fixture)
    reverse_picked_by = _swap_picked_by_team(picked_by, fixture)
    reverse_row = build_map_feature_row(reverse_fixture, map_name, reverse_picked_by, model_bundle["context"])
    reverse_frame = pd.DataFrame([reverse_row], columns=model_bundle["map_model"]["feature_columns"])
    reverse_probability = _predict_classifier_probability(model_bundle["map_model"], reverse_frame)

    return float(np.clip((forward_probability + (1.0 - reverse_probability)) / 2.0, 1e-6, 1 - 1e-6))


def predict_player_stat_lines(
    fixture: MatchFixture,
    selected_maps: list[tuple[str, str | None]],
    model_bundle: dict,
) -> list[dict]:
    if model_bundle.get("player_kills_model") is None or model_bundle.get("player_deaths_model") is None:
        return []
    context = model_bundle["context"]
    match_probability = predict_match_probability(fixture, model_bundle)
    decider_weight = max(0.15, 1.0 - abs(match_probability - 0.5) * 2)
    map_weights = [1.0, 1.0, decider_weight][: len(selected_maps)]
    team_a_players = _likely_lineup(fixture.team_a, context)
    team_b_players = _likely_lineup(fixture.team_b, context)
    projections: list[dict] = []

    for team_name, opponent_team, players in (
        (fixture.team_a, fixture.team_b, team_a_players),
        (fixture.team_b, fixture.team_a, team_b_players),
    ):
        for player_name in players:
            total_kills = 0.0
            total_deaths = 0.0
            agent_name = _preferred_agent(player_name, context)
            for (map_name, _picked_by), weight in zip(selected_maps, map_weights):
                row = build_player_feature_row(fixture, team_name, opponent_team, player_name, map_name, agent_name, context)
                frame = pd.DataFrame([row], columns=model_bundle["player_kills_model"]["feature_columns"])
                kills = float(model_bundle["player_kills_model"]["pipeline"].predict(frame)[0])
                deaths = float(model_bundle["player_deaths_model"]["pipeline"].predict(frame)[0])
                total_kills += max(0.0, kills) * weight
                total_deaths += max(0.0, deaths) * weight
            projections.append(
                {
                    "player_name": player_name,
                    "team_name": team_name,
                    "agent_name": agent_name,
                    "projected_kills": round(total_kills, 1),
                    "projected_deaths": round(total_deaths, 1),
                }
            )
    return projections


def select_maps_for_fixture(fixture: MatchFixture, model_bundle: dict) -> list[tuple[str, str | None]]:
    context = model_bundle["context"]
    team_a_map_rank = _rank_maps_for_team(fixture.team_a, context)
    team_b_map_rank = _rank_maps_for_team(fixture.team_b, context)
    chosen: list[tuple[str, str | None]] = []

    if team_a_map_rank:
        chosen.append((team_a_map_rank[0], fixture.team_a))
    if team_b_map_rank:
        second = next((map_name for map_name in team_b_map_rank if map_name not in {name for name, _ in chosen}), None)
        if second is not None:
            chosen.append((second, fixture.team_b))

    combined = _combined_map_rank(fixture.team_a, fixture.team_b, context)
    decider = next((map_name for map_name in combined if map_name not in {name for name, _ in chosen}), None)
    if decider is not None:
        chosen.append((decider, None))
    return chosen[: max(3, fixture.best_of)]


def build_match_feature_row(fixture: MatchFixture, context: dict) -> dict:
    team_states = context["team_states"]
    team_a_state = team_states.get(fixture.team_a, TeamState())
    team_b_state = team_states.get(fixture.team_b, TeamState())
    player_states = context["player_states"]
    head_to_head = context["head_to_head"]

    return {
        "region": fixture.region,
        "event_name": fixture.event_name,
        "event_stage": fixture.event_stage or "unknown",
        "best_of": fixture.best_of,
        "elo_diff": team_a_state.elo - team_b_state.elo,
        "matches_played_diff": team_a_state.matches - team_b_state.matches,
        "win_rate_diff": _win_rate(team_a_state) - _win_rate(team_b_state),
        "map_win_rate_diff": _map_win_rate(team_a_state) - _map_win_rate(team_b_state),
        "recent_win_rate_diff": _recent_win_rate(team_a_state) - _recent_win_rate(team_b_state),
        "recent_short_win_rate_diff": _recent_win_rate_window(team_a_state, SHORT_WINDOW) - _recent_win_rate_window(team_b_state, SHORT_WINDOW),
        "recent_medium_win_rate_diff": _recent_win_rate_window(team_a_state, MEDIUM_WINDOW) - _recent_win_rate_window(team_b_state, MEDIUM_WINDOW),
        "recent_map_margin_diff": _recent_map_margin(team_a_state) - _recent_map_margin(team_b_state),
        "recent_short_map_margin_diff": _recent_map_margin_window(team_a_state, SHORT_WINDOW) - _recent_map_margin_window(team_b_state, SHORT_WINDOW),
        "recent_medium_map_margin_diff": _recent_map_margin_window(team_a_state, MEDIUM_WINDOW) - _recent_map_margin_window(team_b_state, MEDIUM_WINDOW),
        "rest_days_diff": _rest_days(team_a_state, fixture.match_date) - _rest_days(team_b_state, fixture.match_date),
        "head_to_head_delta": _matchup_delta(fixture.team_a, fixture.team_b, head_to_head),
        "map_pool_overlap": _map_pool_overlap(fixture.team_a, fixture.team_b, context),
        "map_pool_depth_diff": _map_pool_depth(team_a_state) - _map_pool_depth(team_b_state),
        "team_a_best_map_win_rate": _best_map_win_rate(team_a_state),
        "team_b_best_map_win_rate": _best_map_win_rate(team_b_state),
        "player_kills_diff": _team_player_metric(fixture.team_a, player_states, "kills") - _team_player_metric(fixture.team_b, player_states, "kills"),
        "player_deaths_diff": _team_player_metric(fixture.team_a, player_states, "deaths") - _team_player_metric(fixture.team_b, player_states, "deaths"),
        "player_acs_diff": _team_player_metric(fixture.team_a, player_states, "acs") - _team_player_metric(fixture.team_b, player_states, "acs"),
        "lineup_stability_diff": _lineup_stability(fixture.team_a, context) - _lineup_stability(fixture.team_b, context),
    }


def build_map_feature_row(fixture: MatchFixture, map_name: str, picked_by: str | None, context: dict) -> dict:
    team_states = context["team_states"]
    head_to_head_by_map = context["head_to_head_by_map"]
    return {
        "region": fixture.region,
        "event_name": fixture.event_name,
        "map_name": map_name,
        "picked_by": picked_by or "decider",
        "elo_diff": team_states.get(fixture.team_a, TeamState()).elo - team_states.get(fixture.team_b, TeamState()).elo,
        "team_map_win_rate_diff": _team_map_win_rate(fixture.team_a, map_name, context) - _team_map_win_rate(fixture.team_b, map_name, context),
        "team_map_round_diff": _team_map_round_diff(fixture.team_a, map_name, context) - _team_map_round_diff(fixture.team_b, map_name, context),
        "recent_team_form_diff": _recent_win_rate(team_states.get(fixture.team_a, TeamState())) - _recent_win_rate(team_states.get(fixture.team_b, TeamState())),
        "map_head_to_head_delta": _matchup_delta(fixture.team_a, fixture.team_b, head_to_head_by_map.get(map_name, {})),
        "player_map_acs_diff": _team_player_map_metric(fixture.team_a, map_name, context, "acs") - _team_player_map_metric(fixture.team_b, map_name, context, "acs"),
    }


def build_player_feature_row(
    fixture: MatchFixture,
    team_name: str,
    opponent_team: str,
    player_name: str,
    map_name: str,
    agent_name: str | None,
    context: dict,
) -> dict:
    player_state = context["player_states"].get(player_name)
    return {
        "region": fixture.region,
        "event_name": fixture.event_name,
        "map_name": map_name,
        "agent_name": agent_name or "unknown",
        "team_name": team_name,
        "opponent_team": opponent_team,
        "team_elo": context["team_states"].get(team_name, TeamState()).elo,
        "opponent_elo": context["team_states"].get(opponent_team, TeamState()).elo,
        "player_recent_kills": _player_metric(player_state, "kills"),
        "player_recent_deaths": _player_metric(player_state, "deaths"),
        "player_recent_acs": _player_metric(player_state, "acs"),
        "player_map_kills": _player_metric(player_state, "kills", map_name=map_name),
        "player_map_deaths": _player_metric(player_state, "deaths", map_name=map_name),
        "player_agent_kills": _player_metric(player_state, "kills", agent_name=agent_name),
        "player_agent_deaths": _player_metric(player_state, "deaths", agent_name=agent_name),
        "team_map_win_rate": _team_map_win_rate(team_name, map_name, context),
        "opponent_map_win_rate": _team_map_win_rate(opponent_team, map_name, context),
        "lineup_stability": _lineup_stability(team_name, context),
    }


def _build_feature_store(
    ordered_matches: list[VLRMatchRecord],
    maps_by_match: dict[str, list[VLRMapRecord]],
    stats_by_map: dict[str, list[VLRPlayerStatLine]],
) -> dict:
    team_states: dict[str, TeamState] = {}
    player_states: dict[str, PlayerState] = {}
    head_to_head: dict[tuple[str, str], int] = defaultdict(int)
    head_to_head_by_map: dict[str, dict[tuple[str, str], int]] = defaultdict(lambda: defaultdict(int))
    team_recent_players: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    store = {
        "match_rows": [],
        "match_labels": [],
        "match_dates": [],
        "map_rows": [],
        "map_labels": [],
        "map_dates": [],
        "player_rows": [],
        "player_kills": [],
        "player_deaths": [],
        "player_dates": [],
    }

    for match in ordered_matches:
        fixture = MatchFixture(
            match_id=match.match_id,
            region=match.region,
            event_name=match.event_name,
            event_stage=match.event_stage,
            team_a=match.team_a,
            team_b=match.team_b,
            match_date=match.match_date,
            best_of=match.best_of,
        )
        context = {
            "team_states": team_states,
            "player_states": player_states,
            "head_to_head": head_to_head,
            "head_to_head_by_map": head_to_head_by_map,
            "team_recent_players": team_recent_players,
        }
        store["match_rows"].append(build_match_feature_row(fixture, context))
        store["match_labels"].append(1 if match.team_a_maps_won > match.team_b_maps_won else 0)
        store["match_dates"].append(match.match_date)

        for map_record in maps_by_match.get(match.match_id, []):
            store["map_rows"].append(build_map_feature_row(fixture, map_record.map_name, map_record.picked_by, context))
            store["map_labels"].append(1 if map_record.winner_team == map_record.team_a else 0)
            store["map_dates"].append(match.match_date)
            for stat in stats_by_map.get(map_record.map_id, []):
                store["player_rows"].append(
                    build_player_feature_row(fixture, stat.team_name, stat.opponent_team, stat.player_name, map_record.map_name, stat.agent_name, context)
                )
                store["player_kills"].append(stat.kills)
                store["player_deaths"].append(stat.deaths)
                store["player_dates"].append(match.match_date)
            _update_map_and_player_states(map_record, stats_by_map.get(map_record.map_id, []), team_states, player_states, head_to_head_by_map, team_recent_players, match.match_date)

        _update_match_states(match, team_states, head_to_head)

    store["context"] = {
        "team_states": {name: _clone_team_state(state) for name, state in team_states.items()},
        "player_states": {name: _clone_player_state(state) for name, state in player_states.items()},
        "head_to_head": dict(head_to_head),
        "head_to_head_by_map": {map_name: dict(values) for map_name, values in head_to_head_by_map.items()},
        "team_recent_players": {team: dict(players) for team, players in team_recent_players.items()},
    }
    return store


def _update_match_states(match: VLRMatchRecord, team_states: dict[str, TeamState], head_to_head: dict[tuple[str, str], int]) -> None:
    team_a_state = team_states.setdefault(match.team_a, TeamState())
    team_b_state = team_states.setdefault(match.team_b, TeamState())
    team_a_won = match.team_a_maps_won > match.team_b_maps_won
    expected_a = 1 / (1 + 10 ** ((team_b_state.elo - team_a_state.elo) / 400))
    elo_change = ELO_K * ((1.0 if team_a_won else 0.0) - expected_a)
    team_a_state.elo += elo_change
    team_b_state.elo -= elo_change
    margin = match.team_a_maps_won - match.team_b_maps_won
    _apply_match_result(team_a_state, team_a_won, match.team_a_maps_won, match.team_b_maps_won, match.match_date, margin)
    _apply_match_result(team_b_state, not team_a_won, match.team_b_maps_won, match.team_a_maps_won, match.match_date, -margin)
    pair_key = tuple(sorted((match.team_a, match.team_b)))
    if team_a_won:
        head_to_head[pair_key] += 1 if match.team_a <= match.team_b else -1
    else:
        head_to_head[pair_key] -= 1 if match.team_a <= match.team_b else -1


def _update_map_and_player_states(
    map_record: VLRMapRecord,
    stats: list[VLRPlayerStatLine],
    team_states: dict[str, TeamState],
    player_states: dict[str, PlayerState],
    head_to_head_by_map: dict[str, dict[tuple[str, str], int]],
    team_recent_players: dict[str, dict[str, float]],
    match_date: date,
) -> None:
    team_a_state = team_states.setdefault(map_record.team_a, TeamState())
    team_b_state = team_states.setdefault(map_record.team_b, TeamState())
    team_a_state.map_matches[map_record.map_name] += 1
    team_b_state.map_matches[map_record.map_name] += 1
    team_a_state.map_round_diff[map_record.map_name] += map_record.team_a_rounds - map_record.team_b_rounds
    team_b_state.map_round_diff[map_record.map_name] += map_record.team_b_rounds - map_record.team_a_rounds
    if map_record.winner_team == map_record.team_a:
        team_a_state.map_wins[map_record.map_name] += 1
    elif map_record.winner_team == map_record.team_b:
        team_b_state.map_wins[map_record.map_name] += 1

    pair_key = tuple(sorted((map_record.team_a, map_record.team_b)))
    if map_record.winner_team == map_record.team_a:
        head_to_head_by_map[map_record.map_name][pair_key] += 1 if map_record.team_a <= map_record.team_b else -1
    elif map_record.winner_team == map_record.team_b:
        head_to_head_by_map[map_record.map_name][pair_key] -= 1 if map_record.team_a <= map_record.team_b else -1

    for stat in stats:
        player_state = player_states.setdefault(stat.player_name, PlayerState(team_name=stat.team_name))
        player_state.team_name = stat.team_name
        player_state.maps_played += 1
        player_state.recent_kills.append(float(stat.kills))
        player_state.recent_deaths.append(float(stat.deaths))
        player_state.recent_acs.append(float(stat.acs or 0.0))
        player_state.map_kills[stat.map_name].append(float(stat.kills))
        player_state.map_deaths[stat.map_name].append(float(stat.deaths))
        if stat.agent_name:
            player_state.agent_kills[stat.agent_name].append(float(stat.kills))
            player_state.agent_deaths[stat.agent_name].append(float(stat.deaths))
            player_state.last_agent = stat.agent_name
        player_state.last_played = match_date
        team_recent_players[stat.team_name][stat.player_name] += 1.0


def _train_classifier_bundle(
    rows: list[dict],
    labels: list[int],
    dates: list[date],
    version: str,
    *,
    objective: str = "rolling",
) -> dict | None:
    if len(rows) < 20:
        return None
    frame = pd.DataFrame(rows)
    split_index = max(10, int(len(frame) * 0.8))
    if split_index >= len(frame):
        split_index = len(frame) - 1
    train_X = frame.iloc[:split_index]
    val_X = frame.iloc[split_index:]
    train_y = labels[:split_index]
    val_y = labels[split_index:]
    if not val_y:
        return None
    categorical_features = [col for col in frame.columns if frame[col].dtype == object]
    numeric_features = [col for col in frame.columns if col not in categorical_features]
    candidates = _classifier_candidates()
    candidate_results: list[dict] = []
    for candidate in candidates:
        for decay_lambda in (0.005, 0.01, 0.02):
            rolling_accuracy = _rolling_classifier_score(
                frame,
                labels,
                dates,
                candidate,
                categorical_features,
                numeric_features,
                decay_lambda,
            )
            pipeline = _classifier_pipeline(categorical_features, numeric_features, candidate["estimator"])
            _fit_pipeline(
                pipeline,
                train_X,
                train_y,
                _sample_weights(dates[:split_index], dates[-1], decay_lambda),
                step_name="classifier",
                supports_sample_weight=candidate["supports_sample_weight"],
            )
            holdout_accuracy = accuracy_score(val_y, pipeline.predict(val_X))
            probabilities = pipeline.predict_proba(val_X)[:, 1]
            candidate_results.append(
                {
                    "name": candidate["name"],
                    "decay_lambda": decay_lambda,
                    "rolling_accuracy": float(rolling_accuracy),
                    "holdout_accuracy": float(holdout_accuracy),
                    "brier": float(brier_score_loss(val_y, probabilities)),
                }
            )

    if not candidate_results:
        return None
    if objective == "holdout":
        candidate_results.sort(key=lambda item: (item["holdout_accuracy"], item["rolling_accuracy"], -item["brier"]), reverse=True)
    else:
        candidate_results.sort(key=lambda item: (item["rolling_accuracy"], item["holdout_accuracy"], -item["brier"]), reverse=True)
    best_result = candidate_results[0]
    best_candidate = next(candidate for candidate in candidates if candidate["name"] == best_result["name"])
    ensemble_result = _build_classifier_ensemble_candidate(
        candidate_results,
        candidates,
        frame,
        labels,
        dates,
        train_X,
        train_y,
        val_X,
        val_y,
        categorical_features,
        numeric_features,
    )
    if objective == "holdout":
        selected_result = ensemble_result if ensemble_result is not None and (
            ensemble_result["holdout_accuracy"] > best_result["holdout_accuracy"]
            or (
                ensemble_result["holdout_accuracy"] == best_result["holdout_accuracy"]
                and ensemble_result["rolling_accuracy"] > best_result["rolling_accuracy"]
            )
        ) else best_result
    else:
        selected_result = ensemble_result if ensemble_result is not None and (
            ensemble_result["rolling_accuracy"] > best_result["rolling_accuracy"]
            or (
                ensemble_result["rolling_accuracy"] == best_result["rolling_accuracy"]
                and ensemble_result["holdout_accuracy"] > best_result["holdout_accuracy"]
            )
        ) else best_result

    if selected_result.get("kind") == "ensemble":
        final_model = _train_classifier_ensemble(
            selected_result,
            candidates,
            frame,
            labels,
            dates,
            categorical_features,
            numeric_features,
        )
    else:
        final_pipeline = _classifier_pipeline(categorical_features, numeric_features, best_candidate["estimator"])
        _fit_pipeline(
            final_pipeline,
            frame,
            labels,
            _sample_weights(dates, dates[-1], selected_result["decay_lambda"]),
            step_name="classifier",
            supports_sample_weight=best_candidate["supports_sample_weight"],
        )
        calibration = _fit_classifier_calibration(
            train_X,
            train_y,
            val_X,
            val_y,
            dates[:split_index],
            dates[-1],
            selected_result["decay_lambda"],
            best_candidate,
            categorical_features,
            numeric_features,
        )
        final_model = {
            "kind": "single",
            "pipeline": final_pipeline,
            "supports_sample_weight": best_candidate["supports_sample_weight"],
            "decay_lambda": selected_result["decay_lambda"],
            "calibration": calibration,
        }

    return {
        "model": final_model,
        "feature_columns": list(frame.columns),
        "accuracy": float(selected_result["holdout_accuracy"]),
        "version": version,
        "training_samples": len(frame),
        "validation_samples": len(val_y),
        "rolling_accuracy": float(selected_result["rolling_accuracy"]),
        "estimator_name": selected_result["name"],
        "candidate_scores": {f"{item['name']}@{item['decay_lambda']:.3f}": item["holdout_accuracy"] for item in candidate_results},
        "candidate_results": candidate_results[:12],
        "calibration_method": final_model.get("calibration", {}).get("method", "none"),
        "selected_decay_lambda": selected_result["decay_lambda"],
        "selection_objective": objective,
    }


def _train_regressor_bundle(rows: list[dict], targets: list[float], dates: list[date], version: str) -> dict | None:
    if len(rows) < 40:
        return None
    frame = pd.DataFrame(rows)
    split_index = max(20, int(len(frame) * 0.8))
    if split_index >= len(frame):
        split_index = len(frame) - 1
    train_X = frame.iloc[:split_index]
    val_X = frame.iloc[split_index:]
    train_y = targets[:split_index]
    val_y = targets[split_index:]
    if not val_y:
        return None
    categorical_features = [col for col in frame.columns if frame[col].dtype == object]
    numeric_features = [col for col in frame.columns if col not in categorical_features]
    candidates = _regressor_candidates()
    candidate_results: list[dict] = []
    for candidate in candidates:
        for decay_lambda in (0.005, 0.01, 0.02):
            rolling_mae = _rolling_regressor_score(
                frame,
                targets,
                dates,
                candidate,
                categorical_features,
                numeric_features,
                decay_lambda,
            )
            pipeline = _regressor_pipeline(categorical_features, numeric_features, candidate["estimator"])
            _fit_pipeline(
                pipeline,
                train_X,
                train_y,
                _sample_weights(dates[:split_index], dates[-1], decay_lambda),
                step_name="regressor",
                supports_sample_weight=candidate["supports_sample_weight"],
            )
            holdout_mae = mean_absolute_error(val_y, pipeline.predict(val_X))
            candidate_results.append(
                {
                    "name": candidate["name"],
                    "decay_lambda": decay_lambda,
                    "rolling_mae": float(rolling_mae),
                    "holdout_mae": float(holdout_mae),
                }
            )

    if not candidate_results:
        return None
    candidate_results.sort(key=lambda item: (item["rolling_mae"], item["holdout_mae"]))
    best_result = candidate_results[0]
    best_candidate = next(candidate for candidate in candidates if candidate["name"] == best_result["name"])
    final_pipeline = _regressor_pipeline(categorical_features, numeric_features, best_candidate["estimator"])
    _fit_pipeline(
        final_pipeline,
        frame,
        targets,
        _sample_weights(dates, dates[-1], best_result["decay_lambda"]),
        step_name="regressor",
        supports_sample_weight=best_candidate["supports_sample_weight"],
    )
    residual_quantiles = _regressor_residual_quantiles(
        train_X,
        train_y,
        val_X,
        val_y,
        dates[:split_index],
        dates[-1],
        best_result["decay_lambda"],
        best_candidate,
        categorical_features,
        numeric_features,
    )
    return {
        "pipeline": final_pipeline,
        "feature_columns": list(frame.columns),
        "mae": float(best_result["holdout_mae"]),
        "version": version,
        "training_samples": len(frame),
        "validation_samples": len(val_y),
        "rolling_mae": float(best_result["rolling_mae"]),
        "estimator_name": best_result["name"],
        "candidate_scores": {f"{item['name']}@{item['decay_lambda']:.3f}": item["holdout_mae"] for item in candidate_results},
        "candidate_results": candidate_results[:12],
        "selected_decay_lambda": best_result["decay_lambda"],
        "residual_quantiles": residual_quantiles,
    }


def _classifier_pipeline(categorical_features: list[str], numeric_features: list[str], estimator) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_features),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", clone(estimator))])


def _regressor_pipeline(categorical_features: list[str], numeric_features: list[str], estimator) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_features),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", clone(estimator))])


def _classifier_candidates() -> list[dict]:
    candidates = [
        {"name": "logistic_regression", "estimator": LogisticRegression(max_iter=1500, class_weight="balanced"), "supports_sample_weight": True},
        {"name": "random_forest", "estimator": RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1), "supports_sample_weight": True},
        {"name": "random_forest_deep", "estimator": RandomForestClassifier(n_estimators=800, max_depth=None, min_samples_leaf=1, class_weight="balanced_subsample", random_state=42, n_jobs=-1), "supports_sample_weight": True},
        {"name": "extra_trees", "estimator": ExtraTreesClassifier(n_estimators=400, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1), "supports_sample_weight": True},
        {"name": "extra_trees_deep", "estimator": ExtraTreesClassifier(n_estimators=800, max_depth=None, min_samples_leaf=1, random_state=42, n_jobs=-1), "supports_sample_weight": True},
        {"name": "gradient_boosting", "estimator": GradientBoostingClassifier(random_state=42), "supports_sample_weight": True},
        {"name": "hist_gradient_boosting", "estimator": HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=250, random_state=42), "supports_sample_weight": True},
        {"name": "hist_gradient_boosting_deep", "estimator": HistGradientBoostingClassifier(max_depth=8, learning_rate=0.03, max_iter=450, random_state=42), "supports_sample_weight": True},
        {"name": "mlp", "estimator": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4, max_iter=400, early_stopping=True, random_state=42), "supports_sample_weight": False},
    ]
    if LGBMClassifier is not None:
        candidates.append(
            {
                "name": "lightgbm",
                "estimator": LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=42,
                    verbose=-1,
                ),
                "supports_sample_weight": True,
            }
        )
        candidates.append(
            {
                "name": "lightgbm_deep",
                "estimator": LGBMClassifier(
                    n_estimators=600,
                    learning_rate=0.03,
                    num_leaves=63,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    verbose=-1,
                ),
                "supports_sample_weight": True,
            }
        )
    if XGBClassifier is not None:
        candidates.append(
            {
                "name": "xgboost",
                "estimator": XGBClassifier(
                    n_estimators=250,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=4,
                    eval_metric="logloss",
                ),
                "supports_sample_weight": True,
            }
        )
        candidates.append(
            {
                "name": "xgboost_deep",
                "estimator": XGBClassifier(
                    n_estimators=500,
                    learning_rate=0.03,
                    max_depth=8,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=4,
                    eval_metric="logloss",
                ),
                "supports_sample_weight": True,
            }
        )
    if CatBoostClassifier is not None:
        candidates.append(
            {
                "name": "catboost",
                "estimator": CatBoostClassifier(
                    iterations=250,
                    depth=6,
                    learning_rate=0.05,
                    loss_function="Logloss",
                    random_seed=42,
                    verbose=False,
                ),
                "supports_sample_weight": True,
            }
        )
        candidates.append(
            {
                "name": "catboost_deep",
                "estimator": CatBoostClassifier(
                    iterations=500,
                    depth=8,
                    learning_rate=0.03,
                    loss_function="Logloss",
                    random_seed=42,
                    verbose=False,
                ),
                "supports_sample_weight": True,
            }
        )
    return candidates


def _regressor_candidates() -> list[dict]:
    candidates = [
        {"name": "random_forest", "estimator": RandomForestRegressor(n_estimators=400, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1), "supports_sample_weight": True},
        {"name": "extra_trees", "estimator": ExtraTreesRegressor(n_estimators=400, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1), "supports_sample_weight": True},
        {"name": "gradient_boosting", "estimator": GradientBoostingRegressor(random_state=42), "supports_sample_weight": True},
        {"name": "hist_gradient_boosting", "estimator": HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=250, random_state=42), "supports_sample_weight": True},
        {"name": "mlp", "estimator": MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4, max_iter=400, early_stopping=True, random_state=42), "supports_sample_weight": False},
    ]
    if LGBMRegressor is not None:
        candidates.append(
            {
                "name": "lightgbm",
                "estimator": LGBMRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=42,
                    verbose=-1,
                ),
                "supports_sample_weight": True,
            }
        )
    if XGBRegressor is not None:
        candidates.append(
            {
                "name": "xgboost",
                "estimator": XGBRegressor(
                    n_estimators=250,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=4,
                ),
                "supports_sample_weight": True,
            }
        )
    if CatBoostRegressor is not None:
        candidates.append(
            {
                "name": "catboost",
                "estimator": CatBoostRegressor(
                    iterations=250,
                    depth=6,
                    learning_rate=0.05,
                    loss_function="RMSE",
                    random_seed=42,
                    verbose=False,
                ),
                "supports_sample_weight": True,
            }
        )
    return candidates


def _fit_pipeline(
    pipeline: Pipeline,
    X,
    y,
    sample_weights: list[float],
    *,
    step_name: str,
    supports_sample_weight: bool,
) -> None:
    if supports_sample_weight:
        pipeline.fit(X, y, **{f"{step_name}__sample_weight": sample_weights})
        return
    pipeline.fit(X, y)


def _rolling_splits(length: int, *, windows: int, min_train: int, min_validation: int) -> list[tuple[int, int]]:
    if length < (min_train + min_validation):
        return []
    end_positions = np.linspace(min_train + min_validation, length, num=windows, dtype=int)
    splits: list[tuple[int, int]] = []
    for end_index in end_positions:
        val_start = max(min_train, end_index - min_validation)
        if val_start >= end_index:
            continue
        splits.append((val_start, end_index))
    return splits


def _rolling_classifier_score(
    frame: pd.DataFrame,
    labels: list[int],
    dates: list[date],
    candidate: dict,
    categorical_features: list[str],
    numeric_features: list[str],
    decay_lambda: float,
) -> float:
    scores: list[float] = []
    for val_start, val_end in _rolling_splits(len(frame), windows=3, min_train=20, min_validation=max(6, len(frame) // 10)):
        pipeline = _classifier_pipeline(categorical_features, numeric_features, candidate["estimator"])
        _fit_pipeline(
            pipeline,
            frame.iloc[:val_start],
            labels[:val_start],
            _sample_weights(dates[:val_start], dates[val_start - 1], decay_lambda),
            step_name="classifier",
            supports_sample_weight=candidate["supports_sample_weight"],
        )
        score = accuracy_score(labels[val_start:val_end], pipeline.predict(frame.iloc[val_start:val_end]))
        scores.append(float(score))
    return float(sum(scores) / len(scores)) if scores else 0.0


def _rolling_regressor_score(
    frame: pd.DataFrame,
    targets: list[float],
    dates: list[date],
    candidate: dict,
    categorical_features: list[str],
    numeric_features: list[str],
    decay_lambda: float,
) -> float:
    scores: list[float] = []
    for val_start, val_end in _rolling_splits(len(frame), windows=3, min_train=40, min_validation=max(20, len(frame) // 10)):
        pipeline = _regressor_pipeline(categorical_features, numeric_features, candidate["estimator"])
        _fit_pipeline(
            pipeline,
            frame.iloc[:val_start],
            targets[:val_start],
            _sample_weights(dates[:val_start], dates[val_start - 1], decay_lambda),
            step_name="regressor",
            supports_sample_weight=candidate["supports_sample_weight"],
        )
        score = mean_absolute_error(targets[val_start:val_end], pipeline.predict(frame.iloc[val_start:val_end]))
        scores.append(float(score))
    return float(sum(scores) / len(scores)) if scores else float("inf")


def _build_classifier_ensemble_candidate(
    candidate_results: list[dict],
    candidates: list[dict],
    frame: pd.DataFrame,
    labels: list[int],
    dates: list[date],
    train_X: pd.DataFrame,
    train_y: list[int],
    val_X: pd.DataFrame,
    val_y: list[int],
    categorical_features: list[str],
    numeric_features: list[str],
) -> dict | None:
    top_unique: list[dict] = []
    seen = set()
    for result in candidate_results:
        if result["name"] in seen:
            continue
        top_unique.append(result)
        seen.add(result["name"])
        if len(top_unique) == 2:
            break
    if len(top_unique) < 2:
        return None

    probabilities = []
    rolling_scores = []
    for result in top_unique:
        candidate = next(item for item in candidates if item["name"] == result["name"])
        pipeline = _classifier_pipeline(categorical_features, numeric_features, candidate["estimator"])
        _fit_pipeline(
            pipeline,
            train_X,
            train_y,
            _sample_weights(dates[: len(train_y)], dates[len(train_y) - 1], result["decay_lambda"]),
            step_name="classifier",
            supports_sample_weight=candidate["supports_sample_weight"],
        )
        probabilities.append(pipeline.predict_proba(val_X)[:, 1])
        rolling_scores.append(result["rolling_accuracy"])

    ensemble_probabilities = np.mean(np.column_stack(probabilities), axis=1)
    return {
        "kind": "ensemble",
        "name": f"ensemble:{top_unique[0]['name']}+{top_unique[1]['name']}",
        "members": [{"name": item["name"], "decay_lambda": item["decay_lambda"]} for item in top_unique],
        "rolling_accuracy": float(sum(rolling_scores) / len(rolling_scores)),
        "holdout_accuracy": float(accuracy_score(val_y, (ensemble_probabilities >= 0.5).astype(int))),
        "decay_lambda": top_unique[0]["decay_lambda"],
    }


def _train_classifier_ensemble(
    selected_result: dict,
    candidates: list[dict],
    frame: pd.DataFrame,
    labels: list[int],
    dates: list[date],
    categorical_features: list[str],
    numeric_features: list[str],
) -> dict:
    models = []
    for member in selected_result["members"]:
        candidate = next(item for item in candidates if item["name"] == member["name"])
        pipeline = _classifier_pipeline(categorical_features, numeric_features, candidate["estimator"])
        _fit_pipeline(
            pipeline,
            frame,
            labels,
            _sample_weights(dates, dates[-1], member["decay_lambda"]),
            step_name="classifier",
            supports_sample_weight=candidate["supports_sample_weight"],
        )
        models.append(
            {
                "name": member["name"],
                "pipeline": pipeline,
                "supports_sample_weight": candidate["supports_sample_weight"],
                "decay_lambda": member["decay_lambda"],
            }
        )
    return {
        "kind": "ensemble",
        "members": models,
        "calibration": {"method": "none"},
    }


def _fit_classifier_calibration(
    train_X: pd.DataFrame,
    train_y: list[int],
    val_X: pd.DataFrame,
    val_y: list[int],
    train_dates: list[date],
    max_date: date,
    decay_lambda: float,
    candidate: dict,
    categorical_features: list[str],
    numeric_features: list[str],
) -> dict:
    base_pipeline = _classifier_pipeline(categorical_features, numeric_features, candidate["estimator"])
    _fit_pipeline(
        base_pipeline,
        train_X,
        train_y,
        _sample_weights(train_dates, max_date, decay_lambda),
        step_name="classifier",
        supports_sample_weight=candidate["supports_sample_weight"],
    )
    base_probabilities = np.clip(base_pipeline.predict_proba(val_X)[:, 1], 1e-6, 1 - 1e-6)
    baseline_brier = brier_score_loss(val_y, base_probabilities)

    platt = LogisticRegression(max_iter=1000)
    platt.fit(base_probabilities.reshape(-1, 1), val_y)
    platt_probabilities = platt.predict_proba(base_probabilities.reshape(-1, 1))[:, 1]
    platt_brier = brier_score_loss(val_y, platt_probabilities)

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(base_probabilities, val_y)
    isotonic_probabilities = isotonic.predict(base_probabilities)
    isotonic_brier = brier_score_loss(val_y, isotonic_probabilities)

    if platt_brier < baseline_brier and platt_brier <= isotonic_brier:
        return {"method": "platt", "model": platt, "brier": float(platt_brier)}
    if isotonic_brier < baseline_brier:
        return {"method": "isotonic", "model": isotonic, "brier": float(isotonic_brier)}
    return {"method": "none", "brier": float(baseline_brier)}


def _predict_classifier_probability(model_info: dict, frame: pd.DataFrame) -> float:
    if model_info.get("model", {}).get("kind") == "ensemble":
        probabilities = [member["pipeline"].predict_proba(frame)[:, 1] for member in model_info["model"]["members"]]
        probability = float(np.mean(np.column_stack(probabilities), axis=1)[0])
        calibration = model_info["model"].get("calibration", {"method": "none"})
    else:
        pipeline = model_info.get("model", {}).get("pipeline", model_info.get("pipeline"))
        probability = float(pipeline.predict_proba(frame)[0][1])
        calibration = model_info.get("model", {}).get("calibration", {"method": "none"})
    return _apply_calibration(probability, calibration)


def _apply_calibration(probability: float, calibration: dict) -> float:
    method = calibration.get("method", "none")
    model = calibration.get("model")
    clipped = float(np.clip(probability, 1e-6, 1 - 1e-6))
    if method == "platt" and model is not None:
        return float(model.predict_proba(np.array([[clipped]]))[0][1])
    if method == "isotonic" and model is not None:
        return float(model.predict([clipped])[0])
    return clipped


def _regressor_residual_quantiles(
    train_X: pd.DataFrame,
    train_y: list[float],
    val_X: pd.DataFrame,
    val_y: list[float],
    train_dates: list[date],
    max_date: date,
    decay_lambda: float,
    candidate: dict,
    categorical_features: list[str],
    numeric_features: list[str],
) -> dict:
    pipeline = _regressor_pipeline(categorical_features, numeric_features, candidate["estimator"])
    _fit_pipeline(
        pipeline,
        train_X,
        train_y,
        _sample_weights(train_dates, max_date, decay_lambda),
        step_name="regressor",
        supports_sample_weight=candidate["supports_sample_weight"],
    )
    residuals = np.abs(np.array(val_y) - pipeline.predict(val_X))
    return {
        "p50": float(np.quantile(residuals, 0.5)),
        "p80": float(np.quantile(residuals, 0.8)),
        "p95": float(np.quantile(residuals, 0.95)),
    }


def _sample_weights(dates: list[date], max_date: date, decay_lambda: float | None = None) -> list[float]:
    settings = get_settings()
    lambda_value = settings.recency_decay_lambda if decay_lambda is None else decay_lambda
    return [math.exp(-lambda_value * max((max_date - item).days, 0)) for item in dates]


def _apply_match_result(state: TeamState, won: bool, maps_won: int, maps_lost: int, match_date: date, map_margin: int) -> None:
    state.matches += 1
    if won:
        state.wins += 1
    state.maps_won += maps_won
    state.maps_lost += maps_lost
    state.recent_results.append(1 if won else 0)
    state.recent_map_margin.append(map_margin)
    state.last_played = match_date


def _win_rate(state: TeamState) -> float:
    return state.wins / state.matches if state.matches else 0.5


def _map_win_rate(state: TeamState) -> float:
    total = state.maps_won + state.maps_lost
    return state.maps_won / total if total else 0.5


def _recent_win_rate(state: TeamState) -> float:
    return sum(state.recent_results) / len(state.recent_results) if state.recent_results else 0.5


def _recent_win_rate_window(state: TeamState, window: int) -> float:
    values = list(state.recent_results)[-window:]
    return sum(values) / len(values) if values else 0.5


def _recent_map_margin(state: TeamState) -> float:
    return sum(state.recent_map_margin) / len(state.recent_map_margin) if state.recent_map_margin else 0.0


def _recent_map_margin_window(state: TeamState, window: int) -> float:
    values = list(state.recent_map_margin)[-window:]
    return sum(values) / len(values) if values else 0.0


def _rest_days(state: TeamState, match_date: date) -> int:
    return max(0, (match_date - state.last_played).days) if state.last_played else 14


def _matchup_delta(team_a: str, team_b: str, source: dict[tuple[str, str], int]) -> int:
    pair_key = tuple(sorted((team_a, team_b)))
    value = source.get(pair_key, 0)
    return value if team_a <= team_b else -value


def _map_pool_overlap(team_a: str, team_b: str, context: dict) -> float:
    team_states = context["team_states"]
    state_a = team_states.get(team_a, TeamState())
    state_b = team_states.get(team_b, TeamState())
    maps = set(state_a.map_matches.keys()) | set(state_b.map_matches.keys())
    if not maps:
        return 0.0
    return sum(1.0 - abs(_team_map_win_rate(team_a, map_name, context) - _team_map_win_rate(team_b, map_name, context)) for map_name in maps) / len(maps)


def _best_map_win_rate(state: TeamState) -> float:
    values = [state.map_wins[map_name] / state.map_matches[map_name] for map_name in state.map_matches if state.map_matches[map_name] > 0]
    return max(values) if values else 0.5


def _map_pool_depth(state: TeamState) -> int:
    return sum(1 for map_name, matches in state.map_matches.items() if matches >= 3)


def _team_map_win_rate(team_name: str, map_name: str, context: dict) -> float:
    state = context["team_states"].get(team_name, TeamState())
    matches = state.map_matches.get(map_name, 0)
    return state.map_wins.get(map_name, 0) / matches if matches else 0.5


def _team_map_round_diff(team_name: str, map_name: str, context: dict) -> float:
    state = context["team_states"].get(team_name, TeamState())
    matches = state.map_matches.get(map_name, 0)
    return state.map_round_diff.get(map_name, 0) / matches if matches else 0.0


def _player_metric(player_state: PlayerState | None, metric: str, map_name: str | None = None, agent_name: str | None = None) -> float:
    if player_state is None:
        return 20.0 if metric == "kills" else 17.0 if metric == "deaths" else 200.0
    if map_name is not None:
        values = player_state.map_kills.get(map_name) if metric == "kills" else player_state.map_deaths.get(map_name)
        if values:
            return sum(values) / len(values)
    if agent_name is not None:
        values = player_state.agent_kills.get(agent_name) if metric == "kills" else player_state.agent_deaths.get(agent_name)
        if values:
            return sum(values) / len(values)
    values = player_state.recent_kills if metric == "kills" else player_state.recent_deaths if metric == "deaths" else player_state.recent_acs
    return sum(values) / len(values) if values else (20.0 if metric == "kills" else 17.0 if metric == "deaths" else 200.0)


def _team_player_metric(team_name: str, player_states: dict[str, PlayerState], metric: str) -> float:
    values = [_player_metric(state, metric) for state in player_states.values() if state.team_name == team_name]
    return sum(values) / len(values) if values else (20.0 if metric == "kills" else 17.0 if metric == "deaths" else 200.0)


def _team_player_map_metric(team_name: str, map_name: str, context: dict, metric: str) -> float:
    values = [_player_metric(state, metric, map_name=map_name) for state in context["player_states"].values() if state.team_name == team_name]
    return sum(values) / len(values) if values else (20.0 if metric == "kills" else 17.0 if metric == "deaths" else 200.0)


def _lineup_stability(team_name: str, context: dict) -> float:
    recent = context["team_recent_players"].get(team_name, {})
    total = sum(recent.values())
    if total <= 0:
        return 0.0
    return sum(sorted(recent.values(), reverse=True)[:5]) / total


def _rank_maps_for_team(team_name: str, context: dict) -> list[str]:
    state = context["team_states"].get(team_name, TeamState())
    return sorted(COMMON_MAPS, key=lambda map_name: (_team_map_win_rate(team_name, map_name, context), state.map_matches.get(map_name, 0)), reverse=True)


def _combined_map_rank(team_a: str, team_b: str, context: dict) -> list[str]:
    return sorted(
        COMMON_MAPS,
        key=lambda map_name: (
            _team_map_win_rate(team_a, map_name, context) + _team_map_win_rate(team_b, map_name, context) - abs(_team_map_win_rate(team_a, map_name, context) - _team_map_win_rate(team_b, map_name, context)),
            context["team_states"].get(team_a, TeamState()).map_matches.get(map_name, 0) + context["team_states"].get(team_b, TeamState()).map_matches.get(map_name, 0),
        ),
        reverse=True,
    )


def _likely_lineup(team_name: str, context: dict) -> list[str]:
    recent = context["team_recent_players"].get(team_name, {})
    if not recent:
        return [f"{team_name} Player {index}" for index in range(1, 6)]
    return [player for player, _ in sorted(recent.items(), key=lambda item: item[1], reverse=True)[:5]]


def _preferred_agent(player_name: str, context: dict) -> str | None:
    player_state = context["player_states"].get(player_name)
    if player_state is None:
        return None
    if player_state.last_agent:
        return player_state.last_agent
    if player_state.agent_kills:
        return max(player_state.agent_kills.items(), key=lambda item: sum(item[1]) / len(item[1]))[0]
    return None


def _clone_team_state(state: TeamState) -> TeamState:
    cloned = TeamState(
        matches=state.matches,
        wins=state.wins,
        maps_won=state.maps_won,
        maps_lost=state.maps_lost,
        recent_results=deque(state.recent_results, maxlen=RECENT_WINDOW),
        recent_map_margin=deque(state.recent_map_margin, maxlen=RECENT_WINDOW),
        elo=state.elo,
        last_played=state.last_played,
    )
    cloned.map_matches = defaultdict(int, dict(state.map_matches))
    cloned.map_wins = defaultdict(int, dict(state.map_wins))
    cloned.map_round_diff = defaultdict(int, dict(state.map_round_diff))
    return cloned


def _swap_fixture_teams(fixture: MatchFixture) -> MatchFixture:
    return MatchFixture(
        match_id=fixture.match_id,
        region=fixture.region,
        event_name=fixture.event_name,
        event_stage=fixture.event_stage,
        team_a=fixture.team_b,
        team_b=fixture.team_a,
        match_date=fixture.match_date,
        best_of=fixture.best_of,
    )


def _swap_picked_by_team(picked_by: str | None, fixture: MatchFixture) -> str | None:
    if picked_by is None:
        return None
    if picked_by == fixture.team_a:
        return fixture.team_b
    if picked_by == fixture.team_b:
        return fixture.team_a
    return picked_by


def _clone_player_state(state: PlayerState) -> PlayerState:
    cloned = PlayerState(
        team_name=state.team_name,
        maps_played=state.maps_played,
        recent_kills=deque(state.recent_kills, maxlen=RECENT_WINDOW),
        recent_deaths=deque(state.recent_deaths, maxlen=RECENT_WINDOW),
        recent_acs=deque(state.recent_acs, maxlen=RECENT_WINDOW),
        last_played=state.last_played,
        last_agent=state.last_agent,
    )
    cloned.map_kills = defaultdict(lambda: deque(maxlen=RECENT_WINDOW), {key: deque(value, maxlen=RECENT_WINDOW) for key, value in state.map_kills.items()})
    cloned.map_deaths = defaultdict(lambda: deque(maxlen=RECENT_WINDOW), {key: deque(value, maxlen=RECENT_WINDOW) for key, value in state.map_deaths.items()})
    cloned.agent_kills = defaultdict(lambda: deque(maxlen=RECENT_WINDOW), {key: deque(value, maxlen=RECENT_WINDOW) for key, value in state.agent_kills.items()})
    cloned.agent_deaths = defaultdict(lambda: deque(maxlen=RECENT_WINDOW), {key: deque(value, maxlen=RECENT_WINDOW) for key, value in state.agent_deaths.items()})
    return cloned


def _serialize_bundle(bundle: dict) -> dict:
    serialized = dict(bundle)
    serialized["context"] = _serialize_context(bundle["context"])
    return serialized


def _deserialize_bundle(bundle: dict) -> dict:
    deserialized = dict(bundle)
    if "context" in bundle:
        deserialized["context"] = _deserialize_context(bundle["context"])
    return deserialized


def _serialize_context(context: dict) -> dict:
    return {
        "team_states": {name: _serialize_team_state(state) for name, state in context["team_states"].items()},
        "player_states": {name: _serialize_player_state(state) for name, state in context["player_states"].items()},
        "head_to_head": list(context["head_to_head"].items()),
        "head_to_head_by_map": {
            map_name: list(values.items()) for map_name, values in context["head_to_head_by_map"].items()
        },
        "team_recent_players": {team: dict(players) for team, players in context["team_recent_players"].items()},
    }


def _deserialize_context(context: dict) -> dict:
    return {
        "team_states": {name: _deserialize_team_state(state) for name, state in context.get("team_states", {}).items()},
        "player_states": {name: _deserialize_player_state(state) for name, state in context.get("player_states", {}).items()},
        "head_to_head": {tuple(key): value for key, value in context.get("head_to_head", [])},
        "head_to_head_by_map": {
            map_name: {tuple(key): value for key, value in values}
            for map_name, values in context.get("head_to_head_by_map", {}).items()
        },
        "team_recent_players": {
            team: dict(players) for team, players in context.get("team_recent_players", {}).items()
        },
    }


def _serialize_team_state(state: TeamState) -> dict:
    return {
        "matches": state.matches,
        "wins": state.wins,
        "maps_won": state.maps_won,
        "maps_lost": state.maps_lost,
        "recent_results": list(state.recent_results),
        "recent_map_margin": list(state.recent_map_margin),
        "elo": state.elo,
        "last_played": state.last_played.isoformat() if state.last_played else None,
        "map_matches": dict(state.map_matches),
        "map_wins": dict(state.map_wins),
        "map_round_diff": dict(state.map_round_diff),
    }


def _deserialize_team_state(payload: dict) -> TeamState:
    state = TeamState(
        matches=payload.get("matches", 0),
        wins=payload.get("wins", 0),
        maps_won=payload.get("maps_won", 0),
        maps_lost=payload.get("maps_lost", 0),
        recent_results=deque(payload.get("recent_results", []), maxlen=RECENT_WINDOW),
        recent_map_margin=deque(payload.get("recent_map_margin", []), maxlen=RECENT_WINDOW),
        elo=payload.get("elo", INITIAL_ELO),
        last_played=date.fromisoformat(payload["last_played"]) if payload.get("last_played") else None,
    )
    state.map_matches = defaultdict(int, payload.get("map_matches", {}))
    state.map_wins = defaultdict(int, payload.get("map_wins", {}))
    state.map_round_diff = defaultdict(int, payload.get("map_round_diff", {}))
    return state


def _serialize_player_state(state: PlayerState) -> dict:
    return {
        "team_name": state.team_name,
        "maps_played": state.maps_played,
        "recent_kills": list(state.recent_kills),
        "recent_deaths": list(state.recent_deaths),
        "recent_acs": list(state.recent_acs),
        "map_kills": {key: list(value) for key, value in state.map_kills.items()},
        "map_deaths": {key: list(value) for key, value in state.map_deaths.items()},
        "agent_kills": {key: list(value) for key, value in state.agent_kills.items()},
        "agent_deaths": {key: list(value) for key, value in state.agent_deaths.items()},
        "last_played": state.last_played.isoformat() if state.last_played else None,
        "last_agent": state.last_agent,
    }


def _deserialize_player_state(payload: dict) -> PlayerState:
    state = PlayerState(
        team_name=payload.get("team_name", ""),
        maps_played=payload.get("maps_played", 0),
        recent_kills=deque(payload.get("recent_kills", []), maxlen=RECENT_WINDOW),
        recent_deaths=deque(payload.get("recent_deaths", []), maxlen=RECENT_WINDOW),
        recent_acs=deque(payload.get("recent_acs", []), maxlen=RECENT_WINDOW),
        last_played=date.fromisoformat(payload["last_played"]) if payload.get("last_played") else None,
        last_agent=payload.get("last_agent"),
    )
    state.map_kills = defaultdict(
        lambda: deque(maxlen=RECENT_WINDOW),
        {key: deque(value, maxlen=RECENT_WINDOW) for key, value in payload.get("map_kills", {}).items()},
    )
    state.map_deaths = defaultdict(
        lambda: deque(maxlen=RECENT_WINDOW),
        {key: deque(value, maxlen=RECENT_WINDOW) for key, value in payload.get("map_deaths", {}).items()},
    )
    state.agent_kills = defaultdict(
        lambda: deque(maxlen=RECENT_WINDOW),
        {key: deque(value, maxlen=RECENT_WINDOW) for key, value in payload.get("agent_kills", {}).items()},
    )
    state.agent_deaths = defaultdict(
        lambda: deque(maxlen=RECENT_WINDOW),
        {key: deque(value, maxlen=RECENT_WINDOW) for key, value in payload.get("agent_deaths", {}).items()},
    )
    return state
