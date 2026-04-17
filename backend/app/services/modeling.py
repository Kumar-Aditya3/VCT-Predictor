from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date
import math
import os
import pickle
from pathlib import Path
from typing import Any

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
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error
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

try:
    import optuna
except ImportError:  # pragma: no cover - optional dependency
    optuna = None

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None


MODEL_FILE_NAME = "prediction_bundle.pkl"
RECENT_WINDOW = 8
INITIAL_ELO = 1500.0
ELO_K = 24.0
COMMON_MAPS = ("Abyss", "Ascent", "Bind", "Haven", "Icebox", "Lotus", "Pearl", "Split", "Sunset")
SHORT_WINDOW = 3
MEDIUM_WINDOW = 5
MATCH_DECAY_GRID = (0.005, 0.01, 0.015, 0.02, 0.03, 0.04)
MAP_DECAY_GRID = (0.005, 0.01, 0.015, 0.02, 0.03, 0.04)
PLAYER_DECAY_GRID = (0.005, 0.01, 0.015, 0.02, 0.03, 0.04)
WIN_PRIOR_ALPHA = 2.0
WIN_PRIOR_BETA = 2.0
MAP_PRIOR_ALPHA = 1.5
MAP_PRIOR_BETA = 1.5
MAP_RELIABILITY_TARGET = 6.0
MATCH_MIN_TRAIN = 20
MAP_MIN_TRAIN = 40
PLAYER_MIN_TRAIN = 100
ROLLING_WINDOWS = 4
MATCH_OPTUNA_TRIALS = 6
REGRESSOR_OPTUNA_TRIALS = 6
PLASTICITY_BASE_WEIGHT = 0.08
PLASTICITY_RELIABILITY_WEIGHT = 0.18
PLASTICITY_MAX_SHIFT = 0.16
PLASTICITY_EDGE_DAMPING = 1.6


@dataclass
class TeamState:
    matches: int = 0
    wins: int = 0
    maps_won: int = 0
    maps_lost: int = 0
    recent_results: deque[int] = field(default_factory=lambda: deque(maxlen=RECENT_WINDOW))
    recent_map_margin: deque[int] = field(default_factory=lambda: deque(maxlen=RECENT_WINDOW))
    elo: float = INITIAL_ELO
    opponent_elo_sum: float = 0.0
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


@dataclass
class FoldSummary:
    index: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    train_size: int
    validation_size: int


def train_prediction_bundle(
    matches: list[VLRMatchRecord],
    maps: list[VLRMapRecord],
    player_stats: list[VLRPlayerStatLine],
) -> dict | None:
    completed_matches = [match for match in matches if match.status == "completed"]
    completed_maps = [map_record for map_record in maps if map_record.winner_team in {map_record.team_a, map_record.team_b}]
    completed_map_ids = {map_record.map_id for map_record in completed_maps}
    validated_player_rows = [
        stat
        for stat in player_stats
        if stat.kills >= 0 and stat.deaths >= 0 and stat.map_id in completed_map_ids
    ]
    if len(completed_matches) < MATCH_MIN_TRAIN or len(completed_maps) < MAP_MIN_TRAIN:
        return None

    ordered_matches = sorted(completed_matches, key=lambda item: (item.match_date, item.match_id))
    maps_by_match: dict[str, list[VLRMapRecord]] = defaultdict(list)
    stats_by_map: dict[str, list[VLRPlayerStatLine]] = defaultdict(list)
    for map_record in completed_maps:
        maps_by_match[map_record.match_id].append(map_record)
    for map_list in maps_by_match.values():
        map_list.sort(key=lambda item: item.order_index)
    for stat in validated_player_rows:
        stats_by_map[stat.map_id].append(stat)

    feature_store = _build_feature_store(ordered_matches, maps_by_match, stats_by_map)
    if len(feature_store["match_rows"]) < MATCH_MIN_TRAIN or len(feature_store["map_rows"]) < MAP_MIN_TRAIN:
        return None
    dataset_snapshot = _dataset_snapshot(ordered_matches, completed_maps, validated_player_rows)

    match_bundle = _train_classifier_bundle(
        feature_store["match_rows"],
        feature_store["match_labels"],
        feature_store["match_dates"],
        f"vlr-match-{ordered_matches[-1].match_date.isoformat()}",
        task_name="match_winner",
        decay_values=MATCH_DECAY_GRID,
    )
    map_bundle = _train_classifier_bundle(
        feature_store["map_rows"],
        feature_store["map_labels"],
        feature_store["map_dates"],
        f"vlr-map-{ordered_matches[-1].match_date.isoformat()}",
        task_name="map_winner",
        decay_values=MAP_DECAY_GRID,
    )
    player_kills_bundle = None
    player_deaths_bundle = None
    if len(feature_store["player_rows"]) >= PLAYER_MIN_TRAIN and not _is_fast_model_search():
        player_kills_bundle = _train_regressor_bundle(
            feature_store["player_rows"],
            feature_store["player_kills"],
            feature_store["player_dates"],
            f"vlr-player-kills-{ordered_matches[-1].match_date.isoformat()}",
            task_name="player_kills",
            decay_values=PLAYER_DECAY_GRID,
        )
        player_deaths_bundle = _train_regressor_bundle(
            feature_store["player_rows"],
            feature_store["player_deaths"],
            feature_store["player_dates"],
            f"vlr-player-deaths-{ordered_matches[-1].match_date.isoformat()}",
            task_name="player_deaths",
            decay_values=PLAYER_DECAY_GRID,
        )
    if match_bundle is None or map_bundle is None:
        return None

    player_mae = ((player_kills_bundle["mae"] + player_deaths_bundle["mae"]) / 2) if (player_kills_bundle and player_deaths_bundle) else 0.0
    player_cv_score = max(0.0, 1.0 - player_mae / 25.0) if (player_kills_bundle and player_deaths_bundle) else 0.0
    trained_at = pd.Timestamp.utcnow().isoformat()
    model_version = f"vlr-core-{pd.Timestamp.utcnow().strftime('%Y%m%d-%H%M%S')}"
    return {
        "model_version": model_version,
        "trained_at": trained_at,
        "history_start": ordered_matches[0].match_date.isoformat(),
        "history_end": ordered_matches[-1].match_date.isoformat(),
        "match_model": match_bundle,
        "map_model": map_bundle,
        "player_kills_model": player_kills_bundle,
        "player_deaths_model": player_deaths_bundle,
        "dataset_snapshot": dataset_snapshot,
        "experiment": {
            "id": f"experiment-{pd.Timestamp.utcnow().strftime('%Y%m%d-%H%M%S')}",
            "selection_metric": {
                "match_winner": "rolling_log_loss",
                "map_winner": "rolling_log_loss",
                "player_kills": "rolling_mae",
                "player_deaths": "rolling_mae",
            },
            "task_winners": {
                "match_winner": match_bundle["estimator_name"],
                "map_winner": map_bundle["estimator_name"],
                "player_kills": player_kills_bundle["estimator_name"] if player_kills_bundle is not None else "baseline_fallback",
                "player_deaths": player_deaths_bundle["estimator_name"] if player_deaths_bundle is not None else "baseline_fallback",
            },
        },
        "metrics": {
            "winner_accuracy": match_bundle["accuracy"],
            "winner_log_loss": match_bundle["log_loss"],
            "winner_brier": match_bundle["brier"],
            "map_accuracy": map_bundle["accuracy"],
            "map_log_loss": map_bundle["log_loss"],
            "map_brier": map_bundle["brier"],
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
            "rolling_winner_log_loss": match_bundle["rolling_log_loss"],
            "rolling_winner_brier": match_bundle["rolling_brier"],
            "rolling_map_accuracy": map_bundle["rolling_accuracy"],
            "rolling_map_log_loss": map_bundle["rolling_log_loss"],
            "rolling_map_brier": map_bundle["rolling_brier"],
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
            "rolling_winner_log_loss": match_bundle["rolling_log_loss"],
            "rolling_winner_brier": match_bundle["rolling_brier"],
            "rolling_map_accuracy": map_bundle["rolling_accuracy"],
            "rolling_map_log_loss": map_bundle["rolling_log_loss"],
            "rolling_map_brier": map_bundle["rolling_brier"],
            "rolling_player_kills_mae": player_kills_bundle["rolling_mae"] if player_kills_bundle is not None else 0.0,
            "rolling_player_deaths_mae": player_deaths_bundle["rolling_mae"] if player_deaths_bundle is not None else 0.0,
            "rolling_windows_evaluated": len(match_bundle.get("folds", [])),
        },
        "calibration": {
            "match": match_bundle["calibration_method"],
            "map": map_bundle["calibration_method"],
        },
        "residuals": {
            "player_kills": player_kills_bundle["residual_quantiles"] if player_kills_bundle is not None else {},
            "player_deaths": player_deaths_bundle["residual_quantiles"] if player_deaths_bundle is not None else {},
        },
        "folds": {
            "match_winner": match_bundle.get("folds", []),
            "map_winner": map_bundle.get("folds", []),
            "player_kills": player_kills_bundle.get("folds", []) if player_kills_bundle is not None else [],
            "player_deaths": player_deaths_bundle.get("folds", []) if player_deaths_bundle is not None else [],
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
    context = model_bundle["context"]
    forward_row = build_match_feature_row(fixture, context)
    forward_frame = pd.DataFrame([forward_row], columns=model_bundle["match_model"]["feature_columns"])
    forward_probability = _predict_classifier_probability(model_bundle["match_model"], forward_frame)

    reverse_fixture = _swap_fixture_teams(fixture)
    reverse_row = build_match_feature_row(reverse_fixture, context)
    reverse_frame = pd.DataFrame([reverse_row], columns=model_bundle["match_model"]["feature_columns"])
    reverse_probability = _predict_classifier_probability(model_bundle["match_model"], reverse_frame)

    # Make inference invariant to Team A / Team B ordering.
    invariant_probability = (forward_probability + (1.0 - reverse_probability)) / 2.0
    prior_delta = _context_matchup_prior(fixture, context)
    blended_probability = invariant_probability + (0.2 * prior_delta)
    plasticity_delta = _plasticity_adjustment(fixture, context, blended_probability)
    return float(np.clip(blended_probability + plasticity_delta, 1e-6, 1 - 1e-6))


def _context_matchup_prior(fixture: MatchFixture, context: dict) -> float:
    team_states = context["team_states"]
    player_states = context["player_states"]
    head_to_head = context["head_to_head"]
    team_a_state = team_states.get(fixture.team_a, TeamState())
    team_b_state = team_states.get(fixture.team_b, TeamState())

    # Bounded prior signal used to guide close calls with robust context.
    signal = (
        0.45 * (_win_rate(team_a_state) - _win_rate(team_b_state))
        + 0.35 * (_matchup_delta(fixture.team_a, fixture.team_b, head_to_head) / 3.0)
        + 0.2 * ((_team_player_metric(fixture.team_a, player_states, "kills") - _team_player_metric(fixture.team_b, player_states, "kills")) / 5.0)
        - 0.2 * ((_team_player_metric(fixture.team_a, player_states, "deaths") - _team_player_metric(fixture.team_b, player_states, "deaths")) / 5.0)
        + 0.15 * (_lineup_stability(fixture.team_a, context) - _lineup_stability(fixture.team_b, context))
    )
    return float(np.clip(signal, -0.2, 0.2))


def _team_form_trend(state: TeamState) -> float:
    short_wr = _recent_win_rate_window(state, SHORT_WINDOW)
    medium_wr = _recent_win_rate_window(state, MEDIUM_WINDOW)
    long_wr = _adjusted_win_rate(state)
    margin_trend = math.tanh(_recent_map_margin_window(state, SHORT_WINDOW) / 8.0)
    trend = (0.6 * (short_wr - long_wr)) + (0.4 * (medium_wr - long_wr)) + (0.15 * margin_trend)
    return float(np.clip(trend, -0.8, 0.8))


def _team_recent_reliability(team_name: str, state: TeamState, context: dict) -> float:
    recency_samples = min(1.0, len(state.recent_results) / MEDIUM_WINDOW)
    lineup_reliability = max(0.0, min(1.0, _lineup_stability(team_name, context)))
    return recency_samples * (0.5 + (0.5 * lineup_reliability))


def _plasticity_adjustment(fixture: MatchFixture, context: dict, base_probability: float) -> float:
    team_states = context["team_states"]
    team_a_state = team_states.get(fixture.team_a, TeamState())
    team_b_state = team_states.get(fixture.team_b, TeamState())

    trend_delta = _team_form_trend(team_a_state) - _team_form_trend(team_b_state)
    reliability = min(
        _team_recent_reliability(fixture.team_a, team_a_state, context),
        _team_recent_reliability(fixture.team_b, team_b_state, context),
    )
    edge_damp = max(0.35, 1.0 - (PLASTICITY_EDGE_DAMPING * abs(base_probability - 0.5)))
    weight = PLASTICITY_BASE_WEIGHT + (PLASTICITY_RELIABILITY_WEIGHT * reliability)
    return float(np.clip(trend_delta * weight * edge_damp, -PLASTICITY_MAX_SHIFT, PLASTICITY_MAX_SHIFT))


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
    has_player_models = model_bundle.get("player_kills_model") is not None and model_bundle.get("player_deaths_model") is not None
    context = model_bundle["context"]
    match_probability = predict_match_probability(fixture, model_bundle)
    team_a_players = _likely_lineup(fixture.team_a, context)
    team_b_players = _likely_lineup(fixture.team_b, context)
    projections: list[dict] = []

    for team_name, opponent_team, players in (
        (fixture.team_a, fixture.team_b, team_a_players),
        (fixture.team_b, fixture.team_a, team_b_players),
    ):
        for player_name in players:
            agent_name = _preferred_agent(player_name, context)
            team_win_bias = match_probability if team_name == fixture.team_a else (1.0 - match_probability)
            for map_name, picked_by in selected_maps:
                if has_player_models:
                    row = build_player_feature_row(fixture, team_name, opponent_team, player_name, map_name, agent_name, context)
                    frame = pd.DataFrame([row], columns=model_bundle["player_kills_model"]["feature_columns"])
                    kills = float(model_bundle["player_kills_model"]["pipeline"].predict(frame)[0])
                    deaths = float(model_bundle["player_deaths_model"]["pipeline"].predict(frame)[0])
                else:
                    kills, deaths = _baseline_player_stat_projection(
                        team_name,
                        opponent_team,
                        player_name,
                        map_name,
                        picked_by,
                        agent_name,
                        context,
                        team_win_bias,
                    )
                projections.append(
                    {
                        "player_name": player_name,
                        "team_name": team_name,
                        "map_name": map_name,
                        "agent_name": agent_name,
                        "projected_kills": round(max(0.0, kills), 1),
                        "projected_deaths": round(max(0.0, deaths), 1),
                    }
                )
    return projections


def _baseline_player_stat_projection(
    team_name: str,
    opponent_team: str,
    player_name: str,
    map_name: str,
    picked_by: str | None,
    agent_name: str | None,
    context: dict,
    team_win_bias: float,
) -> tuple[float, float]:
    player_state = context["player_states"].get(player_name)
    base_kills = _player_metric(player_state, "kills", map_name=map_name, agent_name=agent_name)
    base_deaths = _player_metric(player_state, "deaths", map_name=map_name, agent_name=agent_name)

    map_edge = _team_map_win_rate(team_name, map_name, context) - _team_map_win_rate(opponent_team, map_name, context)
    picked_bonus = 0.03 if picked_by == team_name else -0.03 if picked_by == opponent_team else 0.0
    win_bias = team_win_bias - 0.5
    stability_edge = _lineup_stability(team_name, context) - _lineup_stability(opponent_team, context)

    kills_factor = 1.0 + (0.16 * win_bias) + (0.12 * map_edge) + picked_bonus + (0.05 * stability_edge)
    deaths_factor = 1.0 - (0.11 * win_bias) - (0.08 * map_edge) - (picked_bonus * 0.7)

    kills = base_kills * max(0.75, min(1.35, kills_factor))
    deaths = base_deaths * max(0.7, min(1.3, deaths_factor))
    return kills, deaths


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
    stage_bucket = _normalize_event_stage(fixture.event_stage)

    return {
        "region": fixture.region,
        "event_name": fixture.event_name,
        "event_stage": fixture.event_stage or "unknown",
        "event_stage_bucket": stage_bucket,
        "is_international_event": 1 if fixture.region == "International" or fixture.event_name in {"Masters", "Champions"} else 0,
        "best_of": fixture.best_of,
        "elo_diff": team_a_state.elo - team_b_state.elo,
        "event_elo_diff": _event_elo(fixture.team_a, fixture.event_name, context) - _event_elo(fixture.team_b, fixture.event_name, context),
        "experience_index_diff": _experience_index(team_a_state) - _experience_index(team_b_state),
        "win_rate_diff": _win_rate(team_a_state) - _win_rate(team_b_state),
        "adjusted_win_rate_diff": _adjusted_win_rate(team_a_state) - _adjusted_win_rate(team_b_state),
        "strength_of_schedule_diff": _strength_of_schedule(team_a_state) - _strength_of_schedule(team_b_state),
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
        "lineup_continuity_diff": _lineup_continuity(fixture.team_a, context) - _lineup_continuity(fixture.team_b, context),
    }


def build_map_feature_row(fixture: MatchFixture, map_name: str, picked_by: str | None, context: dict) -> dict:
    team_states = context["team_states"]
    head_to_head_by_map = context["head_to_head_by_map"]
    return {
        "region": fixture.region,
        "event_name": fixture.event_name,
        "event_stage_bucket": _normalize_event_stage(fixture.event_stage),
        "map_name": map_name,
        "picked_by": picked_by or "decider",
        "elo_diff": team_states.get(fixture.team_a, TeamState()).elo - team_states.get(fixture.team_b, TeamState()).elo,
        "event_elo_diff": _event_elo(fixture.team_a, fixture.event_name, context) - _event_elo(fixture.team_b, fixture.event_name, context),
        "team_map_event_form_diff": _event_map_strength(fixture.team_a, map_name, fixture.event_name, context) - _event_map_strength(fixture.team_b, map_name, fixture.event_name, context),
        "team_map_win_rate_diff": _team_map_win_rate(fixture.team_a, map_name, context) - _team_map_win_rate(fixture.team_b, map_name, context),
        "team_map_round_diff": _team_map_round_diff(fixture.team_a, map_name, context) - _team_map_round_diff(fixture.team_b, map_name, context),
        "recent_team_form_diff": _recent_win_rate(team_states.get(fixture.team_a, TeamState())) - _recent_win_rate(team_states.get(fixture.team_b, TeamState())),
        "map_head_to_head_delta": _matchup_delta(fixture.team_a, fixture.team_b, head_to_head_by_map.get(map_name, {})),
        "player_map_acs_diff": _team_player_map_metric(fixture.team_a, map_name, context, "acs") - _team_player_map_metric(fixture.team_b, map_name, context, "acs"),
        "player_map_kills_diff": _team_player_map_metric(fixture.team_a, map_name, context, "kills") - _team_player_map_metric(fixture.team_b, map_name, context, "kills"),
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
    team_state = context["team_states"].get(team_name, TeamState())
    opponent_state = context["team_states"].get(opponent_team, TeamState())
    return {
        "region": fixture.region,
        "event_name": fixture.event_name,
        "event_stage_bucket": _normalize_event_stage(fixture.event_stage),
        "map_name": map_name,
        "agent_name": agent_name or "unknown",
        "agent_role": _agent_role(agent_name),
        "team_name": team_name,
        "opponent_team": opponent_team,
        "team_elo": team_state.elo,
        "opponent_elo": opponent_state.elo,
        "event_team_elo": _event_elo(team_name, fixture.event_name, context),
        "event_opponent_elo": _event_elo(opponent_team, fixture.event_name, context),
        "player_recent_kills": _player_metric(player_state, "kills"),
        "player_recent_deaths": _player_metric(player_state, "deaths"),
        "player_recent_acs": _player_metric(player_state, "acs"),
        "player_map_kills": _player_metric(player_state, "kills", map_name=map_name),
        "player_map_deaths": _player_metric(player_state, "deaths", map_name=map_name),
        "player_agent_kills": _player_metric(player_state, "kills", agent_name=agent_name),
        "player_agent_deaths": _player_metric(player_state, "deaths", agent_name=agent_name),
        "player_maps_played": player_state.maps_played if player_state is not None else 0,
        "player_missing_history": 0 if player_state is not None else 1,
        "team_map_win_rate": _team_map_win_rate(team_name, map_name, context),
        "opponent_map_win_rate": _team_map_win_rate(opponent_team, map_name, context),
        "lineup_stability": _lineup_stability(team_name, context),
        "lineup_continuity": _lineup_continuity(team_name, context),
        "team_recent_win_rate": _recent_win_rate(team_state),
        "opponent_recent_win_rate": _recent_win_rate(opponent_state),
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
    team_event_history: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: {"matches": 0.0, "wins": 0.0}))
    team_event_map_history: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: {"matches": 0.0, "wins": 0.0}))
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
            "team_event_history": team_event_history,
            "team_event_map_history": team_event_map_history,
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
            _update_map_and_player_states(
                map_record,
                stats_by_map.get(map_record.map_id, []),
                team_states,
                player_states,
                head_to_head_by_map,
                team_recent_players,
                team_event_map_history,
                match.event_name,
                match.match_date,
            )

        _update_match_states(match, team_states, head_to_head, team_event_history)

    store["context"] = {
        "team_states": {name: _clone_team_state(state) for name, state in team_states.items()},
        "player_states": {name: _clone_player_state(state) for name, state in player_states.items()},
        "head_to_head": dict(head_to_head),
        "head_to_head_by_map": {map_name: dict(values) for map_name, values in head_to_head_by_map.items()},
        "team_recent_players": {team: dict(players) for team, players in team_recent_players.items()},
        "team_event_history": {team: {event: dict(values) for event, values in events.items()} for team, events in team_event_history.items()},
        "team_event_map_history": {team: {key: dict(values) for key, values in maps.items()} for team, maps in team_event_map_history.items()},
    }
    return store


def _update_match_states(
    match: VLRMatchRecord,
    team_states: dict[str, TeamState],
    head_to_head: dict[tuple[str, str], int],
    team_event_history: dict[str, dict[str, dict[str, float]]],
) -> None:
    team_a_state = team_states.setdefault(match.team_a, TeamState())
    team_b_state = team_states.setdefault(match.team_b, TeamState())
    team_a_won = match.team_a_maps_won > match.team_b_maps_won
    expected_a = 1 / (1 + 10 ** ((team_b_state.elo - team_a_state.elo) / 400))
    elo_change = ELO_K * ((1.0 if team_a_won else 0.0) - expected_a)
    team_a_state.elo += elo_change
    team_b_state.elo -= elo_change
    margin = match.team_a_maps_won - match.team_b_maps_won
    _apply_match_result(team_a_state, team_a_won, match.team_a_maps_won, match.team_b_maps_won, match.match_date, margin, team_b_state.elo)
    _apply_match_result(team_b_state, not team_a_won, match.team_b_maps_won, match.team_a_maps_won, match.match_date, -margin, team_a_state.elo)
    pair_key = tuple(sorted((match.team_a, match.team_b)))
    if team_a_won:
        head_to_head[pair_key] += 1 if match.team_a <= match.team_b else -1
    else:
        head_to_head[pair_key] -= 1 if match.team_a <= match.team_b else -1
    team_event_history[match.team_a][match.event_name]["matches"] += 1.0
    team_event_history[match.team_b][match.event_name]["matches"] += 1.0
    if team_a_won:
        team_event_history[match.team_a][match.event_name]["wins"] += 1.0
    else:
        team_event_history[match.team_b][match.event_name]["wins"] += 1.0


def _update_map_and_player_states(
    map_record: VLRMapRecord,
    stats: list[VLRPlayerStatLine],
    team_states: dict[str, TeamState],
    player_states: dict[str, PlayerState],
    head_to_head_by_map: dict[str, dict[tuple[str, str], int]],
    team_recent_players: dict[str, dict[str, float]],
    team_event_map_history: dict[str, dict[str, dict[str, float]]],
    event_name: str,
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
    map_event_key = f"{event_name}:{map_record.map_name}"
    team_event_map_history[map_record.team_a][map_event_key]["matches"] += 1.0
    team_event_map_history[map_record.team_b][map_event_key]["matches"] += 1.0
    if map_record.winner_team == map_record.team_a:
        team_event_map_history[map_record.team_a][map_event_key]["wins"] += 1.0
    elif map_record.winner_team == map_record.team_b:
        team_event_map_history[map_record.team_b][map_event_key]["wins"] += 1.0

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
    task_name: str,
    decay_values: tuple[float, ...] = (0.005, 0.01, 0.02),
) -> dict | None:
    if len(rows) < MATCH_MIN_TRAIN:
        return None
    frame = pd.DataFrame(rows)
    split_index = max(MATCH_MIN_TRAIN, int(len(frame) * 0.82))
    split_index = min(split_index, len(frame) - 1)
    train_X = frame.iloc[:split_index]
    val_X = frame.iloc[split_index:]
    train_y = labels[:split_index]
    val_y = labels[split_index:]
    if not val_y or len(train_y) < MATCH_MIN_TRAIN:
        return None
    categorical_features = [col for col in frame.columns if frame[col].dtype == object]
    numeric_features = [col for col in frame.columns if col not in categorical_features]
    candidates = _classifier_candidates()
    candidates = _trim_classifier_candidates(candidates, len(frame))
    decay_values = _trim_decay_values(decay_values, len(frame))
    candidates.extend(_refine_classifier_candidates(task_name, frame, labels, dates, candidates, categorical_features, numeric_features))
    candidate_results: list[dict] = []
    fold_descriptors = _rolling_fold_descriptors(dates, windows=ROLLING_WINDOWS, min_train=MATCH_MIN_TRAIN, min_validation=max(4, len(frame) // 12))
    for candidate in candidates:
        for decay_lambda in decay_values:
            evaluation = _rolling_classifier_score(
                frame,
                labels,
                dates,
                candidate,
                categorical_features,
                numeric_features,
                decay_lambda,
            )
            if not evaluation["folds"]:
                continue
            pipeline = _classifier_pipeline(categorical_features, numeric_features, candidate["estimator"])
            _fit_pipeline(
                pipeline,
                train_X,
                train_y,
                _sample_weights(dates[:split_index], dates[-1], decay_lambda),
                step_name="classifier",
                supports_sample_weight=candidate["supports_sample_weight"],
            )
            probabilities = pipeline.predict_proba(val_X)[:, 1]
            holdout_accuracy = accuracy_score(val_y, (probabilities >= 0.5).astype(int))
            candidate_results.append(
                {
                    "name": candidate["name"],
                    "decay_lambda": decay_lambda,
                    "rolling_accuracy": float(evaluation["accuracy"]),
                    "rolling_log_loss": float(evaluation["log_loss"]),
                    "rolling_brier": float(evaluation["brier"]),
                    "rolling_accuracy_std": float(evaluation["accuracy_std"]),
                    "rolling_log_loss_std": float(evaluation["log_loss_std"]),
                    "rolling_brier_std": float(evaluation["brier_std"]),
                    "holdout_accuracy": float(holdout_accuracy),
                    "holdout_log_loss": _safe_log_loss(val_y, probabilities),
                    "brier": float(brier_score_loss(val_y, probabilities)),
                    "folds": evaluation["folds"],
                }
            )

    if not candidate_results:
        return None
    candidate_results.sort(
        key=lambda item: (
            item["rolling_log_loss"],
            item["rolling_brier"],
            -item["rolling_accuracy"],
            item["rolling_log_loss_std"],
            item["holdout_log_loss"],
        )
    )
    best_result = candidate_results[0]
    best_candidate = next(candidate for candidate in candidates if candidate["name"] == best_result["name"])
    selected_result = best_result

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
        "log_loss": float(selected_result["holdout_log_loss"]),
        "brier": float(selected_result["brier"]),
        "version": version,
        "training_samples": len(frame),
        "validation_samples": len(val_y),
        "rolling_accuracy": float(selected_result["rolling_accuracy"]),
        "rolling_log_loss": float(selected_result["rolling_log_loss"]),
        "rolling_brier": float(selected_result["rolling_brier"]),
        "estimator_name": selected_result["name"],
        "candidate_scores": {f"{item['name']}@{item['decay_lambda']:.3f}": item["rolling_log_loss"] for item in candidate_results},
        "candidate_results": candidate_results[:12],
        "calibration_method": final_model.get("calibration", {}).get("method", "none"),
        "selected_decay_lambda": selected_result["decay_lambda"],
        "selection_objective": "rolling_log_loss",
        "folds": fold_descriptors,
    }


def _train_regressor_bundle(
    rows: list[dict],
    targets: list[float],
    dates: list[date],
    version: str,
    *,
    task_name: str,
    decay_values: tuple[float, ...] = (0.005, 0.01, 0.02),
) -> dict | None:
    if len(rows) < PLAYER_MIN_TRAIN:
        return None
    frame = pd.DataFrame(rows)
    split_index = max(PLAYER_MIN_TRAIN, int(len(frame) * 0.82))
    split_index = min(split_index, len(frame) - 1)
    train_X = frame.iloc[:split_index]
    val_X = frame.iloc[split_index:]
    train_y = targets[:split_index]
    val_y = targets[split_index:]
    if not val_y:
        return None
    categorical_features = [col for col in frame.columns if frame[col].dtype == object]
    numeric_features = [col for col in frame.columns if col not in categorical_features]
    candidates = _regressor_candidates()
    candidates = _trim_regressor_candidates(candidates, len(frame))
    decay_values = _trim_decay_values(decay_values, len(frame))
    candidates.extend(_refine_regressor_candidates(task_name, frame, targets, dates, candidates, categorical_features, numeric_features))
    candidate_results: list[dict] = []
    fold_descriptors = _rolling_fold_descriptors(dates, windows=ROLLING_WINDOWS, min_train=PLAYER_MIN_TRAIN, min_validation=max(12, len(frame) // 14))
    for candidate in candidates:
        for decay_lambda in decay_values:
            evaluation = _rolling_regressor_score(
                frame,
                targets,
                dates,
                candidate,
                categorical_features,
                numeric_features,
                decay_lambda,
            )
            if not evaluation["folds"]:
                continue
            pipeline = _regressor_pipeline(categorical_features, numeric_features, candidate["estimator"])
            _fit_pipeline(
                pipeline,
                train_X,
                train_y,
                _sample_weights(dates[:split_index], dates[-1], decay_lambda),
                step_name="regressor",
                supports_sample_weight=candidate["supports_sample_weight"],
            )
            predictions = pipeline.predict(val_X)
            holdout_mae = mean_absolute_error(val_y, predictions)
            candidate_results.append(
                {
                    "name": candidate["name"],
                    "decay_lambda": decay_lambda,
                    "rolling_mae": float(evaluation["mae"]),
                    "rolling_rmse": float(evaluation["rmse"]),
                    "rolling_mae_std": float(evaluation["mae_std"]),
                    "rolling_rmse_std": float(evaluation["rmse_std"]),
                    "holdout_mae": float(holdout_mae),
                    "holdout_rmse": _safe_rmse(val_y, predictions),
                    "folds": evaluation["folds"],
                }
            )

    if not candidate_results:
        return None
    candidate_results.sort(key=lambda item: (item["rolling_mae"], item["rolling_rmse"], item["rolling_mae_std"], item["holdout_mae"]))
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
        "rmse": float(best_result["holdout_rmse"]),
        "version": version,
        "training_samples": len(frame),
        "validation_samples": len(val_y),
        "rolling_mae": float(best_result["rolling_mae"]),
        "rolling_rmse": float(best_result["rolling_rmse"]),
        "estimator_name": best_result["name"],
        "candidate_scores": {f"{item['name']}@{item['decay_lambda']:.3f}": item["rolling_mae"] for item in candidate_results},
        "candidate_results": candidate_results[:12],
        "selected_decay_lambda": best_result["decay_lambda"],
        "residual_quantiles": residual_quantiles,
        "selection_objective": "rolling_mae",
        "folds": fold_descriptors,
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


class TorchMLPClassifier:
    def __init__(self, input_hidden: tuple[int, ...] = (64, 32), epochs: int = 30, learning_rate: float = 0.0025) -> None:
        self.input_hidden = input_hidden
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.network = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"input_hidden": self.input_hidden, "epochs": self.epochs, "learning_rate": self.learning_rate}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        if torch is None or nn is None:
            raise ImportError("torch is required for TorchMLPClassifier")
        matrix = torch.tensor(np.asarray(X), dtype=torch.float32)
        target = torch.tensor(np.asarray(y), dtype=torch.float32).view(-1, 1)
        layers: list[nn.Module] = []
        input_dim = matrix.shape[1]
        for hidden in self.input_hidden:
            layers.extend([nn.Linear(input_dim, hidden), nn.ReLU()])
            input_dim = hidden
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()
        self.network.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = self.network(matrix)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
        return self

    def predict_proba(self, X):
        assert self.network is not None
        matrix = torch.tensor(np.asarray(X), dtype=torch.float32)
        self.network.eval()
        with torch.no_grad():
            logits = self.network(matrix)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class TorchMLPRegressor:
    def __init__(self, input_hidden: tuple[int, ...] = (64, 32), epochs: int = 30, learning_rate: float = 0.0025) -> None:
        self.input_hidden = input_hidden
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.network = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"input_hidden": self.input_hidden, "epochs": self.epochs, "learning_rate": self.learning_rate}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        if torch is None or nn is None:
            raise ImportError("torch is required for TorchMLPRegressor")
        matrix = torch.tensor(np.asarray(X), dtype=torch.float32)
        target = torch.tensor(np.asarray(y), dtype=torch.float32).view(-1, 1)
        layers: list[nn.Module] = []
        input_dim = matrix.shape[1]
        for hidden in self.input_hidden:
            layers.extend([nn.Linear(input_dim, hidden), nn.ReLU()])
            input_dim = hidden
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        self.network.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            predictions = self.network(matrix)
            loss = loss_fn(predictions, target)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X):
        assert self.network is not None
        matrix = torch.tensor(np.asarray(X), dtype=torch.float32)
        self.network.eval()
        with torch.no_grad():
            predictions = self.network(matrix).cpu().numpy().reshape(-1)
        return predictions


def _classifier_candidates() -> list[dict]:
    candidates = [
        {"name": "logistic_regression", "estimator": LogisticRegression(max_iter=1500, class_weight="balanced"), "supports_sample_weight": True},
        {
            "name": "gradient_boosting_tuned",
            "estimator": GradientBoostingClassifier(
                learning_rate=0.05,
                n_estimators=250,
                max_depth=3,
                random_state=42,
            ),
            "supports_sample_weight": True,
        },
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
    if torch is not None:
        candidates.append(
            {
                "name": "torch_mlp",
                "estimator": TorchMLPClassifier(input_hidden=(64, 32), epochs=30, learning_rate=0.0025),
                "supports_sample_weight": False,
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
    if torch is not None:
        candidates.append(
            {
                "name": "torch_mlp",
                "estimator": TorchMLPRegressor(input_hidden=(64, 32), epochs=30, learning_rate=0.0025),
                "supports_sample_weight": False,
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


def _trim_classifier_candidates(candidates: list[dict], sample_count: int) -> list[dict]:
    if _is_fast_model_search():
        return [candidate for candidate in candidates if candidate["name"] == "logistic_regression"]
    if sample_count >= 180:
        return candidates
    keep_prefixes = ("logistic_regression", "hist_gradient_boosting")
    return [candidate for candidate in candidates if candidate["name"].startswith(keep_prefixes)]


def _trim_regressor_candidates(candidates: list[dict], sample_count: int) -> list[dict]:
    if _is_fast_model_search():
        return [candidate for candidate in candidates if candidate["name"] == "hist_gradient_boosting"]
    if sample_count >= 1200:
        return candidates
    keep_prefixes = ("hist_gradient_boosting",)
    return [candidate for candidate in candidates if candidate["name"].startswith(keep_prefixes)]


def _trim_decay_values(decay_values: tuple[float, ...], sample_count: int) -> tuple[float, ...]:
    if _is_fast_model_search():
        return decay_values[:1]
    if sample_count >= 1200:
        return decay_values
    return decay_values[:2]


def _is_fast_model_search() -> bool:
    return os.getenv("FAST_MODEL_SEARCH", "").strip().lower() in {"1", "true", "yes", "on"}


def _safe_log_loss(y_true, probabilities) -> float:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return float(log_loss(y_true, clipped, labels=[0, 1]))


def _safe_rmse(y_true, predictions) -> float:
    return float(np.sqrt(mean_squared_error(y_true, predictions)))


def _optuna_tune_classifier(
    task_name: str,
    frame: pd.DataFrame,
    labels: list[int],
    dates: list[date],
    family_name: str,
    categorical_features: list[str],
    numeric_features: list[str],
) -> dict | None:
    if optuna is None:
        return None

    def objective(trial):
        candidate = _classifier_candidate_from_trial(family_name, trial)
        if candidate is None:
            return float("inf")
        result = _rolling_classifier_score(frame, labels, dates, candidate, categorical_features, numeric_features, trial.suggest_float("decay_lambda", 0.005, 0.04))
        return result["log_loss"]

    try:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=MATCH_OPTUNA_TRIALS, show_progress_bar=False)
    except Exception:
        return None
    candidate = _classifier_candidate_from_trial(family_name, study.best_trial)
    if candidate is None:
        return None
    candidate["name"] = f"{task_name}_{family_name}_optuna"
    return candidate


def _optuna_tune_regressor(
    task_name: str,
    frame: pd.DataFrame,
    targets: list[float],
    dates: list[date],
    family_name: str,
    categorical_features: list[str],
    numeric_features: list[str],
) -> dict | None:
    if optuna is None:
        return None

    def objective(trial):
        candidate = _regressor_candidate_from_trial(family_name, trial)
        if candidate is None:
            return float("inf")
        result = _rolling_regressor_score(frame, targets, dates, candidate, categorical_features, numeric_features, trial.suggest_float("decay_lambda", 0.005, 0.04))
        return result["mae"]

    try:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=REGRESSOR_OPTUNA_TRIALS, show_progress_bar=False)
    except Exception:
        return None
    candidate = _regressor_candidate_from_trial(family_name, study.best_trial)
    if candidate is None:
        return None
    candidate["name"] = f"{task_name}_{family_name}_optuna"
    return candidate


def _classifier_candidate_from_trial(family_name: str, trial) -> dict | None:
    if family_name == "hist_gradient_boosting":
        return {
            "name": family_name,
            "estimator": HistGradientBoostingClassifier(
                learning_rate=trial.suggest_float("learning_rate", 0.02, 0.08),
                max_depth=trial.suggest_int("max_depth", 4, 10),
                max_iter=trial.suggest_int("max_iter", 180, 420),
                random_state=42,
            ),
            "supports_sample_weight": True,
        }
    if family_name == "lightgbm" and LGBMClassifier is not None:
        return {
            "name": family_name,
            "estimator": LGBMClassifier(
                n_estimators=trial.suggest_int("n_estimators", 220, 520),
                learning_rate=trial.suggest_float("learning_rate", 0.02, 0.08),
                num_leaves=trial.suggest_int("num_leaves", 24, 72),
                subsample=trial.suggest_float("subsample", 0.75, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.75, 1.0),
                random_state=42,
                verbose=-1,
            ),
            "supports_sample_weight": True,
        }
    if family_name == "catboost" and CatBoostClassifier is not None:
        return {
            "name": family_name,
            "estimator": CatBoostClassifier(
                iterations=trial.suggest_int("iterations", 220, 520),
                depth=trial.suggest_int("depth", 5, 9),
                learning_rate=trial.suggest_float("learning_rate", 0.02, 0.08),
                loss_function="Logloss",
                random_seed=42,
                verbose=False,
            ),
            "supports_sample_weight": True,
        }
    return None


def _regressor_candidate_from_trial(family_name: str, trial) -> dict | None:
    if family_name == "hist_gradient_boosting":
        return {
            "name": family_name,
            "estimator": HistGradientBoostingRegressor(
                learning_rate=trial.suggest_float("learning_rate", 0.02, 0.08),
                max_depth=trial.suggest_int("max_depth", 4, 10),
                max_iter=trial.suggest_int("max_iter", 180, 420),
                random_state=42,
            ),
            "supports_sample_weight": True,
        }
    if family_name == "lightgbm" and LGBMRegressor is not None:
        return {
            "name": family_name,
            "estimator": LGBMRegressor(
                n_estimators=trial.suggest_int("n_estimators", 220, 520),
                learning_rate=trial.suggest_float("learning_rate", 0.02, 0.08),
                num_leaves=trial.suggest_int("num_leaves", 24, 72),
                subsample=trial.suggest_float("subsample", 0.75, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.75, 1.0),
                random_state=42,
                verbose=-1,
            ),
            "supports_sample_weight": True,
        }
    if family_name == "catboost" and CatBoostRegressor is not None:
        return {
            "name": family_name,
            "estimator": CatBoostRegressor(
                iterations=trial.suggest_int("iterations", 220, 520),
                depth=trial.suggest_int("depth", 5, 9),
                learning_rate=trial.suggest_float("learning_rate", 0.02, 0.08),
                loss_function="RMSE",
                random_seed=42,
                verbose=False,
            ),
            "supports_sample_weight": True,
        }
    return None


def _refine_classifier_candidates(
    task_name: str,
    frame: pd.DataFrame,
    labels: list[int],
    dates: list[date],
    candidates: list[dict],
    categorical_features: list[str],
    numeric_features: list[str],
) -> list[dict]:
    if optuna is None or len(frame) < 240:
        return []
    family_names = [candidate["name"] for candidate in candidates if candidate["name"] in {"hist_gradient_boosting", "lightgbm", "catboost"}]
    refined: list[dict] = []
    for family_name in family_names[:3]:
        tuned = _optuna_tune_classifier(task_name, frame, labels, dates, family_name, categorical_features, numeric_features)
        if tuned is not None:
            refined.append(tuned)
    return refined


def _refine_regressor_candidates(
    task_name: str,
    frame: pd.DataFrame,
    targets: list[float],
    dates: list[date],
    candidates: list[dict],
    categorical_features: list[str],
    numeric_features: list[str],
) -> list[dict]:
    if optuna is None or len(frame) < 2400:
        return []
    family_names = [candidate["name"] for candidate in candidates if candidate["name"] in {"hist_gradient_boosting", "lightgbm", "catboost"}]
    refined: list[dict] = []
    for family_name in family_names[:3]:
        tuned = _optuna_tune_regressor(task_name, frame, targets, dates, family_name, categorical_features, numeric_features)
        if tuned is not None:
            refined.append(tuned)
    return refined


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


def _rolling_fold_descriptors(dates: list[date], *, windows: int, min_train: int, min_validation: int) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    effective_windows = 2 if _is_fast_model_search() else min(windows, 3 if len(dates) < 200 else windows)
    for index, (val_start, val_end) in enumerate(_rolling_splits(len(dates), windows=effective_windows, min_train=min_train, min_validation=min_validation), start=1):
        descriptors.append(
            FoldSummary(
                index=index,
                train_start=dates[0].isoformat(),
                train_end=dates[val_start - 1].isoformat(),
                validation_start=dates[val_start].isoformat(),
                validation_end=dates[val_end - 1].isoformat(),
                train_size=val_start,
                validation_size=val_end - val_start,
            ).__dict__
        )
    return descriptors


def _rolling_classifier_score(
    frame: pd.DataFrame,
    labels: list[int],
    dates: list[date],
    candidate: dict,
    categorical_features: list[str],
    numeric_features: list[str],
    decay_lambda: float,
 ) -> dict[str, Any]:
    accuracy_scores: list[float] = []
    log_losses: list[float] = []
    briers: list[float] = []
    effective_windows = 2 if _is_fast_model_search() else (3 if len(frame) < 200 else ROLLING_WINDOWS)
    min_validation = max(4, len(frame) // 12)
    folds = _rolling_fold_descriptors(dates, windows=effective_windows, min_train=MATCH_MIN_TRAIN, min_validation=min_validation)
    for fold, (val_start, val_end) in zip(
        folds,
        _rolling_splits(len(frame), windows=effective_windows, min_train=MATCH_MIN_TRAIN, min_validation=min_validation),
        strict=False,
    ):
        pipeline = _classifier_pipeline(categorical_features, numeric_features, candidate["estimator"])
        _fit_pipeline(
            pipeline,
            frame.iloc[:val_start],
            labels[:val_start],
            _sample_weights(dates[:val_start], dates[val_start - 1], decay_lambda),
            step_name="classifier",
            supports_sample_weight=candidate["supports_sample_weight"],
        )
        probabilities = pipeline.predict_proba(frame.iloc[val_start:val_end])[:, 1]
        fold["accuracy"] = float(accuracy_score(labels[val_start:val_end], (probabilities >= 0.5).astype(int)))
        fold["log_loss"] = _safe_log_loss(labels[val_start:val_end], probabilities)
        fold["brier"] = float(brier_score_loss(labels[val_start:val_end], probabilities))
        accuracy_scores.append(fold["accuracy"])
        log_losses.append(fold["log_loss"])
        briers.append(fold["brier"])
    return {
        "accuracy": float(np.mean(accuracy_scores)) if accuracy_scores else 0.0,
        "log_loss": float(np.mean(log_losses)) if log_losses else float("inf"),
        "brier": float(np.mean(briers)) if briers else float("inf"),
        "accuracy_std": float(np.std(accuracy_scores)) if accuracy_scores else 0.0,
        "log_loss_std": float(np.std(log_losses)) if log_losses else 0.0,
        "brier_std": float(np.std(briers)) if briers else 0.0,
        "folds": folds,
    }


def _rolling_regressor_score(
    frame: pd.DataFrame,
    targets: list[float],
    dates: list[date],
    candidate: dict,
    categorical_features: list[str],
    numeric_features: list[str],
    decay_lambda: float,
 ) -> dict[str, Any]:
    maes: list[float] = []
    rmses: list[float] = []
    effective_windows = 2 if _is_fast_model_search() else (3 if len(frame) < 1500 else ROLLING_WINDOWS)
    min_validation = max(12, len(frame) // 14)
    folds = _rolling_fold_descriptors(dates, windows=effective_windows, min_train=PLAYER_MIN_TRAIN, min_validation=min_validation)
    for fold, (val_start, val_end) in zip(
        folds,
        _rolling_splits(len(frame), windows=effective_windows, min_train=PLAYER_MIN_TRAIN, min_validation=min_validation),
        strict=False,
    ):
        pipeline = _regressor_pipeline(categorical_features, numeric_features, candidate["estimator"])
        _fit_pipeline(
            pipeline,
            frame.iloc[:val_start],
            targets[:val_start],
            _sample_weights(dates[:val_start], dates[val_start - 1], decay_lambda),
            step_name="regressor",
            supports_sample_weight=candidate["supports_sample_weight"],
        )
        predictions = pipeline.predict(frame.iloc[val_start:val_end])
        fold["mae"] = float(mean_absolute_error(targets[val_start:val_end], predictions))
        fold["rmse"] = _safe_rmse(targets[val_start:val_end], predictions)
        maes.append(fold["mae"])
        rmses.append(fold["rmse"])
    return {
        "mae": float(np.mean(maes)) if maes else float("inf"),
        "rmse": float(np.mean(rmses)) if rmses else float("inf"),
        "mae_std": float(np.std(maes)) if maes else 0.0,
        "rmse_std": float(np.std(rmses)) if rmses else 0.0,
        "folds": folds,
    }


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


def _apply_match_result(
    state: TeamState,
    won: bool,
    maps_won: int,
    maps_lost: int,
    match_date: date,
    map_margin: int,
    opponent_elo: float,
) -> None:
    state.matches += 1
    if won:
        state.wins += 1
    state.maps_won += maps_won
    state.maps_lost += maps_lost
    state.opponent_elo_sum += opponent_elo
    state.recent_results.append(1 if won else 0)
    state.recent_map_margin.append(map_margin)
    state.last_played = match_date


def _win_rate(state: TeamState) -> float:
    return (state.wins + WIN_PRIOR_ALPHA) / (state.matches + WIN_PRIOR_ALPHA + WIN_PRIOR_BETA)


def _map_win_rate(state: TeamState) -> float:
    total = state.maps_won + state.maps_lost
    return (state.maps_won + MAP_PRIOR_ALPHA) / (total + MAP_PRIOR_ALPHA + MAP_PRIOR_BETA)


def _experience_index(state: TeamState) -> float:
    return math.log1p(state.matches)


def _strength_of_schedule(state: TeamState) -> float:
    if state.matches <= 0:
        return 0.0
    return (state.opponent_elo_sum / state.matches - INITIAL_ELO) / 400.0


def _adjusted_win_rate(state: TeamState) -> float:
    return _win_rate(state) + (0.18 * _strength_of_schedule(state))


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
    wins = state.map_wins.get(map_name, 0)
    return (wins + MAP_PRIOR_ALPHA) / (matches + MAP_PRIOR_ALPHA + MAP_PRIOR_BETA)


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


def _lineup_continuity(team_name: str, context: dict) -> float:
    recent = context["team_recent_players"].get(team_name, {})
    if not recent:
        return 0.0
    top_values = sorted(recent.values(), reverse=True)[:5]
    if not top_values:
        return 0.0
    return sum(value * value for value in top_values) / max(sum(top_values) ** 2, 1.0)


def _normalize_event_stage(stage: str | None) -> str:
    normalized = (stage or "unknown").strip().lower()
    if "grand" in normalized or "final" in normalized:
        return "final"
    if "lower" in normalized:
        return "lower_bracket"
    if "upper" in normalized:
        return "upper_bracket"
    if "playoff" in normalized or "quarter" in normalized or "semi" in normalized:
        return "playoffs"
    if "group" in normalized or "week" in normalized:
        return "group_stage"
    return "other"


def _agent_role(agent_name: str | None) -> str:
    if not agent_name:
        return "unknown"
    role_map = {
        "Jett": "duelist",
        "Raze": "duelist",
        "Phoenix": "duelist",
        "Yoru": "duelist",
        "Sova": "initiator",
        "Fade": "initiator",
        "Skye": "initiator",
        "Breach": "initiator",
        "Omen": "controller",
        "Viper": "controller",
        "Brimstone": "controller",
        "Astra": "controller",
        "Killjoy": "sentinel",
        "Cypher": "sentinel",
        "Sage": "sentinel",
        "Chamber": "sentinel",
    }
    return role_map.get(agent_name, "flex")


def _event_elo(team_name: str, event_name: str, context: dict) -> float:
    event_history = context.get("team_event_history", {}).get(team_name, {}).get(event_name, {})
    matches = float(event_history.get("matches", 0.0))
    wins = float(event_history.get("wins", 0.0))
    event_win_rate = (wins + WIN_PRIOR_ALPHA) / (matches + WIN_PRIOR_ALPHA + WIN_PRIOR_BETA)
    base_elo = context.get("team_states", {}).get(team_name, TeamState()).elo
    return base_elo + ((event_win_rate - 0.5) * 120.0)


def _event_map_strength(team_name: str, map_name: str, event_name: str, context: dict) -> float:
    event_map_key = f"{event_name}:{map_name}"
    history = context.get("team_event_map_history", {}).get(team_name, {}).get(event_map_key, {})
    matches = float(history.get("matches", 0.0))
    wins = float(history.get("wins", 0.0))
    if matches <= 0:
        return _team_map_win_rate(team_name, map_name, context)
    return (wins + MAP_PRIOR_ALPHA) / (matches + MAP_PRIOR_ALPHA + MAP_PRIOR_BETA)


def _dataset_snapshot(
    matches: list[VLRMatchRecord],
    maps: list[VLRMapRecord],
    player_stats: list[VLRPlayerStatLine],
) -> dict[str, Any]:
    per_region: dict[str, int] = defaultdict(int)
    per_event: dict[str, int] = defaultdict(int)
    for match in matches:
        per_region[match.region] += 1
        per_event[match.event_name] += 1
    return {
        "matches": len(matches),
        "maps": len(maps),
        "player_rows": len(player_stats),
        "date_range": {
            "start": matches[0].match_date.isoformat() if matches else None,
            "end": matches[-1].match_date.isoformat() if matches else None,
        },
        "per_region": dict(sorted(per_region.items())),
        "per_event": dict(sorted(per_event.items())),
    }


def _rank_maps_for_team(team_name: str, context: dict) -> list[str]:
    state = context["team_states"].get(team_name, TeamState())
    def _map_score(map_name: str) -> tuple[float, int]:
        matches = state.map_matches.get(map_name, 0)
        reliability = min(1.0, matches / MAP_RELIABILITY_TARGET)
        blended = (_team_map_win_rate(team_name, map_name, context) - 0.5) * reliability + 0.5
        return blended, matches

    return sorted(COMMON_MAPS, key=_map_score, reverse=True)


def _combined_map_rank(team_a: str, team_b: str, context: dict) -> list[str]:
    state_a = context["team_states"].get(team_a, TeamState())
    state_b = context["team_states"].get(team_b, TeamState())

    def _combined_score(map_name: str) -> tuple[float, int]:
        a_matches = state_a.map_matches.get(map_name, 0)
        b_matches = state_b.map_matches.get(map_name, 0)
        reliability = min(1.0, (a_matches + b_matches) / (2 * MAP_RELIABILITY_TARGET))
        a_wr = _team_map_win_rate(team_a, map_name, context)
        b_wr = _team_map_win_rate(team_b, map_name, context)
        parity = 1.0 - abs(a_wr - b_wr)
        strength = (a_wr + b_wr) / 2.0
        return ((0.65 * strength + 0.35 * parity) * reliability, a_matches + b_matches)

    return sorted(
        COMMON_MAPS,
        key=_combined_score,
        reverse=True,
    )


def _likely_lineup(team_name: str, context: dict) -> list[str]:
    recent = context["team_recent_players"].get(team_name, {})
    if not recent:
        return [f"{team_name} Player {index}" for index in range(1, 6)]

    player_states = context.get("player_states", {})
    lineup: list[str] = []
    for player_name, _ in sorted(recent.items(), key=lambda item: item[1], reverse=True):
        player_state = player_states.get(player_name)
        if player_state is not None and player_state.team_name != team_name:
            continue
        lineup.append(player_name)
        if len(lineup) >= 5:
            break

    return lineup if lineup else [player for player, _ in sorted(recent.items(), key=lambda item: item[1], reverse=True)[:5]]


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
        opponent_elo_sum=state.opponent_elo_sum,
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
        "team_event_history": {
            team: {event: dict(values) for event, values in events.items()}
            for team, events in context.get("team_event_history", {}).items()
        },
        "team_event_map_history": {
            team: {key: dict(values) for key, values in maps.items()}
            for team, maps in context.get("team_event_map_history", {}).items()
        },
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
        "team_event_history": {
            team: {event: dict(values) for event, values in events.items()}
            for team, events in context.get("team_event_history", {}).items()
        },
        "team_event_map_history": {
            team: {key: dict(values) for key, values in maps.items()}
            for team, maps in context.get("team_event_map_history", {}).items()
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
        "opponent_elo_sum": state.opponent_elo_sum,
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
        opponent_elo_sum=float(payload.get("opponent_elo_sum", 0.0)),
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
