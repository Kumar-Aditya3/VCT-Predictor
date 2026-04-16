from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import pandas as pd
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error
from sklearn.neural_network import MLPClassifier, MLPRegressor

from app.core.config import get_settings
from app.services import modeling as mdl
from app.services.storage import SQLiteStore


def _decay_grid(preset: str) -> tuple[float, ...]:
    if preset == "full":
        return (0.005, 0.01, 0.015, 0.02, 0.03, 0.04)
    return (0.01, 0.02, 0.03, 0.04)


def _classifier_candidates(preset: str) -> list[dict]:
    candidates: list[dict] = []

    # Linear baseline family
    for c in (0.3, 1.0, 3.0):
        candidates.append(
            {
                "name": f"logreg_c{c}",
                "estimator": LogisticRegression(max_iter=2000, class_weight="balanced", C=c),
                "supports_sample_weight": True,
            }
        )

    # Tree ensembles
    rf_configs = [
        (400, 10, 2),
        (800, None, 1),
    ]
    if preset == "full":
        rf_configs.append((1000, None, 2))

    for n_estimators, max_depth, min_samples_leaf in rf_configs:
        candidates.append(
            {
                "name": f"rf_n{n_estimators}_d{max_depth}_l{min_samples_leaf}",
                "estimator": RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
                "supports_sample_weight": True,
            }
        )
        candidates.append(
            {
                "name": f"xt_n{n_estimators}_d{max_depth}_l{min_samples_leaf}",
                "estimator": ExtraTreesClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                    n_jobs=-1,
                ),
                "supports_sample_weight": True,
            }
        )

    # Boosting families
    for lr, depth, iters in ((0.05, 6, 250), (0.03, 8, 450)):
        candidates.append(
            {
                "name": f"hgb_lr{lr}_d{depth}_it{iters}",
                "estimator": HistGradientBoostingClassifier(
                    learning_rate=lr,
                    max_depth=depth,
                    max_iter=iters,
                    random_state=42,
                ),
                "supports_sample_weight": True,
            }
        )
    for lr, n_estimators, depth in ((0.05, 250, 3), (0.03, 450, 5)):
        candidates.append(
            {
                "name": f"gb_lr{lr}_n{n_estimators}_d{depth}",
                "estimator": GradientBoostingClassifier(
                    learning_rate=lr,
                    n_estimators=n_estimators,
                    max_depth=depth,
                    random_state=42,
                ),
                "supports_sample_weight": True,
            }
        )

    # Neural baseline
    for layers in ((64, 32), (128, 64)):
        candidates.append(
            {
                "name": f"mlp_{layers[0]}_{layers[1]}",
                "estimator": MLPClassifier(
                    hidden_layer_sizes=layers,
                    activation="relu",
                    alpha=1e-4,
                    max_iter=500,
                    early_stopping=True,
                    random_state=42,
                ),
                "supports_sample_weight": False,
            }
        )

    # Optional external gradient boosters
    if mdl.LGBMClassifier is not None:
        for n_estimators, lr, leaves in ((300, 0.05, 31), (600, 0.03, 63)):
            candidates.append(
                {
                    "name": f"lgbm_n{n_estimators}_lr{lr}_l{leaves}",
                    "estimator": mdl.LGBMClassifier(
                        n_estimators=n_estimators,
                        learning_rate=lr,
                        num_leaves=leaves,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=42,
                        verbose=-1,
                    ),
                    "supports_sample_weight": True,
                }
            )

    if mdl.XGBClassifier is not None:
        for n_estimators, lr, depth in ((300, 0.05, 6), (600, 0.03, 8)):
            candidates.append(
                {
                    "name": f"xgb_n{n_estimators}_lr{lr}_d{depth}",
                    "estimator": mdl.XGBClassifier(
                        n_estimators=n_estimators,
                        learning_rate=lr,
                        max_depth=depth,
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

    if mdl.CatBoostClassifier is not None:
        for iterations, depth, lr in ((300, 6, 0.05), (600, 8, 0.03)):
            candidates.append(
                {
                    "name": f"cat_n{iterations}_d{depth}_lr{lr}",
                    "estimator": mdl.CatBoostClassifier(
                        iterations=iterations,
                        depth=depth,
                        learning_rate=lr,
                        loss_function="Logloss",
                        random_seed=42,
                        verbose=False,
                    ),
                    "supports_sample_weight": True,
                }
            )

    return candidates


def _regressor_candidates(preset: str) -> list[dict]:
    candidates: list[dict] = []

    rf_configs = [(400, 10, 2), (800, None, 1)]
    if preset == "full":
        rf_configs.append((1000, None, 2))

    for n_estimators, max_depth, min_samples_leaf in rf_configs:
        candidates.append(
            {
                "name": f"rf_n{n_estimators}_d{max_depth}_l{min_samples_leaf}",
                "estimator": RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                    n_jobs=-1,
                ),
                "supports_sample_weight": True,
            }
        )
        candidates.append(
            {
                "name": f"xt_n{n_estimators}_d{max_depth}_l{min_samples_leaf}",
                "estimator": ExtraTreesRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                    n_jobs=-1,
                ),
                "supports_sample_weight": True,
            }
        )

    for lr, depth, iters in ((0.05, 6, 250), (0.03, 8, 450)):
        candidates.append(
            {
                "name": f"hgb_lr{lr}_d{depth}_it{iters}",
                "estimator": HistGradientBoostingRegressor(
                    learning_rate=lr,
                    max_depth=depth,
                    max_iter=iters,
                    random_state=42,
                ),
                "supports_sample_weight": True,
            }
        )

    for lr, n_estimators, depth in ((0.05, 250, 3), (0.03, 450, 5)):
        candidates.append(
            {
                "name": f"gb_lr{lr}_n{n_estimators}_d{depth}",
                "estimator": GradientBoostingRegressor(
                    learning_rate=lr,
                    n_estimators=n_estimators,
                    max_depth=depth,
                    random_state=42,
                ),
                "supports_sample_weight": True,
            }
        )

    for layers in ((64, 32), (128, 64)):
        candidates.append(
            {
                "name": f"mlp_{layers[0]}_{layers[1]}",
                "estimator": MLPRegressor(
                    hidden_layer_sizes=layers,
                    activation="relu",
                    alpha=1e-4,
                    max_iter=500,
                    early_stopping=True,
                    random_state=42,
                ),
                "supports_sample_weight": False,
            }
        )

    if mdl.LGBMRegressor is not None:
        for n_estimators, lr, leaves in ((300, 0.05, 31), (600, 0.03, 63)):
            candidates.append(
                {
                    "name": f"lgbm_n{n_estimators}_lr{lr}_l{leaves}",
                    "estimator": mdl.LGBMRegressor(
                        n_estimators=n_estimators,
                        learning_rate=lr,
                        num_leaves=leaves,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=42,
                        verbose=-1,
                    ),
                    "supports_sample_weight": True,
                }
            )

    if mdl.XGBRegressor is not None:
        for n_estimators, lr, depth in ((300, 0.05, 6), (600, 0.03, 8)):
            candidates.append(
                {
                    "name": f"xgb_n{n_estimators}_lr{lr}_d{depth}",
                    "estimator": mdl.XGBRegressor(
                        n_estimators=n_estimators,
                        learning_rate=lr,
                        max_depth=depth,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        random_state=42,
                        n_jobs=4,
                    ),
                    "supports_sample_weight": True,
                }
            )

    if mdl.CatBoostRegressor is not None:
        for iterations, depth, lr in ((300, 6, 0.05), (600, 8, 0.03)):
            candidates.append(
                {
                    "name": f"cat_n{iterations}_d{depth}_lr{lr}",
                    "estimator": mdl.CatBoostRegressor(
                        iterations=iterations,
                        depth=depth,
                        learning_rate=lr,
                        loss_function="RMSE",
                        random_seed=42,
                        verbose=False,
                    ),
                    "supports_sample_weight": True,
                }
            )

    return candidates


def _evaluate_classifier_task(task_name: str, rows: list[dict], labels: list[int], dates, candidates: list[dict], decay_values: tuple[float, ...]) -> list[dict]:
    frame = pd.DataFrame(rows)
    split_index = max(10, int(len(frame) * 0.8))
    if split_index >= len(frame):
        split_index = len(frame) - 1

    train_X = frame.iloc[:split_index]
    val_X = frame.iloc[split_index:]
    train_y = labels[:split_index]
    val_y = labels[split_index:]

    categorical_features = [col for col in frame.columns if frame[col].dtype == object]
    numeric_features = [col for col in frame.columns if col not in categorical_features]

    results: list[dict] = []
    for candidate in candidates:
        for decay_lambda in decay_values:
            try:
                rolling_accuracy = mdl._rolling_classifier_score(
                    frame,
                    labels,
                    dates,
                    candidate,
                    categorical_features,
                    numeric_features,
                    decay_lambda,
                )
                pipeline = mdl._classifier_pipeline(categorical_features, numeric_features, candidate["estimator"])
                mdl._fit_pipeline(
                    pipeline,
                    train_X,
                    train_y,
                    mdl._sample_weights(dates[:split_index], dates[-1], decay_lambda),
                    step_name="classifier",
                    supports_sample_weight=candidate["supports_sample_weight"],
                )
                probabilities = pipeline.predict_proba(val_X)[:, 1]
                holdout_accuracy = accuracy_score(val_y, (probabilities >= 0.5).astype(int))
                results.append(
                    {
                        "task": task_name,
                        "model": candidate["name"],
                        "decay_lambda": float(decay_lambda),
                        "rolling_accuracy": float(rolling_accuracy),
                        "holdout_accuracy": float(holdout_accuracy),
                        "brier": float(brier_score_loss(val_y, probabilities)),
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "task": task_name,
                        "model": candidate["name"],
                        "decay_lambda": float(decay_lambda),
                        "error": str(exc),
                    }
                )
    return results


def _evaluate_regressor_task(task_name: str, rows: list[dict], targets: list[float], dates, candidates: list[dict], decay_values: tuple[float, ...]) -> list[dict]:
    frame = pd.DataFrame(rows)
    split_index = max(20, int(len(frame) * 0.8))
    if split_index >= len(frame):
        split_index = len(frame) - 1

    train_X = frame.iloc[:split_index]
    val_X = frame.iloc[split_index:]
    train_y = targets[:split_index]
    val_y = targets[split_index:]

    categorical_features = [col for col in frame.columns if frame[col].dtype == object]
    numeric_features = [col for col in frame.columns if col not in categorical_features]

    results: list[dict] = []
    for candidate in candidates:
        for decay_lambda in decay_values:
            try:
                rolling_mae = mdl._rolling_regressor_score(
                    frame,
                    targets,
                    dates,
                    candidate,
                    categorical_features,
                    numeric_features,
                    decay_lambda,
                )
                pipeline = mdl._regressor_pipeline(categorical_features, numeric_features, candidate["estimator"])
                mdl._fit_pipeline(
                    pipeline,
                    train_X,
                    train_y,
                    mdl._sample_weights(dates[:split_index], dates[-1], decay_lambda),
                    step_name="regressor",
                    supports_sample_weight=candidate["supports_sample_weight"],
                )
                holdout_mae = mean_absolute_error(val_y, pipeline.predict(val_X))
                results.append(
                    {
                        "task": task_name,
                        "model": candidate["name"],
                        "decay_lambda": float(decay_lambda),
                        "rolling_mae": float(rolling_mae),
                        "holdout_mae": float(holdout_mae),
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "task": task_name,
                        "model": candidate["name"],
                        "decay_lambda": float(decay_lambda),
                        "error": str(exc),
                    }
                )
    return results


def _top_rows(rows: list[dict], metric: str, ascending: bool, limit: int = 12) -> list[dict]:
    valid = [row for row in rows if metric in row]
    return sorted(valid, key=lambda item: item[metric], reverse=not ascending)[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run broad model benchmark sweeps for match/map/player tasks.")
    parser.add_argument("--preset", choices=("broad", "full"), default="broad")
    args = parser.parse_args()

    started = perf_counter()
    store = SQLiteStore()
    matches = store.load_matches()
    maps = store.load_maps()
    player_stats = store.load_player_stats()
    ordered_matches = sorted(matches, key=lambda item: (item.match_date, item.match_id))

    maps_by_match: dict[str, list] = {}
    for item in maps:
        maps_by_match.setdefault(item.match_id, []).append(item)
    for map_list in maps_by_match.values():
        map_list.sort(key=lambda item: item.order_index)

    stats_by_map: dict[str, list] = {}
    for stat in player_stats:
        stats_by_map.setdefault(stat.map_id, []).append(stat)

    feature_store = mdl._build_feature_store(ordered_matches, maps_by_match, stats_by_map)

    decay_values = _decay_grid(args.preset)
    classifier_candidates = _classifier_candidates(args.preset)
    regressor_candidates = _regressor_candidates(args.preset)

    match_results = _evaluate_classifier_task(
        "match_winner",
        feature_store["match_rows"],
        feature_store["match_labels"],
        feature_store["match_dates"],
        classifier_candidates,
        decay_values,
    )
    map_results = _evaluate_classifier_task(
        "map_winner",
        feature_store["map_rows"],
        feature_store["map_labels"],
        feature_store["map_dates"],
        classifier_candidates,
        decay_values,
    )
    player_kills_results = _evaluate_regressor_task(
        "player_kills",
        feature_store["player_rows"],
        feature_store["player_kills"],
        feature_store["player_dates"],
        regressor_candidates,
        decay_values,
    )
    player_deaths_results = _evaluate_regressor_task(
        "player_deaths",
        feature_store["player_rows"],
        feature_store["player_deaths"],
        feature_store["player_dates"],
        regressor_candidates,
        decay_values,
    )

    all_rows = match_results + map_results + player_kills_results + player_deaths_results
    completed_at = datetime.now(timezone.utc).isoformat()

    report = {
        "completed_at": completed_at,
        "preset": args.preset,
        "counts": {
            "matches": len(matches),
            "maps": len(maps),
            "player_rows": len(player_stats),
            "classifier_candidates": len(classifier_candidates),
            "regressor_candidates": len(regressor_candidates),
            "decay_values": list(decay_values),
        },
        "top_rankings": {
            "match_winner": _top_rows(match_results, "rolling_accuracy", ascending=False),
            "map_winner": _top_rows(map_results, "rolling_accuracy", ascending=False),
            "player_kills": _top_rows(player_kills_results, "rolling_mae", ascending=True),
            "player_deaths": _top_rows(player_deaths_results, "rolling_mae", ascending=True),
        },
        "results": {
            "match_winner": match_results,
            "map_winner": map_results,
            "player_kills": player_kills_results,
            "player_deaths": player_deaths_results,
        },
        "duration_seconds": round(perf_counter() - started, 2),
    }

    settings = get_settings()
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = settings.artifacts_dir / f"model_benchmark_{args.preset}_{stamp}.json"
    csv_path = settings.artifacts_dir / f"model_benchmark_{args.preset}_{stamp}.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    pd.DataFrame(all_rows).to_csv(csv_path, index=False)

    print(f"benchmark_json={json_path}")
    print(f"benchmark_csv={csv_path}")
    print(f"duration_seconds={report['duration_seconds']}")
    for task_name, rows in report["top_rankings"].items():
        if not rows:
            continue
        top = rows[0]
        if "rolling_accuracy" in top:
            print(
                f"top_{task_name} model={top['model']} decay={top['decay_lambda']} "
                f"rolling_accuracy={top['rolling_accuracy']:.4f} holdout_accuracy={top['holdout_accuracy']:.4f}"
            )
        else:
            print(
                f"top_{task_name} model={top['model']} decay={top['decay_lambda']} "
                f"rolling_mae={top['rolling_mae']:.4f} holdout_mae={top['holdout_mae']:.4f}"
            )


if __name__ == "__main__":
    main()
