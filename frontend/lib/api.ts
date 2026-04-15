import { fallbackPredictionSnapshot } from "./mock-data";
import { MatchFixtureInput, MatchPrediction, ModelPerformance, PredictionSnapshot, ValidationReport } from "./types";

const API_ROOT = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export async function getUpcomingPredictions(): Promise<PredictionSnapshot> {
  try {
    const response = await fetch(`${API_ROOT}/api/v1/predictions/upcoming`, {
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(`upcoming predictions request failed: ${response.status}`);
    }
    return (await response.json()) as PredictionSnapshot;
  } catch {
    return fallbackPredictionSnapshot;
  }
}

export async function getModelPerformance(): Promise<ModelPerformance> {
  try {
    const response = await fetch(`${API_ROOT}/api/v1/model/performance`, {
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(`model performance request failed: ${response.status}`);
    }
    return (await response.json()) as ModelPerformance;
  } catch {
    return {
      last_run_at: fallbackPredictionSnapshot.generated_at,
      prediction_mode: fallbackPredictionSnapshot.prediction_mode,
      model_version: fallbackPredictionSnapshot.model_version,
      upcoming_prediction_count: fallbackPredictionSnapshot.predictions.length,
      compared_matches: 0,
      winner_accuracy: 0,
      map_accuracy: 0,
      player_kd_mae: 0,
      tier1_results_count: 0,
      training_samples: 0,
      map_training_samples: 0,
      player_training_samples: 0,
      cv_accuracy: 0,
      map_cv_accuracy: 0,
      player_cv_score: 0,
      rolling_winner_accuracy: 0,
      rolling_map_accuracy: 0,
      rolling_player_kd_mae: 0,
      match_estimator: "fallback",
      map_estimator: "fallback",
      player_kills_estimator: "fallback",
      player_deaths_estimator: "fallback",
      match_calibration: "none",
      map_calibration: "none",
      excluded_match_rows: 0,
      excluded_map_rows: 0,
      excluded_player_rows: 0,
      candidate_leaderboards: {},
      source: fallbackPredictionSnapshot.source,
    };
  }
}

export async function getValidationReport(): Promise<ValidationReport> {
  try {
    const response = await fetch(`${API_ROOT}/api/v1/data/validation`, {
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(`validation request failed: ${response.status}`);
    }
    return (await response.json()) as ValidationReport;
  } catch {
    return {
      generated_at: fallbackPredictionSnapshot.generated_at,
      source: fallbackPredictionSnapshot.source,
      compared_matches: 0,
      winner_accuracy: 0,
      compared_maps: 0,
      map_accuracy: 0,
      player_rows: 0,
      player_kd_mae: 0,
      rolling_windows_evaluated: 0,
      integrity_issue_count: 0,
      calibration_summary: {},
      rolling_winner_accuracy: 0,
      rolling_map_accuracy: 0,
      rolling_player_kd_mae: 0,
      status: "fallback",
    };
  }
}

export async function predictCustomFixtures(fixtures: MatchFixtureInput[]): Promise<MatchPrediction[]> {
  let response: Response;
  try {
    response = await fetch(`${API_ROOT}/api/v1/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ fixtures }),
    });
  } catch {
    throw new Error("The local predictor API is offline. Start the backend server on http://localhost:8000.");
  }

  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail || `predict request failed: ${response.status}`);
  }

  const payload = (await response.json()) as { predictions: MatchPrediction[] };
  return payload.predictions;
}
