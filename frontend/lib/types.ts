export type MapPrediction = {
  map_name: string;
  team_a_win_probability: number;
  projected_score: string;
  picked_by?: string | null;
};

export type PlayerProjection = {
  player_name: string;
  team_name: string;
  map_name?: string | null;
  agent_name?: string | null;
  projected_kills: number;
  projected_deaths: number;
};

export type MatchPrediction = {
  match_id: string;
  team_a: string;
  team_b: string;
  region?: string | null;
  event_name?: string | null;
  event_stage?: string | null;
  match_date?: string | null;
  team_a_match_win_probability: number;
  confidence_score?: number | null;
  prediction_generated_at?: string | null;
  sample_size?: number | null;
  map_predictions: MapPrediction[];
  player_projections: PlayerProjection[];
  model_version: string;
};

export type MatchFixtureInput = {
  match_id: string;
  region: string;
  event_name: string;
  event_stage?: string | null;
  team_a: string;
  team_b: string;
  match_date: string;
  best_of: number;
};

export type PredictionSnapshot = {
  generated_at: string;
  source: string;
  prediction_mode: string;
  model_version: string;
  predictions: MatchPrediction[];
};

export type ModelPerformance = {
  last_run_at?: string | null;
  prediction_mode: string;
  model_version: string;
  upcoming_prediction_count: number;
  compared_matches: number;
  winner_accuracy: number;
  map_accuracy: number;
  player_kd_mae: number;
  tier1_results_count: number;
  training_samples: number;
  map_training_samples: number;
  player_training_samples: number;
  cv_accuracy: number;
  map_cv_accuracy: number;
  player_cv_score: number;
  rolling_winner_accuracy: number;
  rolling_map_accuracy: number;
  rolling_player_kd_mae: number;
  match_estimator?: string | null;
  map_estimator?: string | null;
  player_kills_estimator?: string | null;
  player_deaths_estimator?: string | null;
  match_calibration?: string | null;
  map_calibration?: string | null;
  excluded_match_rows: number;
  excluded_map_rows: number;
  excluded_player_rows: number;
  candidate_leaderboards: Record<string, Array<Record<string, unknown>>>;
  source: string;
};

export type ValidationReport = {
  generated_at: string;
  source: string;
  compared_matches: number;
  winner_accuracy: number;
  compared_maps: number;
  map_accuracy: number;
  player_rows: number;
  player_kd_mae: number;
  rolling_windows_evaluated: number;
  integrity_issue_count: number;
  calibration_summary: Record<string, string>;
  rolling_winner_accuracy: number;
  rolling_map_accuracy: number;
  rolling_player_kd_mae: number;
  status: string;
};
