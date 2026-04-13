import { PredictionSnapshot } from "./types";

export const fallbackPredictionSnapshot: PredictionSnapshot = {
  generated_at: new Date("2026-04-08T16:36:28Z").toISOString(),
  source: "frontend_fallback",
  prediction_mode: "bootstrap",
  model_version: "bootstrap-deterministic-v1",
  predictions: [
    {
      match_id: "pac-masters-001",
      team_a: "Paper Rex",
      team_b: "Gen.G",
      team_a_match_win_probability: 0.57,
      confidence_score: 0.78,
      prediction_generated_at: "2026-04-08T16:36:28Z",
      sample_size: 64,
      map_predictions: [
        { map_name: "Ascent", team_a_win_probability: 0.6, projected_score: "13-10", picked_by: "Paper Rex" },
        { map_name: "Bind", team_a_win_probability: 0.49, projected_score: "11-13", picked_by: "Gen.G" },
        { map_name: "Lotus", team_a_win_probability: 0.58, projected_score: "13-11", picked_by: null },
      ],
      player_projections: [
        { player_name: "something", team_name: "Paper Rex", agent_name: "Jett", projected_kills: 47.3, projected_deaths: 42.4 },
        { player_name: "Meteor", team_name: "Gen.G", agent_name: "Raze", projected_kills: 45.7, projected_deaths: 43.6 },
      ],
      model_version: "bootstrap-deterministic-v1",
    },
    {
      match_id: "emea-s2-014",
      team_a: "FNATIC",
      team_b: "Team Heretics",
      team_a_match_win_probability: 0.54,
      confidence_score: 0.71,
      prediction_generated_at: "2026-04-08T16:36:28Z",
      sample_size: 64,
      map_predictions: [
        { map_name: "Ascent", team_a_win_probability: 0.55, projected_score: "13-10", picked_by: "FNATIC" },
        { map_name: "Bind", team_a_win_probability: 0.5, projected_score: "13-11", picked_by: "Team Heretics" },
        { map_name: "Lotus", team_a_win_probability: 0.53, projected_score: "13-10", picked_by: null },
      ],
      player_projections: [
        { player_name: "Chronicle", team_name: "FNATIC", agent_name: "Sova", projected_kills: 46.9, projected_deaths: 42.7 },
        { player_name: "Wo0t", team_name: "Team Heretics", agent_name: "Raze", projected_kills: 46.1, projected_deaths: 43.3 },
      ],
      model_version: "bootstrap-deterministic-v1",
    },
  ],
};
