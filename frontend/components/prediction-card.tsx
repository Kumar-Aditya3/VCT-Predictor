"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";

import type { MatchPrediction } from "@/lib/types";

import { StatPill } from "./stat-pill";

type PredictionCardProps = {
  prediction: MatchPrediction;
  index: number;
};

export function PredictionCard({ prediction, index }: PredictionCardProps) {
  const [expanded, setExpanded] = useState(false);
  const teamBProbability = 1 - prediction.team_a_match_win_probability;
  const teamATone = prediction.team_a_match_win_probability > teamBProbability ? "good" : "neutral";
  const teamBTone = teamBProbability > prediction.team_a_match_win_probability ? "good" : "neutral";
  const contextLabel = [prediction.region, prediction.event_name, prediction.event_stage].filter(Boolean).join(" | ");
  const confidenceLabel =
    prediction.confidence_score == null ? null : `${Math.round(prediction.confidence_score * 100)}% confidence`;
  const freshnessLabel = prediction.prediction_generated_at
    ? `Updated ${new Date(prediction.prediction_generated_at).toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      })}`
    : null;
  const supportLabel = prediction.sample_size == null ? null : `${prediction.sample_size} training samples`;

  const playerStats = useMemo(() => {
    const hasPerMapRows = prediction.player_projections.some((player) => player.map_name);
    const grouped = new Map<
      string,
      {
        player_name: string;
        team_name: string;
        agent_name?: string | null;
        total_kills: number;
        total_deaths: number;
        map_rows: Array<{ map_name: string; projected_kills: number; projected_deaths: number }>;
      }
    >();

    for (const player of prediction.player_projections) {
      const key = `${player.team_name}::${player.player_name}`;
      if (!grouped.has(key)) {
        grouped.set(key, {
          player_name: player.player_name,
          team_name: player.team_name,
          agent_name: player.agent_name,
          total_kills: 0,
          total_deaths: 0,
          map_rows: [],
        });
      }

      const current = grouped.get(key)!;
      if (!current.agent_name && player.agent_name) {
        current.agent_name = player.agent_name;
      }

      if (hasPerMapRows && player.map_name) {
        current.map_rows.push({
          map_name: player.map_name,
          projected_kills: player.projected_kills,
          projected_deaths: player.projected_deaths,
        });
        current.total_kills += player.projected_kills;
        current.total_deaths += player.projected_deaths;
      }

      if (!hasPerMapRows) {
        current.total_kills += player.projected_kills;
        current.total_deaths += player.projected_deaths;
      }
    }

    const players = Array.from(grouped.values()).sort((a, b) => b.total_kills - a.total_kills);
    return {
      hasPerMapRows,
      teamA: players.filter((player) => player.team_name === prediction.team_a),
      teamB: players.filter((player) => player.team_name === prediction.team_b),
      totalPlayers: players.length,
    };
  }, [prediction.player_projections, prediction.team_a, prediction.team_b]);

  const previewPlayers = [...playerStats.teamA.slice(0, 3), ...playerStats.teamB.slice(0, 3)];

  return (
    <motion.section
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: index * 0.08 }}
      className="rounded-xl2 border border-line bg-surface/80 p-5 shadow-glow backdrop-blur"
      role="button"
      tabIndex={0}
      onClick={() => setExpanded((current) => !current)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          setExpanded((current) => !current);
        }
      }}
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.18em] text-muted">{prediction.match_id}</p>
          <h3 className="mt-1 text-xl font-bold">
            {prediction.team_a} <span className="text-muted">vs</span> {prediction.team_b}
          </h3>
          {contextLabel ? <p className="mt-1 text-sm text-muted">{contextLabel}</p> : null}
          <p className="mt-2 text-xs uppercase tracking-[0.14em] text-accent/90">{expanded ? "Click to collapse full stats" : "Click match to expand full stats"}</p>
        </div>
        <div className="flex items-center gap-2">
          <StatPill
            label={prediction.team_a}
            value={`${Math.round(prediction.team_a_match_win_probability * 100)}%`}
            tone={teamATone}
          />
          <StatPill
            label={prediction.team_b}
            value={`${Math.round(teamBProbability * 100)}%`}
            tone={teamBTone}
          />
        </div>
      </div>

      {confidenceLabel || freshnessLabel || supportLabel ? (
        <div className="mt-3 flex flex-wrap gap-2 text-xs text-muted">
          {confidenceLabel ? <span className="rounded-full border border-line bg-black/20 px-3 py-1">{confidenceLabel}</span> : null}
          {supportLabel ? <span className="rounded-full border border-line bg-black/20 px-3 py-1">{supportLabel}</span> : null}
          {freshnessLabel ? <span className="rounded-full border border-line bg-black/20 px-3 py-1">{freshnessLabel}</span> : null}
        </div>
      ) : null}

      <div className="mt-4 grid gap-3 md:grid-cols-3">
        {prediction.map_predictions.map((map) => (
          <div key={map.map_name} className="rounded-xl border border-line bg-surface2/70 p-3">
            <p className="text-sm font-semibold">{map.map_name}</p>
            <p className="mt-1 text-xs text-muted">
              Projected score: {map.projected_score}
              {map.picked_by ? ` • Picked by ${map.picked_by}` : " • Decider"}
            </p>
            <div className="mt-2 h-2 overflow-hidden rounded-full bg-black/30">
              <div
                className="h-full rounded-full bg-gradient-to-r from-accent to-accent2"
                style={{ width: `${Math.round(map.team_a_win_probability * 100)}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4 rounded-xl border border-line bg-surface2/40 p-3">
        <p className="text-xs uppercase tracking-[0.14em] text-muted">Player Projections (K / D)</p>
        {playerStats.totalPlayers > 0 ? (
          <div className="mt-2 grid gap-2 md:grid-cols-2">
            {previewPlayers.map((player) => (
              <div key={`${player.team_name}-${player.player_name}`} className="flex items-center justify-between rounded-lg bg-black/20 px-3 py-2">
                <div>
                  <p className="text-sm font-semibold">{player.player_name}</p>
                  <p className="text-xs text-muted">
                    {player.team_name}
                    {player.agent_name ? ` • ${player.agent_name}` : ""}
                  </p>
                </div>
                <p className="font-semibold">
                  {player.total_kills.toFixed(1)} / {player.total_deaths.toFixed(1)}
                </p>
              </div>
            ))}
          </div>
        ) : (
          <p className="mt-2 text-sm text-muted">No player projections available for this fixture.</p>
        )}
      </div>

      {expanded && playerStats.totalPlayers > 0 ? (
        <div className="mt-4 grid gap-3 lg:grid-cols-2">
          {[prediction.team_a, prediction.team_b].map((teamName) => {
            const teamPlayers = teamName === prediction.team_a ? playerStats.teamA : playerStats.teamB;
            return (
              <div key={teamName} className="rounded-xl border border-line bg-surface2/60 p-3">
                <p className="text-sm font-semibold">{teamName}</p>
                <div className="mt-2 space-y-2">
                  {teamPlayers.map((player) => {
                    const kd = player.total_deaths > 0 ? player.total_kills / player.total_deaths : player.total_kills;
                    return (
                      <div key={`${teamName}-${player.player_name}-detail`} className="rounded-lg bg-black/20 px-3 py-2">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <p className="text-sm font-semibold">{player.player_name}</p>
                          <p className="text-xs text-muted">K/D {kd.toFixed(2)}</p>
                        </div>
                        <p className="mt-1 text-sm">
                          <span className="font-semibold">Series:</span> {player.total_kills.toFixed(1)} / {player.total_deaths.toFixed(1)}
                        </p>
                        {playerStats.hasPerMapRows && player.map_rows.length > 0 ? (
                          <div className="mt-2 space-y-1 text-xs text-muted">
                            {player.map_rows.map((mapRow) => {
                              const mapKd = mapRow.projected_deaths > 0 ? mapRow.projected_kills / mapRow.projected_deaths : mapRow.projected_kills;
                              return (
                                <p key={`${teamName}-${player.player_name}-${mapRow.map_name}`}>
                                  {mapRow.map_name}: {mapRow.projected_kills.toFixed(1)} / {mapRow.projected_deaths.toFixed(1)} (K/D {mapKd.toFixed(2)})
                                </p>
                              );
                            })}
                          </div>
                        ) : null}
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      ) : null}
    </motion.section>
  );
}
