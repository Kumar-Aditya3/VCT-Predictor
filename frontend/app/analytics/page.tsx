import { getModelPerformance } from "@/lib/api";

export default async function AnalyticsPage() {
  const performance = await getModelPerformance();

  return (
    <main className="grid-ambient min-h-screen">
      <div className="mx-auto w-full max-w-7xl px-4 py-8 md:px-8 md:py-12">
        <section className="rounded-xl2 border border-line bg-surface/70 p-6 shadow-glow backdrop-blur">
          <p className="text-xs uppercase tracking-[0.22em] text-muted">Analytics</p>
          <h1 className="mt-2 text-3xl font-bold md:text-5xl">Operational Model Snapshot</h1>
          <p className="mt-3 max-w-3xl text-sm text-muted md:text-base">
            This page reflects the latest pipeline artifact or a frontend fallback if the API is unavailable.
          </p>
        </section>

        <section className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <MetricCard label="Prediction Mode" value={performance.prediction_mode} />
          <MetricCard label="Model Version" value={performance.model_version} />
          <MetricCard label="Upcoming Matches" value={String(performance.upcoming_prediction_count)} />
          <MetricCard label="Tier 1 Results Seen" value={String(performance.tier1_results_count)} />
        </section>

        <section className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          <MetricCard label="Compared Matches" value={String(performance.compared_matches)} />
          <MetricCard label="Winner Accuracy" value={`${Math.round(performance.winner_accuracy * 100)}%`} />
          <MetricCard label="Rolling Winner Accuracy" value={`${Math.round(performance.rolling_winner_accuracy * 100)}%`} />
          <MetricCard label="Map Accuracy" value={`${Math.round(performance.map_accuracy * 100)}%`} />
          <MetricCard label="Rolling Map Accuracy" value={`${Math.round(performance.rolling_map_accuracy * 100)}%`} />
          <MetricCard label="Player K/D MAE" value={performance.player_kd_mae.toFixed(2)} />
          <MetricCard label="Rolling Player K/D MAE" value={performance.rolling_player_kd_mae.toFixed(2)} />
          <MetricCard label="Training Samples" value={String(performance.training_samples)} />
          <MetricCard label="Map Samples" value={String(performance.map_training_samples)} />
          <MetricCard label="Player Samples" value={String(performance.player_training_samples)} />
          <MetricCard label="Holdout Accuracy" value={`${Math.round(performance.cv_accuracy * 100)}%`} />
          <MetricCard label="Map Holdout Accuracy" value={`${Math.round(performance.map_cv_accuracy * 100)}%`} />
          <MetricCard label="Player CV Score" value={`${Math.round(performance.player_cv_score * 100)}%`} />
          <MetricCard label="Artifact Source" value={performance.source} />
          <MetricCard label="Last Run" value={performance.last_run_at ? new Date(performance.last_run_at).toLocaleString() : "Not run yet"} />
        </section>

        <section className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <MetricCard label="Match Model" value={performance.match_estimator || "n/a"} />
          <MetricCard label="Map Model" value={performance.map_estimator || "n/a"} />
          <MetricCard label="Player Kills Model" value={performance.player_kills_estimator || "n/a"} />
          <MetricCard label="Player Deaths Model" value={performance.player_deaths_estimator || "n/a"} />
          <MetricCard label="Match Calibration" value={performance.match_calibration || "none"} />
          <MetricCard label="Map Calibration" value={performance.map_calibration || "none"} />
          <MetricCard label="Excluded Maps" value={String(performance.excluded_map_rows)} />
          <MetricCard label="Excluded Player Rows" value={String(performance.excluded_player_rows)} />
        </section>

        <section className="mt-6 grid gap-4 xl:grid-cols-2">
          <LeaderboardCard title="Match Candidates" rows={performance.candidate_leaderboards.match || []} />
          <LeaderboardCard title="Map Candidates" rows={performance.candidate_leaderboards.map || []} />
          <LeaderboardCard title="Player Kills Candidates" rows={performance.candidate_leaderboards.player_kills || []} />
          <LeaderboardCard title="Player Deaths Candidates" rows={performance.candidate_leaderboards.player_deaths || []} />
        </section>
      </div>
    </main>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-line bg-surface2/75 p-5">
      <p className="text-xs uppercase tracking-[0.18em] text-muted">{label}</p>
      <p className="mt-3 text-xl font-bold text-text">{value}</p>
    </div>
  );
}

function LeaderboardCard({ title, rows }: { title: string; rows: Array<Record<string, unknown>> }) {
  return (
    <div className="rounded-xl border border-line bg-surface2/75 p-5">
      <p className="text-xs uppercase tracking-[0.18em] text-muted">{title}</p>
      <div className="mt-3 space-y-2">
        {rows.length ? (
          rows.slice(0, 5).map((row, index) => (
            <div key={`${title}-${index}`} className="rounded-lg bg-black/20 px-3 py-2 text-sm text-text">
              <div className="font-semibold">{String(row.name || "candidate")}</div>
              <div className="text-xs text-muted">
                {Object.entries(row)
                  .filter(([key]) => key !== "name")
                  .map(([key, value]) => `${key}: ${String(value)}`)
                  .join(" • ")}
              </div>
            </div>
          ))
        ) : (
          <p className="text-sm text-muted">No candidate search metadata yet.</p>
        )}
      </div>
    </div>
  );
}
