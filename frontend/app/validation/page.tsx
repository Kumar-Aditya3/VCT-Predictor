import { getValidationReport } from "@/lib/api";

export default async function ValidationPage() {
  const report = await getValidationReport();

  return (
    <main className="grid-ambient min-h-screen">
      <div className="mx-auto w-full max-w-7xl px-4 py-8 md:px-8 md:py-12">
        <section className="rounded-xl2 border border-line bg-surface/70 p-6 shadow-glow backdrop-blur">
          <p className="text-xs uppercase tracking-[0.22em] text-muted">Validation</p>
          <h1 className="mt-2 text-3xl font-bold md:text-5xl">VLR Ground-Truth Alignment</h1>
          <p className="mt-3 max-w-3xl text-sm text-muted md:text-base">
            Validation reflects the latest trained Tier 1 bundle across match winners, map outcomes, and player kill/death projections.
          </p>
        </section>

        <section className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <MetricCard label="Status" value={report.status} />
          <MetricCard label="Compared Matches" value={String(report.compared_matches)} />
          <MetricCard label="Winner Accuracy" value={`${Math.round(report.winner_accuracy * 100)}%`} />
          <MetricCard label="Rolling Winner Accuracy" value={`${Math.round(report.rolling_winner_accuracy * 100)}%`} />
          <MetricCard label="Compared Maps" value={String(report.compared_maps)} />
          <MetricCard label="Map Accuracy" value={`${Math.round(report.map_accuracy * 100)}%`} />
          <MetricCard label="Rolling Map Accuracy" value={`${Math.round(report.rolling_map_accuracy * 100)}%`} />
          <MetricCard label="Player Rows" value={String(report.player_rows)} />
          <MetricCard label="Player K/D MAE" value={report.player_kd_mae.toFixed(2)} />
          <MetricCard label="Rolling Player K/D MAE" value={report.rolling_player_kd_mae.toFixed(2)} />
          <MetricCard label="Integrity Issues" value={String(report.integrity_issue_count)} />
          <MetricCard label="Rolling Windows" value={String(report.rolling_windows_evaluated)} />
        </section>

        <section className="mt-6 rounded-xl border border-line bg-surface2/75 p-5">
          <p className="text-xs uppercase tracking-[0.18em] text-muted">Report Metadata</p>
          <dl className="mt-4 grid gap-4 md:grid-cols-2">
            <div>
              <dt className="text-sm text-muted">Generated</dt>
              <dd className="mt-1 text-lg font-semibold text-text">{new Date(report.generated_at).toLocaleString()}</dd>
            </div>
            <div>
              <dt className="text-sm text-muted">Source</dt>
              <dd className="mt-1 text-lg font-semibold text-text">{report.source}</dd>
            </div>
            <div>
              <dt className="text-sm text-muted">Match Calibration</dt>
              <dd className="mt-1 text-lg font-semibold text-text">{report.calibration_summary.match || "none"}</dd>
            </div>
            <div>
              <dt className="text-sm text-muted">Map Calibration</dt>
              <dd className="mt-1 text-lg font-semibold text-text">{report.calibration_summary.map || "none"}</dd>
            </div>
          </dl>
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
