import { CustomMatchupForm } from "@/components/custom-matchup-form";
import { Hero } from "@/components/hero";
import { PredictionCard } from "@/components/prediction-card";
import { getUpcomingPredictions } from "@/lib/api";

export default async function HomePage() {
  const snapshot = await getUpcomingPredictions();
  const sourceLabel =
    snapshot.source === "frontend_fallback" || snapshot.source === "bootstrap_fallback"
      ? "Fallback Bootstrap Data"
      : "API Snapshot";

  return (
    <main className="grid-ambient min-h-screen">
      <div className="mx-auto w-full max-w-7xl px-4 py-8 md:px-8 md:py-12">
        <Hero
          sourceLabel={sourceLabel}
          predictionMode={snapshot.prediction_mode}
          matchCount={snapshot.predictions.length}
        />

        <section className="mt-4 rounded-xl border border-line bg-surface/60 px-4 py-3 text-sm text-muted shadow-glow">
          <strong className="text-text">Generated:</strong> {new Date(snapshot.generated_at).toLocaleString()}
          {"  "}
          <strong className="ml-4 text-text">Model:</strong> {snapshot.model_version}
        </section>

        <CustomMatchupForm />

        <section className="mt-6 grid gap-4 md:mt-8 md:gap-5">
          {snapshot.predictions.map((prediction, index) => (
            <PredictionCard key={prediction.match_id} prediction={prediction} index={index} />
          ))}
        </section>
      </div>
    </main>
  );
}
