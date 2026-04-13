type StatPillProps = {
  label: string;
  value: string;
  tone?: "neutral" | "good" | "bad";
};

export function StatPill({ label, value, tone = "neutral" }: StatPillProps) {
  const toneClasses =
    tone === "good"
      ? "border-good/30 bg-good/10 text-good"
      : tone === "bad"
      ? "border-bad/30 bg-bad/10 text-bad"
      : "border-line bg-surface2 text-text";

  return (
    <div className={`rounded-full border px-3 py-1 text-xs font-semibold ${toneClasses}`}>
      <span className="mr-2 text-muted">{label}</span>
      <span>{value}</span>
    </div>
  );
}
