"use client";

import { useEffect, useId, useRef, useState, useTransition } from "react";

import { predictCustomFixtures } from "@/lib/api";
import type { MatchFixtureInput, MatchPrediction } from "@/lib/types";

import { PredictionCard } from "./prediction-card";

const regions = ["Americas", "EMEA", "Pacific", "China", "International"] as const;
const events = ["Kickoff", "Split 1", "Split 2", "Masters", "Champions"] as const;
const bestOfOptions = [
  { value: "3", label: "Best of 3" },
  { value: "5", label: "Best of 5" },
] as const;

function todayString() {
  return new Date().toISOString().slice(0, 10);
}

export function CustomMatchupForm() {
  const [result, setResult] = useState<MatchPrediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();
  const [form, setForm] = useState<MatchFixtureInput>({
    match_id: "custom-ui-001",
    region: "International",
    event_name: "Masters",
    event_stage: "Playoffs",
    team_a: "Paper Rex",
    team_b: "Gen.G",
    match_date: todayString(),
    best_of: 3,
  });

  function update<K extends keyof MatchFixtureInput>(key: K, value: MatchFixtureInput[K]) {
    setForm((current) => {
      if (key === "event_name") {
        const nextEvent = value as MatchFixtureInput["event_name"];
        const forceInternational = nextEvent === "Masters" || nextEvent === "Champions";
        return {
          ...current,
          event_name: nextEvent,
          region: forceInternational ? "International" : current.region,
        };
      }

      return { ...current, [key]: value };
    });
  }

  function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);

    if (!form.team_a.trim() || !form.team_b.trim()) {
      setError("Enter both team names.");
      return;
    }
    if (form.team_a.trim().toLowerCase() === form.team_b.trim().toLowerCase()) {
      setError("Choose two different teams.");
      return;
    }

    startTransition(async () => {
      try {
        const predictions = await predictCustomFixtures([
          {
            ...form,
            match_id: `custom-${Date.now()}`,
            team_a: form.team_a.trim(),
            team_b: form.team_b.trim(),
            event_stage: form.event_stage?.trim() || undefined,
          },
        ]);
        setResult(predictions[0] ?? null);
      } catch (submissionError) {
        setResult(null);
        setError(submissionError instanceof Error ? submissionError.message : "Prediction failed.");
      }
    });
  }

  return (
    <section className="mt-6 rounded-xl2 border border-line bg-surface/80 p-5 shadow-glow backdrop-blur">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.18em] text-muted">Custom Matchup</p>
          <h2 className="mt-1 text-2xl font-bold">Test Any Tier 1 Fixture</h2>
          <p className="mt-2 max-w-2xl text-sm text-muted">
            Submit a custom Tier 1 matchup and the dashboard will call the live `/api/v1/predict` endpoint.
          </p>
          <p className="mt-2 max-w-2xl text-xs text-muted/80">
            Use <span className="text-text">International</span> for cross-region Masters and Champions matchups.
          </p>
          {form.event_name === "Masters" || form.event_name === "Champions" ? (
            <p className="mt-2 max-w-2xl text-xs text-accent/90">
              Region is locked to International for this event.
            </p>
          ) : null}
        </div>
      </div>

      <form onSubmit={onSubmit} className="mt-5 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Field label="Region">
          <SelectField
            value={form.region}
            onChange={(value) => update("region", value)}
            options={regions.map((region) => ({ value: region, label: region }))}
            disabled={form.event_name === "Masters" || form.event_name === "Champions"}
          />
        </Field>

        <Field label="Event">
          <SelectField
            value={form.event_name}
            onChange={(value) => update("event_name", value)}
            options={events.map((eventName) => ({ value: eventName, label: eventName }))}
          />
        </Field>

        <Field label="Stage">
          <input
            value={form.event_stage || ""}
            onChange={(event) => update("event_stage", event.target.value)}
            className={textInputClassName}
            placeholder="Playoffs"
          />
        </Field>

        <Field label="Best Of">
          <SelectField
            value={String(form.best_of)}
            onChange={(value) => update("best_of", Number(value))}
            options={bestOfOptions}
          />
        </Field>

        <Field label="Team A">
          <input
            value={form.team_a}
            onChange={(event) => update("team_a", event.target.value)}
            className={textInputClassName}
            placeholder="Paper Rex"
          />
        </Field>

        <Field label="Team B">
          <input
            value={form.team_b}
            onChange={(event) => update("team_b", event.target.value)}
            className={textInputClassName}
            placeholder="Gen.G"
          />
        </Field>

        <Field label="Match Date">
          <input
            type="date"
            value={form.match_date}
            onChange={(event) => update("match_date", event.target.value)}
            className={dateInputClassName}
          />
        </Field>

        <div className="flex items-end">
          <button
            type="submit"
            disabled={isPending}
            className="w-full rounded-xl border border-accent/60 bg-gradient-to-r from-accent/20 via-accent/10 to-accent2/20 px-4 py-3 text-sm font-semibold text-text shadow-[0_10px_30px_rgba(60,224,183,0.12)] transition hover:border-accent hover:from-accent/30 hover:to-accent2/25 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isPending ? "Running Prediction..." : "Predict Matchup"}
          </button>
        </div>
      </form>

      {error ? (
        <div className="mt-4 rounded-xl border border-[var(--bad)]/50 bg-[var(--bad)]/10 px-4 py-3 text-sm text-text">
          {error}
        </div>
      ) : null}

      {result ? (
        <div className="mt-6">
          <PredictionCard prediction={result} index={0} />
        </div>
      ) : null}
    </section>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-2 block text-xs uppercase tracking-[0.18em] text-muted">{label}</span>
      {children}
    </label>
  );
}

function SelectField({
  value,
  onChange,
  options,
  disabled = false,
}: {
  value: string;
  onChange: (value: string) => void;
  options: ReadonlyArray<{ value: string; label: string }>;
  disabled?: boolean;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const listboxId = useId();
  const selectedOption = options.find((option) => option.value === value) ?? options[0];

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    function handlePointerDown(event: MouseEvent) {
      if (!containerRef.current?.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setIsOpen(false);
      }
    }

    window.addEventListener("mousedown", handlePointerDown);
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("mousedown", handlePointerDown);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen]);

  return (
    <div ref={containerRef} className="relative">
      <button
        type="button"
        className={`${selectTriggerClassName} ${disabled ? "cursor-not-allowed border-accent/30 text-muted/70" : ""}`}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-controls={listboxId}
        onClick={() => {
          if (!disabled) {
            setIsOpen((current) => !current);
          }
        }}
        disabled={disabled}
      >
        <span>{selectedOption?.label ?? value}</span>
        <ChevronIcon open={isOpen} />
      </button>

      {isOpen && !disabled ? (
        <div
          id={listboxId}
          role="listbox"
          className="absolute z-30 mt-2 w-full overflow-hidden rounded-xl border border-accent/30 bg-[linear-gradient(180deg,rgba(19,23,34,0.98),rgba(14,18,27,0.98))] p-1 shadow-[0_20px_60px_rgba(3,8,20,0.55)] backdrop-blur"
        >
          {options.map((option) => {
            const selected = option.value === value;
            return (
              <button
                key={option.value}
                type="button"
                role="option"
                aria-selected={selected}
                onClick={() => {
                  onChange(option.value);
                  setIsOpen(false);
                }}
                className={`flex w-full items-center justify-between rounded-lg px-3 py-2.5 text-left text-sm transition ${
                  selected
                    ? "bg-accent/18 text-text shadow-[inset_0_0_0_1px_rgba(60,224,183,0.22)]"
                    : "text-muted hover:bg-white/5 hover:text-text"
                }`}
              >
                <span>{option.label}</span>
                {selected ? <span className="text-xs uppercase tracking-[0.18em] text-accent">Selected</span> : null}
              </button>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg
      viewBox="0 0 20 20"
      aria-hidden="true"
      className={`h-4 w-4 shrink-0 text-muted transition ${open ? "rotate-180 text-accent" : ""}`}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M5.5 7.5 10 12l4.5-4.5" />
    </svg>
  );
}

const textInputClassName =
  "w-full rounded-xl border border-line bg-[linear-gradient(180deg,rgba(26,32,48,0.92),rgba(19,23,34,0.96))] px-4 py-3 text-sm text-text outline-none transition placeholder:text-muted focus:border-accent focus:shadow-[0_0_0_1px_rgba(60,224,183,0.25)]";

const selectTriggerClassName =
  "flex w-full items-center justify-between gap-3 rounded-xl border border-line bg-[linear-gradient(180deg,rgba(26,32,48,0.92),rgba(19,23,34,0.96))] px-4 py-3 text-sm text-text outline-none transition hover:border-accent/50 focus-visible:border-accent focus-visible:shadow-[0_0_0_1px_rgba(60,224,183,0.25)]";

const dateInputClassName =
  "w-full rounded-xl border border-line bg-[linear-gradient(180deg,rgba(26,32,48,0.92),rgba(19,23,34,0.96))] px-4 py-3 text-sm text-text outline-none transition [color-scheme:dark] focus:border-accent focus:shadow-[0_0_0_1px_rgba(60,224,183,0.25)]";
