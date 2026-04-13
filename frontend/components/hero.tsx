"use client";

import { motion } from "framer-motion";

import { StatPill } from "./stat-pill";

type HeroProps = {
  sourceLabel: string;
  predictionMode: string;
  matchCount: number;
};

export function Hero({ sourceLabel, predictionMode, matchCount }: HeroProps) {
  return (
    <motion.header
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="rounded-xl2 border border-line bg-surface/70 p-6 shadow-glow backdrop-blur"
    >
      <p className="text-xs uppercase tracking-[0.22em] text-muted">VCT Tier 1 Intelligence</p>
      <h1 className="mt-2 text-3xl font-bold md:text-5xl">Weekly Match And Player Outcome Predictor</h1>
      <p className="mt-3 max-w-3xl text-sm text-muted md:text-base">
        Scope locked to Kickoff, Split 1, Split 2, Masters, and Champions across Americas, EMEA, Pacific, and China.
      </p>
      <div className="mt-5 flex flex-wrap gap-2">
        <StatPill label="Mode" value={predictionMode} />
        <StatPill label="Refresh" value="Weekly" />
        <StatPill label="Source" value={sourceLabel} />
        <StatPill label="Matches" value={String(matchCount)} />
      </div>
    </motion.header>
  );
}
