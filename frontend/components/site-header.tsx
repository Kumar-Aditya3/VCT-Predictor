import Link from "next/link";

const links = [
  { href: "/", label: "Predictions" },
  { href: "/analytics", label: "Analytics" },
  { href: "/validation", label: "Validation" },
];

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-40 border-b border-line/80 bg-[rgba(9,11,15,0.82)] backdrop-blur">
      <div className="mx-auto flex w-full max-w-7xl items-center justify-between px-4 py-4 md:px-8">
        <div>
          <p className="text-xs uppercase tracking-[0.22em] text-muted">VCT Tier 1 Predictor</p>
          <p className="mt-1 text-sm text-text/80">Live VLR-backed bootstrap analytics dashboard</p>
        </div>
        <nav className="flex gap-2">
          {links.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="rounded-full border border-line bg-surface2/70 px-4 py-2 text-sm font-semibold text-text transition hover:border-accent hover:text-accent"
            >
              {link.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
