import { useOverview } from "@/lib/api";
import { fmtPrice, fmtSignedPct } from "@/lib/format";
import { useSelection } from "@/lib/selection";
import { COMMODITY_META, sortCommodities } from "@/lib/theme";

/** Scrolling tape of the selected market's session close and change per commodity. */
export function Ticker() {
  const { run, round, market } = useSelection();
  const { data } = useOverview(run, round, market);
  const snap = data?.snapshot ?? [];
  if (!snap.length) return <div className="h-8 border-b border-ink-700 bg-ink-900/60" />;
  const order = sortCommodities(snap.map((s) => s.commodity));
  const items = order.map((c) => snap.find((s) => s.commodity === c)!);
  const row = (key: string) => (
    <div key={key} className="flex shrink-0 items-center gap-8 pr-8" aria-hidden={key === "b"}>
      {items.map((s) => {
        const up = s.pct >= 0;
        return (
          <div key={`${key}-${s.commodity}`} className="flex items-center gap-2 font-mono text-[11.5px]">
            <span className="h-1.5 w-1.5 rounded-full" style={{ background: COMMODITY_META[s.commodity]?.color }} />
            <span className="font-semibold text-fg-2">{COMMODITY_META[s.commodity]?.ticker ?? s.commodity}</span>
            <span className="text-fg">{fmtPrice(s.last)}</span>
            <span className={up ? "text-good" : "text-critical"}>
              {up ? "▲" : "▼"} {fmtSignedPct(s.pct)}
            </span>
            {s.events > 0 && <span className="rounded border border-warn/30 px-1 text-[10px] text-warn">{s.events} episodes</span>}
          </div>
        );
      })}
    </div>
  );
  return (
    <div className="group relative flex h-8 items-center overflow-hidden border-b border-ink-700 bg-ink-900/60">
      <div className="flex animate-marquee group-hover:[animation-play-state:paused]">
        {row("a")}
        {row("b")}
      </div>
      <div className="pointer-events-none absolute inset-y-0 left-0 w-16 bg-gradient-to-r from-ink-950 to-transparent" />
      <div className="pointer-events-none absolute inset-y-0 right-0 w-16 bg-gradient-to-l from-ink-950 to-transparent" />
    </div>
  );
}
