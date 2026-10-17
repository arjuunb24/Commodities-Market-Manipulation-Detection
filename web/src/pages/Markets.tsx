import clsx from "clsx";
import { motion } from "framer-motion";
import { Globe2 } from "lucide-react";
import { useSearchParams } from "react-router-dom";
import { ErrorNote, PageHeader, Skeleton } from "@/components/ui";
import { useOverview } from "@/lib/api";
import { useSelection } from "@/lib/selection";
import { COMMODITY_META, sortCommodities } from "@/lib/theme";
import { AllMarkets } from "./markets/AllMarkets";
import { CommodityTerminal } from "./markets/CommodityTerminal";

export default function Markets() {
  const { run, round, market } = useSelection();
  const [params, setParams] = useSearchParams();
  const overview = useOverview(run, round, market);
  const commodities = sortCommodities(overview.data?.commodities ?? []);
  const sel = params.get("c");
  const active = sel && commodities.includes(sel) ? sel : "all";

  const choose = (c: string) =>
    setParams((p) => {
      const n = new URLSearchParams(p);
      if (c === "all") n.delete("c");
      else n.set("c", c);
      return n;
    });

  return (
    <div>
      <PageHeader
        eyebrow="Markets"
        title={active === "all" ? "Cross-asset session" : `${COMMODITY_META[active]?.label ?? active} session`}
        description="Eight simulated commodity futures books, each calibrated to its contract's real volatility and tick grid. Shaded spans are the simulator's ground-truth manipulation episodes."
      />

      <div className="sticky top-[108px] z-20 -mx-1 mb-5 overflow-x-auto px-1 pb-1 md:top-[64px]">
        <div className="inline-flex min-w-full gap-1 rounded-xl border border-ink-700 bg-ink-900/90 p-1 backdrop-blur-xl">
          {["all", ...commodities].map((c) => {
            const on = c === active;
            const meta = COMMODITY_META[c];
            const snap = overview.data?.snapshot.find((s) => s.commodity === c);
            return (
              <button
                key={c}
                onClick={() => choose(c)}
                className={clsx(
                  "focus-ring relative flex shrink-0 items-center gap-2 rounded-lg px-3 py-1.5 text-[12.5px] font-medium transition-colors",
                  on ? "text-fg" : "text-fg-muted hover:text-fg-2",
                )}
              >
                {on && <motion.span layoutId="commodity-tab" className="absolute inset-0 rounded-lg border border-ink-600 bg-ink-700" transition={{ type: "spring", stiffness: 480, damping: 38 }} />}
                <span className="relative z-10 flex items-center gap-2">
                  {c === "all" ? <Globe2 size={14} /> : <span className="h-2 w-2 rounded-full" style={{ background: meta?.color }} />}
                  {c === "all" ? "All markets" : meta?.label ?? c}
                  {snap && snap.events > 0 && <span className="rounded bg-warn/15 px-1 font-mono text-[10px] text-warn">{snap.events}</span>}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {overview.error ? (
        <ErrorNote error={overview.error} />
      ) : !overview.data ? (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            {Array.from({ length: 8 }).map((_, i) => (
              <Skeleton key={i} className="h-28" />
            ))}
          </div>
          <Skeleton className="h-96" />
        </div>
      ) : active === "all" ? (
        <AllMarkets data={overview.data} onPick={choose} />
      ) : (
        <CommodityTerminal commodity={active} />
      )}
    </div>
  );
}
