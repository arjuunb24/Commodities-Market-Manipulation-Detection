import { Layers } from "lucide-react";
import { marketLabel, useSelection } from "@/lib/selection";
import { Callout, Segmented } from "../ui";

/** Choose what metric views describe: one test market (default) or every market pooled. */
export function ScopeSwitch() {
  const { roundInfo, market, pooled, setMarket, setPooled } = useSelection();
  const markets = roundInfo?.markets ?? [];
  if (markets.length < 2) return null;
  const value = pooled ? "__pooled" : market ?? markets[0].name;
  return (
    <div className="flex items-center gap-2">
      <span className="eyebrow hidden sm:inline">Scope</span>
      <Segmented
        size="sm"
        value={value}
        onChange={(v) => (v === "__pooled" ? setPooled(true) : setMarket(v))}
        options={[
          ...markets.map((m) => ({ value: m.name, label: marketLabel(m.name, markets.length), title: "This test market only" })),
          { value: "__pooled", label: <><Layers size={12} /> Both pooled</>, title: "All test markets of each round pooled (the pipeline's official numbers)" },
        ]}
      />
    </div>
  );
}

export function useScopeLabel() {
  const { roundInfo, market, pooled } = useSelection();
  const n = roundInfo?.markets.length ?? 1;
  if (pooled || n < 2) return n < 2 ? "Test market" : "Both markets pooled";
  return market ? marketLabel(market, n) : "Test market";
}

/** Explains what the per-market scope can and cannot show for this run. */
export function ScopeNote({ frozenByMarket }: { frozenByMarket?: boolean }) {
  const { pooled, roundInfo } = useSelection();
  const label = useScopeLabel();
  if (pooled || (roundInfo?.markets.length ?? 1) < 2) return null;
  if (frozenByMarket === false)
    return (
      <Callout title={`Showing ${label} only`}>
        Retrained-detector metrics and manipulation effect are this market's own. Runs made before per-market recording stored frozen-detector results
        (before retraining) pooled only, so frozen coverage, frozen lines and mutation scores are shown from both markets pooled and labelled
        &ldquo;pooled&rdquo;. New runs record all of them per market.
      </Callout>
    );
  return null;
}
