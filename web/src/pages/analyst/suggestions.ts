import { useMemo } from "react";
import { useCases, useRoundSummary, useStrategist } from "@/lib/api";
import { marketLabel, useSelection } from "@/lib/selection";
import { PERSONAS, PERSONA_META } from "@/lib/theme";

export interface Suggestion {
  /** short chip text */
  label: string;
  /** the question actually sent */
  prompt: string;
  /** what it is about, for the chip's caption */
  kind: "accounts" | "account" | "performance" | "adversary" | "false-alerts" | "compare";
}

const pct = (v?: number | null) => (typeof v === "number" ? `${Math.round(v * 100)}%` : "");

/**
 * Questions built from what the user is looking at: the selected round and market, its top
 * case, its weakest persona, its false alerts and the strategist's most effective mutation.
 */
export function useSuggestions(): Suggestion[] {
  const { run, round, market, roundInfo, scopeMarket, pooled } = useSelection();
  const cases = useCases(run, round, market);
  const summary = useRoundSummary(run, round, scopeMarket);
  const strat = useStrategist(run);

  return useMemo(() => {
    if (round === undefined) return [];
    const nMarkets = roundInfo?.markets.length ?? 1;
    const m = market ? marketLabel(market, nMarkets) : "the test market";
    const where = nMarkets > 1 ? `round ${round}, ${m}` : `round ${round}`;
    const out: Suggestion[] = [];
    const cs = cases.data?.cases ?? [];

    // persona with the most cases in this market
    const counts = new Map<string, number>();
    cs.filter((c) => c.persona !== "anomaly").forEach((c) => counts.set(c.persona, (counts.get(c.persona) ?? 0) + 1));
    const busiest = [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
    if (busiest) {
      const label = PERSONA_META[busiest].label.toLowerCase();
      out.push({
        kind: "accounts",
        label: `Who was flagged for ${label} in ${where}?`,
        prompt: `Which accounts were flagged for ${label} in ${where}, and why? One row per account with its ground-truth status and the top SHAP features.`,
      });
    }

    const top = cs.find((c) => c.persona !== "anomaly");
    if (top) {
      out.push({
        kind: "account",
        label: `What did ${top.trader_id} do?`,
        prompt: `Explain why ${top.trader_id} (${top.commodity.replace(/_/g, " ")}) was flagged for ${PERSONA_META[top.persona]?.label.toLowerCase()} in ${where}, and summarise its orders and trades in that span.`,
      });
    }

    const noEp = cases.data?.kpi.no_episode ?? 0;
    if (noEp > 0) {
      out.push({
        kind: "false-alerts",
        label: `Why ${noEp} alerts with no episode?`,
        prompt: `In ${where}, which accounts raised alerts with no ground-truth episode, what kind of traders are they and what drove those alerts?`,
      });
    }

    const post = summary.data?.personas ?? {};
    const weakest = PERSONAS.filter((p) => typeof post[p]?.f1 === "number").sort((a, b) => (post[a].f1 ?? 1) - (post[b].f1 ?? 1))[0];
    if (weakest) {
      out.push({
        kind: "performance",
        label: `Why is ${PERSONA_META[weakest].label.toLowerCase()} weakest?`,
        prompt: `${PERSONA_META[weakest].label} has the lowest F1 (${post[weakest].f1?.toFixed(2)}) in ${pooled || nMarkets < 2 ? `round ${round}` : where}. How has it changed across rounds, how well did the frozen detector catch it, and what do its missed or weak cases have in common?`,
      });
    }

    // strategist's best-scoring mutation so far
    let best: { round: number; persona: string; score: number; cov?: number | null } | null = null;
    for (const h of strat.data ?? []) {
      for (const [p, d] of Object.entries(h.personas)) {
        if (typeof d.score === "number" && (!best || d.score > best.score)) best = { round: h.round, persona: p, score: d.score, cov: d.frozen_coverage };
      }
    }
    if (best) {
      out.push({
        kind: "adversary",
        label: `How did ${PERSONA_META[best.persona]?.label.toLowerCase()} evade in round ${best.round}?`,
        prompt: `The strategist's most effective mutation was ${PERSONA_META[best.persona]?.label.toLowerCase()} in round ${best.round} (score ${best.score.toFixed(2)}${best.cov != null ? `, frozen coverage ${pct(best.cov)}` : ""}). What did it change, why, and how did the detector recover after retraining?`,
      });
    }

    if (nMarkets > 1) {
      out.push({
        kind: "compare",
        label: `Compare Market A and B in round ${round}`,
        prompt: `Compare detection in Market A and Market B for round ${round}: F1 and recall per persona, and any persona where the markets differ a lot.`,
      });
    }
    out.push({
      kind: "performance",
      label: "Hardest persona before retraining?",
      prompt: "Across all rounds, which persona was hardest for the frozen detector to catch (before retraining), and how quickly did retraining recover it?",
    });
    return out;
  }, [round, market, roundInfo, cases.data, summary.data, strat.data, pooled]);
}
