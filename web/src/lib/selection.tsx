import { createContext, useCallback, useContext, useEffect, useMemo, type ReactNode } from "react";
import { useSearchParams } from "react-router-dom";
import { useRounds, useRuns, type RoundInfo, type RunSummary } from "./api";

/**
 * Run / round / test-market selection, kept in the URL (?run=&round=&market=) so every
 * view is a shareable link. Defaults: newest run, its latest round, the first test market.
 */
interface Selection {
  runs: RunSummary[];
  run?: string;
  runInfo?: RunSummary;
  rounds: RoundInfo[];
  round?: number;
  roundInfo?: RoundInfo;
  market?: string;
  /** true = metric views show the pooled (all test markets) numbers instead of the selected market's */
  pooled: boolean;
  /** market to scope metric views to, or undefined when pooled */
  scopeMarket?: string;
  setPooled: (pooled: boolean) => void;
  loading: boolean;
  setRun: (run: string) => void;
  setRound: (round: number) => void;
  setMarket: (market: string) => void;
  search: string;
}

const Ctx = createContext<Selection | null>(null);

export function SelectionProvider({ children }: { children: ReactNode }) {
  const [params, setParams] = useSearchParams();
  const runsQ = useRuns();
  const runs = useMemo(() => (runsQ.data ?? []).filter((r) => r.rounds.length > 0), [runsQ.data]);

  const runParam = params.get("run");
  const run = runs.find((r) => r.run === runParam)?.run ?? runs[0]?.run;
  const runInfo = runs.find((r) => r.run === run);
  const roundsQ = useRounds(run);
  const rounds = roundsQ.data ?? [];

  const roundParam = params.get("round");
  const roundNum = roundParam !== null ? Number(roundParam) : NaN;
  const roundInfo = rounds.find((r) => r.round === roundNum) ?? rounds[rounds.length - 1];
  const round = roundInfo?.round;

  const marketParam = params.get("market");
  const market = roundInfo?.markets.find((m) => m.name === marketParam)?.name ?? roundInfo?.markets[0]?.name;

  const pooled = params.get("scope") === "pooled";

  const update = useCallback(
    (patch: Record<string, string | null>) => {
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          for (const [k, v] of Object.entries(patch)) {
            if (v === null) next.delete(k);
            else next.set(k, v);
          }
          return next;
        },
        { replace: false },
      );
    },
    [setParams],
  );

  // Make the resolved defaults explicit in the URL once they are known.
  useEffect(() => {
    if (!run || round === undefined) return;
    if (params.get("run") !== run || params.get("round") !== String(round) || (market && params.get("market") !== market)) {
      update({ run, round: String(round), market: market ?? null });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run, round, market]);

  const value: Selection = {
    runs,
    run,
    runInfo,
    rounds,
    round,
    roundInfo,
    market,
    pooled,
    scopeMarket: pooled ? undefined : market,
    setPooled: (p) => update({ scope: p ? "pooled" : null }),
    loading: runsQ.isLoading || (!!run && roundsQ.isLoading),
    setRun: (r) => update({ run: r, round: null, market: null, case: null }),
    setRound: (r) => update({ round: String(r), market: null, case: null }),
    setMarket: (m) => update({ market: m, case: null, scope: null }),
    search: run && round !== undefined ? `?run=${encodeURIComponent(run)}&round=${round}${market ? `&market=${market}` : ""}${pooled ? "&scope=pooled" : ""}` : "",
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useSelection() {
  const v = useContext(Ctx);
  if (!v) throw new Error("useSelection outside SelectionProvider");
  return v;
}

export const marketLabel = (name: string, total: number) => {
  const idx = name === "test" ? 1 : Number(name.split("_")[1]) + 1;
  return total > 1 ? `Market ${String.fromCharCode(64 + idx)}` : "Test market";
};
