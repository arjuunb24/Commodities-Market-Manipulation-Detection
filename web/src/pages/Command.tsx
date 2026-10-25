import { motion } from "framer-motion";
import { ArrowUpRight, Bot, Crosshair, Layers3, ShieldCheck } from "lucide-react";
import { useMemo } from "react";
import { Link } from "react-router-dom";
import type { EChartsOption } from "echarts";
import { EChart } from "@/components/charts/EChart";
import { base, grid, legend, tooltip, ttRow, ttTitle, valueAxis, axisCommon, hexA } from "@/components/charts/base";
import { NAV } from "@/components/shell/Sidebar";
import {
  AnimatedNumber,
  Callout,
  ChartCard,
  ErrorNote,
  PageHeader,
  PersonaMark,
  SectionTitle,
  Skeleton,
  StatGrid,
  StatTile,
  StatusBadge,
  fadeUp,
  stagger,
} from "@/components/ui";
import { useCases, useOverview, usePerformance, useRoundSummary } from "@/lib/api";
import { fmtInt, fmtNum, fmtPct, runLabel, tickClock } from "@/lib/format";
import { marketLabel, useSelection } from "@/lib/selection";
import { ScopeNote, ScopeSwitch, useScopeLabel } from "@/components/shell/ScopeSwitch";
import { CAT, INK, PERSONAS, PERSONA_META } from "@/lib/theme";

const mean = (xs: (number | null | undefined)[]) => {
  const v = xs.filter((x): x is number => typeof x === "number");
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
};

/** The selected test market's own figures (the tiles above pool every test market of the round). */
function MarketStrip() {
  const { run, round, market, roundInfo, search } = useSelection();
  const ov = useOverview(run, round, market);
  const cs = useCases(run, round, market);
  const total = roundInfo?.markets.length ?? 1;
  const name = market ? marketLabel(market, total) : "Test market";
  const range = roundInfo?.markets.find((m) => m.name === market)?.tick_range;
  const k = ov.data?.kpi;
  const c = cs.data?.kpi;
  const items: { label: string; value: number | undefined; fmt: (v: number) => string; tone?: string }[] = [
    { label: "Trades", value: k?.trades, fmt: fmtInt },
    { label: "Volume", value: k?.volume, fmt: fmtInt },
    { label: "Episodes", value: k?.events, fmt: fmtInt },
    { label: "Persona cases", value: c?.persona_cases, fmt: fmtInt },
    { label: "Confirmed", value: c?.confirmed, fmt: fmtInt, tone: "text-good" },
    { label: "No episode", value: c?.no_episode, fmt: fmtInt, tone: c?.no_episode ? "text-warn" : undefined },
  ];
  return (
    <motion.div key={market} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }} className="panel mt-4 px-5 py-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <span className="text-[13.5px] font-semibold text-fg">{name}</span>
          <span className="ml-2 font-mono text-[11.5px] text-fg-muted">
            {range ? `session ${tickClock(range[0])} to ${tickClock(range[1])}` : ""}
          </span>
        </div>
        <span className="text-[11.5px] text-fg-muted">
          {total > 1 ? "Activity and detector cases in this test market. Switch with the Scope control or the top bar." : "The only test market of this round."}
        </span>
      </div>
      <div className="mt-3 grid grid-cols-3 gap-x-6 gap-y-3 sm:grid-cols-6">
        {items.map((it) => (
          <div key={it.label}>
            <div className="text-[10.5px] uppercase tracking-[0.1em] text-fg-muted">{it.label}</div>
            <div className={`mt-1 text-[18px] font-semibold tracking-tight ${it.tone ?? "text-fg"}`}>
              {it.value === undefined ? <Skeleton className="h-5 w-14" /> : <AnimatedNumber value={it.value} format={it.fmt} />}
            </div>
          </div>
        ))}
      </div>
      <div className="mt-3 flex flex-wrap gap-4 text-[12px]">
        <Link to={{ pathname: "/markets", search }} className="text-accent hover:underline">Open this market</Link>
        <Link to={{ pathname: "/surveillance", search }} className="text-accent hover:underline">Review its cases</Link>
      </div>
    </motion.div>
  );
}

export default function Command() {
  const { run, runInfo, round, search, scopeMarket } = useSelection();
  const perf = usePerformance(run, scopeMarket);
  const summary = useRoundSummary(run, round, scopeMarket);
  const scope = useScopeLabel();
  const frozenMissing = !!scopeMarket && summary.data?.frozen_by_market === false;
  // Frozen-detector figures were stored pooled only before per-market recording: fall back to them, labelled.
  const pooledPerf = usePerformance(frozenMissing ? run : undefined, undefined);
  const pooledSummary = useRoundSummary(frozenMissing ? run : undefined, round, undefined);
  const lbl = run ? runLabel(run) : null;

  const series = useMemo(() => {
    const rows = perf.data?.rows ?? [];
    const preRows = frozenMissing ? pooledPerf.data?.rows ?? [] : rows;
    const rounds = perf.data?.rounds ?? [];
    const by = (src: typeof rows, phase: string, key: "f1" | "caught_coverage") =>
      rounds.map((r) => mean(src.filter((x) => x.round === r && x.phase === phase).map((x) => x[key])));
    return { rounds, post: by(rows, "post", "f1"), pre: by(preRows, "pre", "f1"), cov: by(preRows, "pre", "caught_coverage") };
  }, [perf.data, pooledPerf.data, frozenMissing]);
  const frozenName = frozenMissing ? "Frozen detector (pooled)" : "Frozen detector";

  const s = summary.data;
  const post = s?.personas ?? {};
  const pre = (frozenMissing ? pooledSummary.data?.pre_retrain : s?.pre_retrain) ?? {};
  const avgF1 = mean(Object.values(post).map((m) => m.f1));
  const avgCov = mean(Object.values(pre).map((m) => m.caught_coverage ?? m.episode_coverage));
  const ms = runInfo?.mutation_sources ?? {};
  const agentic = ms.agentic ?? 0;
  const totalMut = Object.values(runInfo?.mutation_sources ?? {}).reduce((a, b) => a + b, 0);

  const resilience: EChartsOption = base({
    grid: grid({ top: 40, right: 48 }),
    legend: legend({ data: ["After retraining", frozenName] }),
    tooltip: tooltip({
      formatter: (ps: unknown) => {
        const arr = ps as { axisValue: string; seriesName: string; value: number | null; color: string }[];
        return ttTitle(`Round ${arr[0]?.axisValue}`) + arr.map((p) => ttRow(p.color, p.seriesName, p.value === null ? "-" : fmtNum(p.value, 3))).join("");
      },
    }),
    xAxis: { type: "category", data: series.rounds.map(String), boundaryGap: false, ...axisCommon, name: "Round", nameLocation: "middle", nameGap: 26 },
    yAxis: valueAxis({ min: 0, max: 1, splitNumber: 4 }),
    series: [
      {
        name: "After retraining",
        type: "line",
        data: series.post,
        symbol: "circle",
        symbolSize: 8,
        lineStyle: { width: 2, color: CAT[0] },
        itemStyle: { color: CAT[0], borderColor: INK.surface, borderWidth: 2 },
        areaStyle: { color: { type: "linear", x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: hexA(CAT[0], 0.22) }, { offset: 1, color: hexA(CAT[0], 0) }] } },
        endLabel: { show: true, color: INK.secondary, fontSize: 11, formatter: (p: { value: unknown }) => fmtNum(p.value as number, 3) },
      },
      {
        name: frozenName,
        type: "line",
        data: series.pre,
        symbol: "emptyCircle",
        symbolSize: 8,
        connectNulls: false,
        lineStyle: { width: 2, color: CAT[1] },
        itemStyle: { color: CAT[1], borderColor: INK.surface, borderWidth: 2 },
      },
    ],
  });

  return (
    <div>
      <PageHeader
        right={<ScopeSwitch />}
        eyebrow="Run brief"
        title={lbl ? `Adversarial run, ${lbl.date}` : "Adversarial run"}
        description={
          <>
            Seed {lbl?.seed} · {runInfo?.rounds.length ?? 0} rounds · {runInfo?.n_test_markets ?? "-"} held-out test markets per round. Each round the
            strategist mutates four manipulation personas, the frozen detector is scored against them, then retrained on everything seen so far.
          </>
        }
      />

      <StatGrid className="grid-cols-2 lg:grid-cols-5">
        <StatTile label={`Mean F1 · R${round}`} value={avgF1} format={(v) => v.toFixed(3)} tone="accent" icon={<ShieldCheck size={15} />} sub={`After retraining · ${scope}`} />
        <StatTile
          label="Frozen coverage"
          value={avgCov}
          format={(v) => `${(v * 100).toFixed(0)}%`}
          icon={<Crosshair size={15} />}
          sub={round === 0 ? "Starts at round 1" : frozenMissing ? "Pooled: not stored per market for this run" : `Caught before retraining · ${scope}`}
        />
        <StatTile label="Alert windows" value={s?.n_flags ?? null} format={(v) => fmtInt(v)} icon={<Layers3 size={15} />} sub={`Round ${round} · ${scope}`} />
        <StatTile label="LLM mutations" value={agentic} format={(v) => `${Math.round(v)} / ${totalMut}`} icon={<Bot size={15} />} sub={`Kept after probing · ${ms.code_exploration ?? 0} won by code exploration`} />
        <StatTile label="LLM spend" value={runInfo?.llm_cost_usd ?? null} format={(v) => `$${v.toFixed(3)}`} sub={`${runInfo?.llm_calls ?? 0} calls this run`} />
      </StatGrid>

      <MarketStrip />
      <div className="mt-4">
        <ScopeNote frozenByMarket={summary.data?.frozen_by_market} />
      </div>

      <div className="mt-4">
        {perf.isLoading ? <Skeleton className="h-14" /> : perf.error ? <ErrorNote error={perf.error} /> : <Callout kind="accent" title="Evasion and recovery">{perf.data?.headline}</Callout>}
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-[1.35fr_1fr]">
        <ChartCard
          title="Detector resilience"
          subtitle={`${scope}. Mean F1 over the four personas. The frozen line is the detector from the previous round meeting this round's mutations; the gap to the retrained line is what retraining recovered.`}
          table={{
            columns: [
              { key: "round", label: "Round" },
              { key: "post", label: "F1 retrained", align: "right", format: (v) => fmtNum(v as number, 3) },
              { key: "pre", label: "F1 frozen", align: "right", format: (v) => fmtNum(v as number, 3) },
              { key: "cov", label: "Frozen coverage", align: "right", format: (v) => fmtPct(v as number) },
            ],
            rows: series.rounds.map((r, i) => ({ round: r, post: series.post[i], pre: series.pre[i], cov: series.cov[i] })),
          }}
        >
          {perf.isLoading ? <Skeleton className="h-[290px]" /> : <EChart option={resilience} height={290} />}
        </ChartCard>

        <div className="panel p-4 sm:p-5">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-[14px] font-semibold text-fg">Round {round} at a glance <span className="font-normal text-fg-muted">· {scope}</span></h3>
            <span className="eyebrow hidden sm:inline">Persona · F1 · frozen coverage</span>
          </div>
          <motion.ul variants={stagger(0.07)} initial="hidden" animate="show" className="space-y-2.5">
            {PERSONAS.filter((p) => post[p]).map((p) => {
              const m = post[p];
              const fr = pre[p];
              const src = s?.mutation_source?.[p];
              const eff = s?.effectiveness_ratio?.[p];
              return (
                <motion.li key={p} variants={fadeUp} className="rounded-lg border border-ink-700 bg-ink-800/50 p-3">
                  <div className="flex items-center justify-between gap-2">
                    <span className="flex items-center gap-2 text-[13px] font-medium text-fg">
                      <PersonaMark persona={p} size={10} />
                      {PERSONA_META[p].label}
                    </span>
                    {src && src !== "baseline" && (
                      <StatusBadge kind={src === "agentic" || src === "code_exploration" ? "neutral" : "warning"}>
                        {src === "agentic" ? "LLM mutation" : src === "code_exploration" ? "Code exploration" : src.replace(/_/g, " ")}
                      </StatusBadge>
                    )}
                  </div>
                  <div className="mt-2 grid grid-cols-2 gap-2 font-mono text-[11.5px] sm:grid-cols-4">
                    <div>
                      <div className="text-fg-muted">F1</div>
                      <div className="text-fg">{fmtNum(m.f1, 3)}</div>
                    </div>
                    <div>
                      <div className="text-fg-muted">Recall</div>
                      <div className="text-fg">{fmtNum(m.recall, 3)}</div>
                    </div>
                    <div>
                      <div className="text-fg-muted">{frozenMissing ? "Frozen (pooled)" : "Frozen cov."}</div>
                      <div className="text-fg">{fr ? fmtPct(fr.caught_coverage ?? fr.episode_coverage) : "-"}</div>
                    </div>
                    <div>
                      <div className="text-fg-muted">Effect</div>
                      <div className="text-fg">{eff !== undefined && eff !== null ? `${fmtNum(eff)}x` : "-"}</div>
                    </div>
                  </div>
                </motion.li>
              );
            })}
          </motion.ul>
        </div>
      </div>

      <SectionTitle>Workspace</SectionTitle>
      <motion.div variants={stagger(0.05)} initial="hidden" animate="show" className="grid grid-cols-2 gap-3 xl:grid-cols-5">
        {NAV.slice(1).map((n) => (
          <motion.div key={n.to} variants={fadeUp}>
            <Link
              to={{ pathname: n.to, search }}
              className="focus-ring group flex h-full flex-col rounded-xl border border-ink-700 bg-ink-850/70 p-4 transition hover:-translate-y-0.5 hover:border-ink-600 hover:bg-ink-800"
            >
              <div className="flex items-center justify-between">
                <n.icon size={18} className="text-accent" strokeWidth={1.8} />
                <ArrowUpRight size={15} className="text-fg-muted transition group-hover:translate-x-0.5 group-hover:-translate-y-0.5 group-hover:text-fg" />
              </div>
              <div className="mt-4 text-[14px] font-semibold text-fg">{n.label}</div>
              <div className="mt-1 text-[12.5px] leading-relaxed text-fg-muted">{n.hint}</div>
            </Link>
          </motion.div>
        ))}
      </motion.div>
    </div>
  );
}
