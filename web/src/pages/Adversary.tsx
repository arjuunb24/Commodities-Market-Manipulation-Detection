import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowRight, Bot, ChevronDown, FlaskConical, PenLine, ShieldAlert, Undo2 } from "lucide-react";
import { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { EChart } from "@/components/charts/EChart";
import { axisCommon, base, grid, legend, tooltip, ttRow, ttTitle, valueAxis } from "@/components/charts/base";
import { ChartCard, ErrorNote, PageHeader, PersonaMark, SectionTitle, Segmented, Skeleton, StatGrid, StatTile, StatusBadge, fadeUp, stagger } from "@/components/ui";
import { usePerformance, useStrategist, type StrategistPersona } from "@/lib/api";
import { fmtNum, fmtPct, fmtVal } from "@/lib/format";
import { useSelection } from "@/lib/selection";
import { ScopeNote, ScopeSwitch, useScopeLabel } from "@/components/shell/ScopeSwitch";
import { INK, PERSONAS, PERSONA_META } from "@/lib/theme";

type Lens = "score" | "ratio" | "coverage";

function SourceBadge({ source }: { source: string }) {
  if (source === "agentic")
    return (
      <span className="inline-flex items-center gap-1 rounded-md border border-accent/30 bg-accent/10 px-1.5 py-0.5 text-[11px] font-medium text-accent">
        <Bot size={12} /> LLM strategist
      </span>
    );
  if (source === "hand_authored")
    return (
      <span className="inline-flex items-center gap-1 rounded-md border border-warn/30 bg-warn/10 px-1.5 py-0.5 text-[11px] font-medium text-warn">
        <PenLine size={12} /> Hand-written fallback
      </span>
    );
  return <span className="rounded-md border border-ink-600 px-1.5 py-0.5 text-[11px] text-fg-muted">{source.replace(/_/g, " ")}</span>;
}

function Metric({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div title={hint}>
      <div className="text-[10.5px] uppercase tracking-[0.1em] text-fg-muted">{label}</div>
      <div className="num mt-0.5 font-mono text-[15px] font-medium text-fg">{value}</div>
    </div>
  );
}

function PersonaCard({ persona, d }: { persona: string; d: StrategistPersona }) {
  const [open, setOpen] = useState(false);
  const m = PERSONA_META[persona];
  const relabel = d.frozen_coverage !== null && d.frozen_coverage !== undefined && d.own_label_coverage !== null && d.own_label_coverage !== undefined && Math.abs(d.own_label_coverage - d.frozen_coverage) >= 0.05;
  const effLow = typeof d.effect_ratio === "number" && d.effect_ratio < 0.5;
  return (
    <motion.div variants={fadeUp} className="flex flex-col rounded-xl border border-ink-700 bg-ink-850/80 p-4">
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 text-[13.5px] font-semibold text-fg">
          <PersonaMark persona={persona} size={11} /> {m?.label}
        </div>
        <SourceBadge source={d.source} />
      </div>

      <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-3">
        <Metric label={d.frozen_from_pooled ? "Frozen cov. (pooled)" : "Frozen coverage"} value={fmtPct(d.frozen_coverage)} hint="Share of each episode's windows that raised ANY alert from the detector trained before this mutation (lower = better evasion). Marked pooled when this run did not store it per market." />
        <Metric label={d.frozen_from_pooled ? "Score (pooled)" : "Score"} value={fmtNum(d.score, 3)} hint="(0.5 x (1 - coverage) + 0.3 x missed + 0.2 x delay) x min(1, effect ratio); 0 below half the round-0 effect. Marked pooled when this run did not store it per market." />
        <Metric label="Retrained recall" value={fmtPct(d.retrained_recall)} hint="Recall after the detector retrained on this round." />
        <Metric label="Effect vs R0" value={typeof d.effect_ratio === "number" ? `${fmtNum(d.effect_ratio)}x` : "-"} hint="Manipulation effectiveness relative to round 0." />
      </div>

      <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-tint/[0.05]" title="Evasion (1 - frozen coverage)">
        <motion.div
          className="h-full rounded-full"
          style={{ background: m?.color }}
          initial={{ width: 0 }}
          animate={{ width: `${Math.max(0, 1 - (d.frozen_coverage ?? 1)) * 100}%` }}
          transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
        />
      </div>
      <div className="mt-1 flex justify-between font-mono text-[10.5px] text-fg-muted">
        <span>evasion {fmtPct(1 - (d.frozen_coverage ?? 1))}</span>
        <span>
          caught {d.episodes_caught ?? "-"}/{d.episodes_total ?? "-"} · delay {fmtNum(d.frozen_delay, 0)} ticks
        </span>
      </div>

      <div className="mt-3 space-y-1.5">
        {relabel && (
          <div className="text-[11.5px] leading-relaxed text-fg-muted">
            Own label caught {fmtPct(d.own_label_coverage)}
            {typeof d.family_coverage === "number" && <>, fake-depth family {fmtPct(d.family_coverage)}</>}; the rest was flagged under another persona's label.
          </div>
        )}
        {typeof d.weak_share === "number" && d.weak_share > 0 && (
          <div className="text-[11.5px] text-fg-muted">{fmtPct(d.weak_share)} of episodes were barely caught (under 20% of their windows).</div>
        )}
        {effLow && <StatusBadge kind="critical">Effect below half of round 0, score zeroed</StatusBadge>}
        {d.recovered_from_round !== null && d.recovered_from_round !== undefined && (
          <StatusBadge kind="warning">Effect recovery: reset to round {d.recovered_from_round} first</StatusBadge>
        )}
        {d.guard_reverts.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 text-[11.5px] text-warn">
            <Undo2 size={12} /> Effect guard reverted {d.guard_reverts.map((g) => `${g.knob} ${fmtVal(g.old)} to ${fmtVal(g.new)}`).join(", ")}
          </div>
        )}
      </div>

      <div className="mt-3 border-t border-ink-700 pt-3">
        <div className="eyebrow mb-1.5">Changes</div>
        {d.changes.length ? (
          <ul className="space-y-1">
            {d.changes.map((c) => (
              <li key={c.knob} className="flex items-center justify-between gap-2 font-mono text-[11.5px]">
                <span className="truncate text-fg-2">{c.knob}</span>
                <span className="flex shrink-0 items-center gap-1.5">
                  <span className="text-fg-muted">{fmtVal(c.old)}</span>
                  <ArrowRight size={11} className="text-fg-muted" />
                  <span className="font-semibold text-fg">{fmtVal(c.new)}</span>
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <div className="text-[12px] text-fg-muted">No parameter changes.</div>
        )}
      </div>

      {d.probe && d.probe.candidates.length > 0 && (
        <div className="mt-3 border-t border-ink-700 pt-3">
          <div className="eyebrow mb-1.5 flex items-center gap-1.5">
            <FlaskConical size={11} /> Probe-tested candidates
          </div>
          <table className="w-full font-mono text-[11px]">
            <thead>
              <tr className="text-fg-muted">
                <th className="py-0.5 text-left font-normal">#</th>
                <th className="py-0.5 text-right font-normal">cov</th>
                <th className="py-0.5 text-right font-normal">effect</th>
                <th className="py-0.5 text-right font-normal">score</th>
              </tr>
            </thead>
            <tbody>
              {d.probe.candidates.map((c, i) => {
                const chosen = i === d.probe!.chosen;
                return (
                  <tr key={i} className={clsx(chosen ? "text-accent" : "text-fg-2")} title={Object.entries(c.changes ?? {}).map(([k, v]) => `${k}=${fmtVal(v)}`).join(", ")}>
                    <td className="py-0.5">
                      {i}
                      {c.origin ? " exp" : ""}
                      {chosen ? " kept" : ""}
                    </td>
                    <td className="py-0.5 text-right">{fmtPct(c.coverage)}</td>
                    <td className="py-0.5 text-right">{typeof c.effect_ratio === "number" ? `${fmtNum(c.effect_ratio)}x` : "-"}</td>
                    <td className="py-0.5 text-right">{fmtNum(c.score, 3)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {(d.reasoning.points.length > 0 || d.reasoning.notes.length > 0) && (
        <div className="mt-auto pt-3">
          <button onClick={() => setOpen((o) => !o)} className="focus-ring flex w-full items-center justify-between rounded-lg border border-ink-700 px-3 py-2 text-[12px] text-fg-2 hover:bg-tint/[0.02]">
            Strategist reasoning
            <ChevronDown size={14} className={clsx("transition-transform", open && "rotate-180")} />
          </button>
          <AnimatePresence initial={false}>
            {open && (
              <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                <ol className="mt-2 list-decimal space-y-1.5 pl-5 text-[12px] leading-relaxed text-fg-2 marker:text-fg-muted">
                  {d.reasoning.points.map((p, i) => (
                    <li key={i}>{p}</li>
                  ))}
                </ol>
                {d.reasoning.notes.map((n, i) => (
                  <div key={i} className="mt-2 rounded-md bg-ink-800 px-2.5 py-1.5 font-mono text-[10.5px] leading-relaxed text-fg-muted">
                    {n}
                  </div>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </motion.div>
  );
}

const FROZEN_FIELDS = ["frozen_coverage", "own_label_coverage", "family_coverage", "weak_share", "episodes_caught", "episodes_total", "frozen_delay", "frozen_recall", "score"] as const;
const pick = (o: StrategistPersona, keys: readonly (keyof StrategistPersona)[]) => Object.fromEntries(keys.map((k) => [k, o[k]])) as Partial<StrategistPersona>;

export default function Adversary() {
  const { run, scopeMarket } = useSelection();
  const q = useStrategist(run, scopeMarket);
  const perf = usePerformance(run, scopeMarket);
  const scope = useScopeLabel();
  const [lens, setLens] = useState<Lens>("coverage");
  const [openRound, setOpenRound] = useState<number | null>(null);

  // Runs made before per-market recording have no frozen-detector results per market. Rather
  // than leave those figures empty, fall back to the pooled ones and tag them as pooled.
  const frozenMissing = !!scopeMarket && perf.data?.frozen_by_market === false;
  const pooledQ = useStrategist(frozenMissing ? run : undefined, undefined);
  const hist = useMemo(() => {
    const base = q.data ?? [];
    if (!frozenMissing || !pooledQ.data) return base;
    return base.map((h) => {
      const pooledRound = pooledQ.data!.find((x) => x.round === h.round);
      return {
        ...h,
        personas: Object.fromEntries(
          Object.entries(h.personas).map(([p, d]) => {
            const pd = pooledRound?.personas[p];
            if (!pd || d.frozen_coverage !== null && d.frozen_coverage !== undefined) return [p, d];
            return [p, { ...d, ...pick(pd, FROZEN_FIELDS), frozen_from_pooled: true }];
          }),
        ),
      };
    });
  }, [q.data, pooledQ.data, frozenMissing]);
  const rounds = hist.map((h) => h.round);
  const latest = rounds[rounds.length - 1];
  const expanded = openRound ?? latest;

  const lensOpt = useMemo((): EChartsOption => {
    const key: Record<Lens, keyof StrategistPersona> = { score: "score", ratio: "effect_ratio", coverage: "frozen_coverage" };
    const pct = lens === "coverage";
    const series = PERSONAS.map((p) => ({
      name: PERSONA_META[p].label,
      type: "line" as const,
      data: hist.map((h) => (h.personas[p]?.[key[lens]] as number | null | undefined) ?? null),
      symbol: PERSONA_META[p].symbol,
      symbolSize: 8,
      lineStyle: { width: 2, color: PERSONA_META[p].color },
      itemStyle: { color: PERSONA_META[p].color, borderColor: INK.surface, borderWidth: 2 },
      endLabel: { show: true, color: INK.secondary, fontSize: 10.5, formatter: () => PERSONA_META[p].short },
      labelLayout: { moveOverlap: "shiftY" as const },
      emphasis: { focus: "series" as const },
    }));
    if (lens === "ratio")
      (series[0] as Record<string, unknown>).markLine = {
        silent: true,
        symbol: "none",
        lineStyle: { color: INK.ref, type: "solid" },
        label: { color: INK.muted, fontSize: 10, formatter: (p: { value: number }) => (p.value === 0.5 ? "floor 0.5x" : "round 0") },
        data: [{ yAxis: 0.5 }, { yAxis: 1 }],
      };
    return base({
      grid: grid({ top: 40, right: 44 }),
      legend: legend(),
      tooltip: tooltip({
        formatter: (ps: unknown) => {
          const arr = ps as { axisValue: string; seriesName: string; value: number | null; color: string }[];
          return ttTitle(`Round ${arr[0]?.axisValue}`) + arr.filter((x) => x.value !== null).map((x) => ttRow(x.color, x.seriesName, pct ? fmtPct(x.value, 1) : lens === "ratio" ? `${fmtNum(x.value)}x` : fmtNum(x.value, 3))).join("");
        },
      }),
      xAxis: { type: "category", data: rounds.map(String), boundaryGap: false, ...axisCommon, splitLine: { show: false }, name: "Round", nameLocation: "middle", nameGap: 26 },
      yAxis: valueAxis({ min: 0, max: pct ? 1 : undefined, splitNumber: 4, axisLabel: { ...axisCommon.axisLabel, formatter: (v: number) => (pct ? `${Math.round(v * 100)}%` : lens === "ratio" ? `${v.toFixed(1)}x` : v.toFixed(2)) } }),
      series,
    });
  }, [hist, rounds, lens]);

  if (q.error) return <ErrorNote error={q.error} />;
  if (!q.data)
    return (
      <div className="space-y-4">
        <Skeleton className="h-12 w-80" />
        <Skeleton className="h-80" />
      </div>
    );

  const all = hist.flatMap((h) => Object.entries(h.personas).map(([p, d]) => ({ round: h.round, persona: p, d })));
  const llm = all.filter((x) => x.d.source === "agentic").length;
  const best = all.reduce<(typeof all)[number] | null>((b, x) => ((x.d.score ?? -1) > (b?.d.score ?? -1) ? x : b), null);
  const reverts = all.reduce((n, x) => n + x.d.guard_reverts.length, 0);
  const probed = all.filter((x) => x.d.probe && x.d.probe.candidates.length > 1).length;

  return (
    <div>
      <PageHeader
        eyebrow="Adversary"
        title="Strategist mutations"
        right={<ScopeSwitch />}
        description={`Before each round an LLM strategist re-tunes every manipulation persona to slip past the detector while keeping the manipulation effective. Candidates are probe-tested against the frozen detector first; a code guard reverts changes that would kill the effect. Outcomes below: ${scope}.`}
      />

      {!hist.length ? (
        <div className="panel p-8 text-center text-[13px] text-fg-muted">No strategist rounds yet; mutations start at round 1.</div>
      ) : (
        <>
          <div className="mb-4">
            <ScopeNote frozenByMarket={perf.data?.frozen_by_market} />
          </div>
          <StatGrid className="grid-cols-2 lg:grid-cols-4">
            <StatTile label="LLM mutations" value={llm} format={(v) => `${Math.round(v)} / ${all.length}`} icon={<Bot size={15} />} sub="The rest used the hand-written library" />
            <StatTile label="Best evasion score" value={best?.d.score ?? null} format={(v) => v.toFixed(3)} tone="accent" sub={best ? `${PERSONA_META[best.persona]?.label}, round ${best.round}${frozenMissing ? " · pooled" : ""}` : undefined} />
            <StatTile label="Probe-tested" value={probed} format={(v) => String(Math.round(v))} icon={<FlaskConical size={15} />} sub="Mutations chosen among several candidates" />
            <StatTile label="Guard reverts" value={reverts} format={(v) => String(Math.round(v))} icon={<ShieldAlert size={15} />} sub="Knob changes undone to keep the effect" />
          </StatGrid>

          <div className="mt-4">
            <ChartCard
              title="How each persona fared"
              subtitle={
                (frozenMissing && lens !== "ratio" ? "Pooled across both markets: this run did not store frozen-detector results per market. " : "") +
                (lens === "coverage"
                  ? "Frozen-detector coverage: share of episode windows caught before retraining. Lower is better for the adversary."
                  : lens === "ratio"
                    ? "Manipulation effect relative to round 0. Below the 0.5x floor a mutation scores zero: it evaded by not manipulating."
                    : "Mutation score = evasion x retained effect (capped at 1x).")
              }
              actions={
                <Segmented
                  size="sm"
                  value={lens}
                  onChange={setLens}
                  options={[
                    { value: "coverage", label: "Coverage" },
                    { value: "ratio", label: "Effect" },
                    { value: "score", label: "Score" },
                  ]}
                />
              }
              table={{
                columns: [
                  { key: "round", label: "Round" },
                  { key: "persona", label: "Persona", format: (v) => PERSONA_META[String(v)]?.label ?? String(v) },
                  { key: "cov", label: "Frozen cov.", align: "right", format: (v) => fmtPct(v as number) },
                  { key: "eff", label: "Effect", align: "right", format: (v) => (typeof v === "number" ? `${fmtNum(v)}x` : "-") },
                  { key: "score", label: "Score", align: "right", format: (v) => fmtNum(v as number, 3) },
                ],
                rows: all.map((x) => ({ round: x.round, persona: x.persona, cov: x.d.frozen_coverage, eff: x.d.effect_ratio, score: x.d.score })),
              }}
            >
              <EChart option={lensOpt} height={320} />
            </ChartCard>
          </div>

          {perf.data && (
            <div className="mt-3 text-[12px] text-fg-muted">
              Effect metrics: spoofing and layering use fake depth share, wash trading the self-crossed share of volume, pump and dump the ignition pressure.
            </div>
          )}

          <SectionTitle>Round by round</SectionTitle>
          <div className="relative space-y-3 pl-6">
            <div className="absolute bottom-2 left-[7px] top-2 w-px bg-ink-600" />
            {[...hist].reverse().map((h) => {
              const isOpen = h.round === expanded;
              const scores = Object.values(h.personas).map((p) => p.score ?? 0);
              return (
                <div key={h.round} className="relative">
                  <div className={clsx("absolute -left-6 top-4 h-3.5 w-3.5 rounded-full border-2", isOpen ? "border-accent bg-accent/30" : "border-ink-600 bg-ink-900")} />
                  <div className="panel overflow-hidden">
                    <button onClick={() => setOpenRound(isOpen ? -1 : h.round)} className="focus-ring flex w-full flex-wrap items-center justify-between gap-3 px-5 py-3.5 text-left">
                      <div className="flex items-center gap-3">
                        <span className="font-mono text-[13px] font-semibold text-fg">Round {h.round}</span>
                        <span className="text-[12px] text-fg-muted">mutations tested in this round</span>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="hidden items-center gap-3 sm:flex">
                          {PERSONAS.filter((p) => h.personas[p]).map((p) => (
                            <span key={p} className="flex items-center gap-1 font-mono text-[11px] text-fg-2" title={`${PERSONA_META[p].label} score`}>
                              <PersonaMark persona={p} /> {fmtNum(h.personas[p].score, 2)}
                            </span>
                          ))}
                        </div>
                        <span className="font-mono text-[11px] text-fg-muted">best {fmtNum(Math.max(...scores), 3)}</span>
                        <ChevronDown size={16} className={clsx("text-fg-muted transition-transform", isOpen && "rotate-180")} />
                      </div>
                    </button>
                    <AnimatePresence initial={false}>
                      {isOpen && (
                        <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.3 }} className="overflow-hidden">
                          <motion.div variants={stagger(0.06)} initial="hidden" animate="show" className="grid gap-3 border-t border-ink-700 p-4 md:grid-cols-2 2xl:grid-cols-4">
                            {PERSONAS.filter((p) => h.personas[p]).map((p) => (
                              <PersonaCard key={p} persona={p} d={h.personas[p]} />
                            ))}
                          </motion.div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
