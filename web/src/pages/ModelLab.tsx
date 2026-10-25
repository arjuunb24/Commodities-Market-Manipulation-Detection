import clsx from "clsx";
import { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { EChart } from "@/components/charts/EChart";
import { axisCommon, base, grid, legend, tooltip, ttRow, ttTitle, valueAxis, MONO, hexA, narrow } from "@/components/charts/base";
import { Callout, ChartCard, Chip, DataTable, ErrorNote, PageHeader, PersonaMark, SectionTitle, Segmented, Skeleton, StatGrid, StatTile } from "@/components/ui";
import { useParams as useParamHistory, usePerformance, type PerfRow } from "@/lib/api";
import { fmtNum, fmtPct, fmtVal } from "@/lib/format";
import { useSelection } from "@/lib/selection";
import { ScopeNote, ScopeSwitch, useScopeLabel } from "@/components/shell/ScopeSwitch";
import { CAT, INK, MARKS, PERSONAS, PERSONA_META, SEQ_BLUE } from "@/lib/theme";

type Metric = "f1" | "precision" | "recall" | "fpr";
const METRIC_LABEL: Record<Metric, string> = { f1: "F1", precision: "Precision", recall: "Recall", fpr: "False positive rate" };

const mean = (xs: (number | null | undefined)[]) => {
  const v = xs.filter((x): x is number => typeof x === "number");
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
};

function personaLines(rows: PerfRow[], rounds: number[], metric: Metric, phase: "pre" | "post", dashed = false) {
  return PERSONAS.filter((p) => rows.some((r) => r.persona === p)).map((p) => {
    const m = PERSONA_META[p];
    const data = rounds.map((r) => rows.find((x) => x.round === r && x.persona === p && x.phase === phase)?.[metric] ?? null);
    return {
      name: `${m.label}${dashed ? " · frozen" : ""}`,
      type: "line" as const,
      data,
      connectNulls: false,
      symbol: dashed ? `empty${m.symbol === "rect" ? "Rect" : m.symbol === "triangle" ? "Triangle" : m.symbol === "diamond" ? "Diamond" : "Circle"}` : m.symbol,
      symbolSize: 8,
      lineStyle: { width: 2, color: m.color, type: dashed ? ("dashed" as const) : ("solid" as const), opacity: dashed ? 0.85 : 1 },
      itemStyle: { color: m.color, borderColor: INK.surface, borderWidth: dashed ? 1.5 : 2 },
      emphasis: { focus: "series" as const },
      endLabel: dashed ? undefined : { show: true, color: INK.secondary, fontSize: 10.5, formatter: () => m.short },
      labelLayout: { moveOverlap: "shiftY" as const },
    };
  });
}

function lineOption(series: unknown[], rounds: number[], yMax = 1, pct = false): EChartsOption {
  return base({
    grid: grid({ top: 44, right: 40 }),
    legend: legend({ type: "scroll", left: 0, right: 0 }),
    tooltip: tooltip({
      formatter: (ps: unknown) => {
        const arr = ps as { axisValue: string; seriesName: string; value: number | null; color: string }[];
        return ttTitle(`Round ${arr[0]?.axisValue}`) + arr.filter((p) => p.value !== null && p.value !== undefined).map((p) => ttRow(p.color, p.seriesName, pct ? fmtPct(p.value, 1) : fmtNum(p.value, 3))).join("");
      },
    }),
    xAxis: { type: "category", data: rounds.map(String), boundaryGap: false, ...axisCommon, splitLine: { show: false }, name: "Round", nameLocation: "middle", nameGap: 26 },
    yAxis: valueAxis({ min: 0, max: yMax, splitNumber: 4, axisLabel: { ...axisCommon.axisLabel, formatter: (v: number) => (pct ? `${Math.round(v * 100)}%` : v.toFixed(2)) } }),
    series: series as EChartsOption["series"],
  });
}

export default function ModelLab() {
  const { run, scopeMarket } = useSelection();
  const q = usePerformance(run, scopeMarket);
  const scope = useScopeLabel();
  const params = useParamHistory(run);
  const [prePost, setPrePost] = useState<"recall" | "f1">("recall");
  const [evo, setEvo] = useState<Metric>("f1");
  const [paramPersonas, setParamPersonas] = useState<string[]>([...PERSONAS]);

  const d = q.data;
  const rows = d?.rows ?? [];
  const rounds = d?.rounds ?? [];
  const post = rows.filter((r) => r.phase === "post");
  const latest = rounds[rounds.length - 1];
  const last = post.filter((r) => r.round === latest);

  const hasFrozen = q.data?.frozen_by_market !== false;
  // Older runs stored frozen-detector results pooled only: draw those, labelled, instead of nothing.
  const pooledQ = usePerformance(!hasFrozen ? run : undefined, undefined);
  const prePostOpt = useMemo(() => {
    const frozen = hasFrozen
      ? personaLines(rows, rounds, prePost, "pre", true)
      : personaLines(pooledQ.data?.rows ?? [], rounds, prePost, "pre", true).map((sr) => ({ ...sr, name: sr.name.replace("· frozen", "· frozen (pooled)") }));
    return lineOption([...personaLines(rows, rounds, prePost, "post"), ...frozen], rounds);
  }, [rows, rounds, prePost, hasFrozen, pooledQ.data]);
  const evoOpt = useMemo(() => {
    const series = personaLines(rows, rounds, evo, "post");
    if (evo !== "fpr" && series.length)
      (series[0] as Record<string, unknown>).markLine = {
        silent: true,
        symbol: "none",
        lineStyle: { color: INK.ref, type: "solid" },
        label: { color: INK.muted, fontSize: 10, formatter: "{c}" },
        data: [{ yAxis: 0.9 }, { yAxis: 0.7 }],
      };
    const maxFpr = Math.max(0.01, ...post.map((r) => r.fpr ?? 0)) * 1.3;
    return lineOption(series, rounds, evo === "fpr" ? +maxFpr.toFixed(3) : 1, evo === "fpr");
  }, [rows, rounds, evo, post]);

  const heat = useMemo((): EChartsOption => {
    const ps = PERSONAS.filter((p) => post.some((r) => r.persona === p));
    const cells: [number, number, number | null][] = [];
    ps.forEach((p, y) => rounds.forEach((r, x) => cells.push([x, y, post.find((v) => v.round === r && v.persona === p)?.f1 ?? null])));
    return base({
      grid: grid({ top: 8, right: 70, bottom: 6 }),
      tooltip: { ...tooltip({ trigger: "item" }), formatter: (p: unknown) => {
        const v = (p as { value: [number, number, number] }).value;
        return ttTitle(`Round ${rounds[v[0]]}`) + ttRow(PERSONA_META[ps[v[1]]].color, PERSONA_META[ps[v[1]]].label, fmtNum(v[2], 3));
      } },
      xAxis: { type: "category", data: rounds.map((r) => `R${r}`), position: "top", ...axisCommon, splitLine: { show: false }, axisLine: { show: false } },
      yAxis: { type: "category", data: ps.map((p) => PERSONA_META[p].label), inverse: true, ...axisCommon, splitLine: { show: false }, axisLine: { show: false }, axisLabel: { color: INK.secondary, fontSize: 11.5 } },
      visualMap: { min: 0.5, max: 1, orient: "vertical", right: 0, top: "middle", itemHeight: 130, itemWidth: 10, inRange: { color: SEQ_BLUE.slice(1, 6) }, textStyle: { color: INK.muted, fontFamily: MONO, fontSize: 10 }, text: ["1.0", "0.5"] },
      series: [{ type: "heatmap", data: cells, itemStyle: { borderColor: INK.surface, borderWidth: 3, borderRadius: 4 }, label: { show: true, color: INK.primary, fontFamily: MONO, fontSize: narrow() ? 9 : 11.5, formatter: (p: { value: unknown }) => fmtNum((p.value as number[])[2], narrow() ? 2 : 3) } }],
    });
  }, [post, rounds]);

  const radar = useMemo((): EChartsOption => {
    const first = rounds[0];
    const show = rounds.length > 1 ? [first, latest] : [first];
    const colors = [MARKS.neutral, CAT[0]];
    return base({
      legend: legend({ data: show.map((r) => `Round ${r}`), right: "center", top: 0 }),
      tooltip: { ...tooltip({ trigger: "item" }) },
      radar: {
        center: ["50%", "56%"],
        radius: "66%",
        indicator: [{ name: "Precision", max: 1 }, { name: "Recall", max: 1 }, { name: "F1", max: 1 }, { name: "1 - FPR x10", max: 1 }],
        axisName: { color: INK.secondary, fontSize: 11.5 },
        splitLine: { lineStyle: { color: INK.grid } },
        splitArea: { show: false },
        axisLine: { lineStyle: { color: INK.grid } },
      },
      series: [{
        type: "radar",
        data: show.map((r, i) => {
          const rr = post.filter((x) => x.round === r);
          return {
            name: `Round ${r}`,
            value: [mean(rr.map((x) => x.precision)), mean(rr.map((x) => x.recall)), mean(rr.map((x) => x.f1)), 1 - Math.min(1, (mean(rr.map((x) => x.fpr)) ?? 0) * 10)],
            lineStyle: { color: colors[i], width: 2 },
            itemStyle: { color: colors[i] },
            areaStyle: { color: i ? hexA(CAT[0], 0.18) : hexA(MARKS.neutral, 0.08) },
            symbolSize: 6,
          };
        }),
      }],
    });
  }, [post, rounds, latest]);

  const pr = useMemo((): EChartsOption => {
    const iso = [0.5, 0.7, 0.8, 0.9].map((f) => {
      const pts: [number, number][] = [];
      for (let p = f / 2 + 0.005; p <= 1.0001; p += 0.005) {
        const r = (f * p) / (2 * p - f);
        if (r > 0 && r <= 1) pts.push([+p.toFixed(3), +r.toFixed(4)]);
      }
      return {
        type: "line" as const,
        data: pts,
        showSymbol: false,
        silent: true,
        lineStyle: { color: INK.grid, width: 1 },
        endLabel: { show: true, formatter: `F1 ${f}`, color: INK.muted, fontSize: 10 },
        tooltip: { show: false },
      };
    });
    const ps = PERSONAS.filter((p) => post.some((r) => r.persona === p));
    const maxSup = Math.max(1, ...post.map((r) => r.support ?? 0));
    // Zoom to where the points are (iso-F1 curves still give the scale).
    const lowP = Math.max(0, Math.floor((Math.min(...post.map((r) => r.precision ?? 1)) - 0.05) * 10) / 10);
    const lowR = Math.max(0, Math.floor((Math.min(...post.map((r) => r.recall ?? 1)) - 0.05) * 10) / 10);
    return base({
      grid: grid({ top: 40, right: 50 }),
      legend: legend({ data: ps.map((p) => PERSONA_META[p].label) }),
      tooltip: { ...tooltip({ trigger: "item" }), formatter: (p: unknown) => {
        const q2 = p as { seriesName: string; value: [number, number, number, number]; color: string };
        return ttTitle(`${q2.seriesName} · round ${q2.value[2]}`) + ttRow(q2.color, "Precision", fmtNum(q2.value[0], 3)) + ttRow(q2.color, "Recall", fmtNum(q2.value[1], 3)) + ttRow(INK.muted, "Support", String(q2.value[3]));
      } },
      xAxis: valueAxis({ min: Math.min(lowP, 0.5), max: 1, name: "Precision", nameLocation: "middle", nameGap: 26 }),
      // On phones the one-row legend takes the axis-name slot; the tooltip still names both axes.
      yAxis: valueAxis({ min: Math.min(lowR, 0.5), max: 1, name: narrow() ? "" : "Recall" }),
      series: [
        ...iso,
        ...ps.map((p) => {
          const pts = post.filter((r) => r.persona === p).sort((a, b) => a.round - b.round);
          return {
            name: PERSONA_META[p].label,
            type: "line" as const,
            data: pts.map((r) => [r.precision, r.recall, r.round, r.support ?? 0]),
            symbol: PERSONA_META[p].symbol,
            symbolSize: (v: number[]) => 8 + 14 * Math.sqrt((v[3] ?? 0) / maxSup),
            lineStyle: { color: PERSONA_META[p].color, width: 1, opacity: 0.4 },
            itemStyle: { color: PERSONA_META[p].color, borderColor: INK.surface, borderWidth: 2 },
            label: { show: true, formatter: (q3: { value: unknown }) => `R${(q3.value as number[])[2]}`, position: "top", color: INK.muted, fontSize: 9.5 },
            labelLayout: { hideOverlap: true },
          };
        }),
      ],
    });
  }, [post]);

  const thresholds = useMemo((): EChartsOption => {
    const ps = PERSONAS;
    return lineOption(
      ps.map((p) => ({
        name: PERSONA_META[p].label,
        type: "line",
        step: "middle",
        data: rounds.map((r) => d?.thresholds[String(r)]?.[p] ?? null),
        symbol: PERSONA_META[p].symbol,
        symbolSize: 7,
        lineStyle: { width: 1.8, color: PERSONA_META[p].color },
        itemStyle: { color: PERSONA_META[p].color, borderColor: INK.surface, borderWidth: 2 },
      })),
      rounds,
    );
  }, [d, rounds]);

  if (q.error) return <ErrorNote error={q.error} />;
  if (!d)
    return (
      <div className="space-y-4">
        <Skeleton className="h-12 w-80" />
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
        <Skeleton className="h-96" />
      </div>
    );

  const metricCols = [
    { key: "round", label: "Round", align: "right" as const },
    { key: "persona", label: "Persona", format: (v: unknown) => <span className="flex items-center gap-1.5"><PersonaMark persona={String(v)} />{PERSONA_META[String(v)]?.label}</span> },
    { key: "phase", label: "Detector", format: (v: unknown) => (v === "pre" ? "Frozen" : "Retrained") },
    ...(["precision", "recall", "f1", "family_f1", "fpr", "caught_coverage", "family_coverage", "caught_weak_share", "support"] as const).map((k) => ({
      key: k,
      label:
        k === "caught_coverage" ? "Any-alert cov." : k === "family_coverage" ? "Family cov." : k === "family_f1" ? "Fake-depth F1" : k === "caught_weak_share" ? "Barely caught" : k === "fpr" ? "FPR" : k[0].toUpperCase() + k.slice(1),
      align: "right" as const,
      format: (v: unknown) => (typeof v === "number" ? (k === "support" ? String(v) : v.toFixed(k === "fpr" ? 4 : 3)) : "-"),
    })),
  ];

  const paramRows = (params.data?.rows ?? []).filter((r) => paramPersonas.includes(r.persona));
  const paramRounds = params.data?.rounds ?? [];

  return (
    <div>
      <PageHeader
        eyebrow="Model Lab"
        title="Detection performance"
        right={<ScopeSwitch />}
        description={`Per-persona XGBoost classifiers with an IsolationForest anomaly layer, retrained every round on all data so far. Scope: ${scope}.`}
      />

      <StatGrid className="grid-cols-2 lg:grid-cols-5">
        <StatTile label="Rounds" value={rounds.length} format={(v) => String(Math.round(v))} />
        <StatTile label={`Mean F1 · R${latest}`} value={mean(last.map((r) => r.f1))} format={(v) => v.toFixed(3)} tone="accent" />
        <StatTile label="Mean precision" value={mean(last.map((r) => r.precision))} format={(v) => v.toFixed(3)} />
        <StatTile label="Mean recall" value={mean(last.map((r) => r.recall))} format={(v) => v.toFixed(3)} />
        <StatTile label="Mean FPR" value={mean(last.map((r) => r.fpr))} format={(v) => `${(v * 100).toFixed(2)}%`} />
      </StatGrid>

      <div className="mt-4 space-y-3">
        <ScopeNote frozenByMarket={d.frozen_by_market} />
        {d.frozen_by_market !== false && <Callout kind="accent" title="Evasion and recovery">{d.headline}</Callout>}
      </div>

      <div className="mt-4">
        <ChartCard
          title="Frozen versus retrained"
          subtitle={
            hasFrozen
              ? "Dashed, hollow markers: this round's mutated personas against the detector frozen at the end of the previous round (how well the mutation evaded). Solid: after retraining on every round so far (recovery)."
              : `Solid: ${scope}, after retraining. Dashed: frozen detector, pooled across both markets because this run did not store it per market.`
          }
          actions={<Segmented size="sm" value={prePost} onChange={setPrePost} options={[{ value: "recall", label: "Recall" }, { value: "f1", label: "F1" }]} />}
          table={{ columns: metricCols.slice(0, 6), rows: rows as unknown as Record<string, unknown>[] }}
        >
          <EChart option={prePostOpt} height={360} />
        </ChartCard>
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-[1.4fr_1fr]">
        <ChartCard title="F1 by persona and round" subtitle="Retrained detector. Darker = weaker.">
          <EChart option={heat} height={260} />
        </ChartCard>
        <ChartCard title="Baseline against latest" subtitle="Mean over personas. The FPR axis is scaled x10 so small changes show.">
          <EChart option={radar} height={260} />
        </ChartCard>
      </div>

      <div className="mt-4">
        <ChartCard
          title="Metric evolution"
          subtitle="Retrained detector across rounds, one line per persona. Reference lines at 0.70 and 0.90."
          actions={<Segmented size="sm" value={evo} onChange={setEvo} options={(Object.keys(METRIC_LABEL) as Metric[]).map((m) => ({ value: m, label: m === "fpr" ? "FPR" : METRIC_LABEL[m] }))} />}
        >
          <EChart option={evoOpt} height={330} />
        </ChartCard>
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-2">
        <ChartCard title="Precision-recall trade-off" subtitle="Each mark is a persona in one round; size follows the number of positive windows. Faint curves are iso-F1 lines.">
          <EChart option={pr} height={360} />
        </ChartCard>
        <ChartCard title="Decision thresholds" subtitle="Per-persona probability threshold tuned on out-of-fold predictions at each retrain (smoothed plateau centre, sticky for at most two retrains).">
          <EChart option={thresholds} height={360} />
        </ChartCard>
      </div>

      <SectionTitle>Full metrics</SectionTitle>
      <DataTable spec={{ columns: metricCols, rows: rows as unknown as Record<string, unknown>[] }} maxHeight={420} dense />

      <SectionTitle
        right={
          <div className="flex flex-wrap gap-1.5">
            {PERSONAS.map((p) => (
              <Chip key={p} active={paramPersonas.includes(p)} onClick={() => setParamPersonas((s) => (s.includes(p) ? s.filter((x) => x !== p) : [...s, p]))}>
                <PersonaMark persona={p} /> {PERSONA_META[p].label}
              </Chip>
            ))}
          </div>
        }
      >
        Parameter evolution
      </SectionTitle>
      <p className="-mt-1 mb-3 text-[12.5px] text-fg-muted">Parameters each persona actually ran with in each round. Highlighted cells changed from the round before.</p>
      {params.isLoading ? (
        <Skeleton className="h-64" />
      ) : !paramRows.length ? (
        <div className="panel p-6 text-center text-[12.5px] text-fg-muted">This run has no recorded parameters.</div>
      ) : (
        <div className="panel overflow-auto p-0" style={{ maxHeight: 520 }}>
          <table className="w-full border-collapse text-[12.5px]">
            <thead className="sticky top-0 z-10 bg-ink-800/95 backdrop-blur">
              <tr>
                <th className="border-b border-ink-600 px-3 py-2 text-left font-semibold text-fg-2">Persona</th>
                <th className="border-b border-ink-600 px-3 py-2 text-left font-semibold text-fg-2">Knob</th>
                {paramRounds.map((r) => (
                  <th key={r} className="border-b border-ink-600 px-3 py-2 text-right font-mono font-semibold text-fg-2">R{r}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {paramRows.map((r) => (
                <tr key={`${r.persona}-${r.parameter}`} className="border-b border-ink-700/70 hover:bg-tint/[0.02]">
                  <td className="whitespace-nowrap px-3 py-1.5 text-fg-2"><span className="flex items-center gap-1.5"><PersonaMark persona={r.persona} />{PERSONA_META[r.persona]?.label}</span></td>
                  <td className="whitespace-nowrap px-3 py-1.5 font-mono text-[11.5px] text-fg">{r.parameter}</td>
                  {paramRounds.map((rd) => {
                    const changed = r.changed.includes(String(rd));
                    return (
                      <td key={rd} className={clsx("num whitespace-nowrap px-3 py-1.5 text-right font-mono text-[11.5px]", changed ? "bg-accent/[0.08] font-semibold text-accent" : "text-fg-2")}>
                        {fmtVal(r.values[String(rd)])}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
