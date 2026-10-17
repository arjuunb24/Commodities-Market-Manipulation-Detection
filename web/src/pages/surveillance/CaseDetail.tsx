import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import { BadgeCheck, Database, FileText, Loader2, Sparkles, TriangleAlert } from "lucide-react";
import { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { EChart } from "@/components/charts/EChart";
import { axisCommon, base, grid, legend, tooltip, ttRow, ttTitle, valueAxis, hexA } from "@/components/charts/base";
import { DataTable, ErrorNote, Gauge, PersonaBadge, PersonaMark, Segmented, Skeleton, StatusBadge } from "@/components/ui";
import { useAiStatus, useCaseDetail, useCaseRaw, useNarrative, type CaseDetail as Detail } from "@/lib/api";
import { fmtInt, fmtNum, fmtPct, fmtPrice, fmtVal, tickClock } from "@/lib/format";
import { FEATURES, STATUS_TEXT, confidenceLabel, evidenceLabel, featureLabel } from "@/lib/glossary";
import { useSelection } from "@/lib/selection";
import { CAT, COMMODITY_META, INK, MARKS, PERSONA_META } from "@/lib/theme";


function Confidence({ d }: { d: Detail }) {
  const anomaly = d.persona === "anomaly";
  const value = anomaly ? d.iso_score : d.persona_scores?.[d.persona] ?? d.score;
  const thr = anomaly ? d.thresholds.anomaly : d.thresholds[d.persona];
  const margin = thr !== undefined ? value - thr : null;
  const color = PERSONA_META[d.persona]?.color ?? INK.muted;
  const others = ["spoofing", "layering", "wash_trading", "pump_and_dump"].filter((p) => d.persona_scores && p in d.persona_scores);
  return (
    <div className="rounded-xl border border-ink-700 bg-ink-800/50 p-4">
      <div className="flex items-center justify-between">
        <span className="eyebrow">Prediction confidence</span>
        {margin !== null && <span className="text-[11.5px] font-medium text-fg-2">{confidenceLabel(margin)}</span>}
      </div>
      <div className="mt-2 flex items-baseline gap-2">
        <span className="text-[30px] font-semibold leading-none tracking-tight text-fg">{fmtPct(value, 1)}</span>
        <span className="text-[12px] text-fg-muted">{anomaly ? "IsolationForest percentile" : `${PERSONA_META[d.persona]?.label.toLowerCase()} probability`}</span>
      </div>
      <div className="mt-3">
        <Gauge value={value} threshold={thr} color={color} height={9} />
      </div>
      <div className="mt-2 font-mono text-[11px] text-fg-muted">
        {thr !== undefined ? `threshold ${thr.toFixed(2)} · margin ${margin! >= 0 ? "+" : ""}${margin!.toFixed(2)}` : "threshold not recorded"} · tick = decision threshold
      </div>
      {others.length > 0 && (
        <div className="mt-4 space-y-2.5 border-t border-ink-700 pt-3">
          <div className="eyebrow">Every persona classifier on this account-window</div>
          {others.map((p) => {
            const pv = d.persona_scores[p];
            const pt = d.thresholds[p];
            const over = pt !== undefined && pv >= pt;
            return (
              <div key={p}>
                <div className="flex items-center justify-between text-[12px]">
                  <span className="flex items-center gap-1.5 text-fg-2">
                    <PersonaMark persona={p} /> {PERSONA_META[p].label}
                    {over && <span className="rounded bg-tint/5 px-1 text-[10px] text-fg">over threshold</span>}
                  </span>
                  <span className="num font-mono text-fg-2">
                    {fmtPct(pv, 1)} <span className="text-fg-muted">/ {pt?.toFixed(2) ?? "-"}</span>
                  </span>
                </div>
                <div className="mt-1">
                  <Gauge value={pv} threshold={pt} color={PERSONA_META[p].color} height={5} />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function priceChart(d: Detail): EChartsOption {
  const pts = d.price.ticks.map((t, i) => [t, d.price.prices[i]]);
  return base({
    grid: grid({ top: 36, bottom: 6 }),
    legend: legend({ data: ["Market price", "Account buys", "Account sells"] }),
    tooltip: tooltip({
      trigger: "axis",
      formatter: (ps: unknown) => {
        const arr = ps as { seriesName: string; value: number[]; color: string }[];
        const t = arr[0]?.value?.[0];
        return ttTitle(tickClock(t ?? 0)) + arr.map((p) => ttRow(p.color, p.seriesName, p.value[2] ? `${fmtPrice(p.value[1])} x ${p.value[2]}` : fmtPrice(p.value[1]))).join("");
      },
    }),
    xAxis: valueAxis({ min: "dataMin", max: "dataMax", splitLine: { show: false }, axisLabel: { ...axisCommon.axisLabel, formatter: (v: number) => tickClock(v) } }),
    yAxis: valueAxis({ scale: true, axisLabel: { ...axisCommon.axisLabel, formatter: (v: number) => fmtPrice(v) } }),
    series: [
      {
        name: "Market price",
        type: "line",
        step: "end",
        showSymbol: false,
        data: pts,
        lineStyle: { width: 1.4, color: MARKS.price },
        itemStyle: { color: MARKS.price },
        markArea: {
          silent: true,
          itemStyle: { color: hexA(INK.accent, 0.07), borderColor: hexA(INK.accent, 0.35), borderWidth: 1 },
          label: { show: true, color: INK.muted, fontSize: 10, position: "insideTopLeft", formatter: "flagged span" },
          data: [[{ xAxis: d.start_tick }, { xAxis: d.end_tick }]],
        },
      },
      {
        name: "Account buys",
        type: "scatter",
        symbol: "triangle",
        symbolSize: 9,
        data: d.buys.map((b) => [b.tick, b.price, b.qty]),
        itemStyle: { color: MARKS.up, borderColor: INK.surface, borderWidth: 1.5 },
      },
      {
        name: "Account sells",
        type: "scatter",
        symbol: "triangle",
        symbolRotate: 180,
        symbolSize: 9,
        data: d.sells.map((b) => [b.tick, b.price, b.qty]),
        itemStyle: { color: MARKS.down, borderColor: INK.surface, borderWidth: 1.5 },
      },
    ],
  });
}

function shapChart(d: Detail): EChartsOption {
  const items = [...d.shap].slice(0, 12).reverse();
  return base({
    grid: grid({ top: 6, left: 4, right: 56, bottom: 4 }),
    tooltip: tooltip({
      trigger: "item",
      formatter: (p: unknown) => {
        const q = p as { dataIndex: number };
        const it = items[q.dataIndex];
        return ttTitle(featureLabel(it.feature)) + ttRow(it.value > 0 ? CAT[7] : CAT[0], "SHAP", (it.value > 0 ? "+" : "") + it.value.toFixed(4)) +
          `<div style="max-width:260px;white-space:normal;color:${INK.muted};margin-top:6px;font-size:11px">${FEATURES[it.feature] ?? ""}</div>`;
      },
    }),
    xAxis: valueAxis({ axisLabel: { ...axisCommon.axisLabel, formatter: (v: number) => v.toFixed(1) } }),
    yAxis: { type: "category", data: items.map((i) => featureLabel(i.feature)), ...axisCommon, splitLine: { show: false }, axisLabel: { color: INK.secondary, fontSize: 11 } },
    series: [
      {
        type: "bar",
        barWidth: 10,
        data: items.map((i) => ({ value: i.value, itemStyle: { color: i.value > 0 ? CAT[7] : CAT[0], borderRadius: i.value > 0 ? [0, 3, 3, 0] : [3, 0, 0, 3] } })),
        label: { show: true, position: "right", color: INK.secondary, fontFamily: '"IBM Plex Mono"', fontSize: 10, formatter: (p: { value: unknown }) => ((p.value as number) > 0 ? "+" : "") + (p.value as number).toFixed(2) },
        markLine: { silent: true, symbol: "none", data: [{ xAxis: 0 }], lineStyle: { color: INK.ref, type: "solid" }, label: { show: false } },
      },
    ],
  });
}

function Narrative({ id }: { id: string }) {
  const { run, round, market } = useSelection();
  const ai = useAiStatus();
  const m = useNarrative(run, round, market, id);
  const available = ai.data?.narrative_generator.available;
  const res = m.data;
  return (
    <div className="rounded-xl border border-ink-700 bg-ink-800/40 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="flex items-center gap-1.5 text-[13px] font-semibold text-fg">
            <Sparkles size={14} className="text-accent" /> Case note
          </div>
          <div className="mt-0.5 text-[11.5px] text-fg-muted">AI explanation of the mechanism, fact-checked against the detector output. Simulation only.</div>
        </div>
        <button
          disabled={!available || m.isPending}
          onClick={() => m.mutate()}
          className="focus-ring inline-flex h-8 items-center gap-1.5 rounded-lg bg-accent px-3 text-[12.5px] font-semibold text-accent-ink transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
          title={available ? undefined : ai.data?.narrative_generator.reason ?? "Checking LLM availability"}
        >
          {m.isPending ? <Loader2 size={14} className="animate-spin" /> : <FileText size={14} />}
          {m.isPending ? "Writing and checking" : res ? "Regenerate" : "Generate note"}
        </button>
      </div>
      <AnimatePresence mode="wait">
        {m.isPending && (
          <motion.div key="wait" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="mt-3 space-y-2">
            <Skeleton className="h-3 w-full" />
            <Skeleton className="h-3 w-11/12" />
            <Skeleton className="h-3 w-4/5" />
          </motion.div>
        )}
        {!m.isPending && (res || m.error) && (
          <motion.div key="res" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="mt-3">
            {m.error ? (
              <ErrorNote error={m.error} />
            ) : res?.error ? (
              <div className="text-[12.5px] text-fg-muted">AI features are unavailable: {res.error} The SHAP chart is the evidence for this case.</div>
            ) : res?.grounding_check_passed ? (
              <>
                <StatusBadge kind="good">Verified: every number appears in the detector output</StatusBadge>
                <p className="mt-2.5 text-[13px] leading-relaxed text-fg-2">{res.narrative_text}</p>
              </>
            ) : (
              <div className="flex gap-2 text-[12.5px] text-fg-muted">
                <TriangleAlert size={14} className="mt-0.5 shrink-0 text-warn" />
                The generated note mentioned numbers that are not in the detector output, so it was discarded. The SHAP chart is the evidence for this case.
              </div>
            )}
          </motion.div>
        )}
        {!available && ai.data && !res && (
          <div className="mt-3 text-[12px] text-fg-muted">LLM unavailable: {ai.data.narrative_generator.reason}</div>
        )}
      </AnimatePresence>
    </div>
  );
}

function RawData({ id }: { id: string }) {
  const { run, round, market } = useSelection();
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState<"orders" | "trades">("orders");
  const q = useCaseRaw(run, round, market, id, open);
  const cols = (rows: Record<string, unknown>[]) =>
    rows.length
      ? Object.keys(rows[0]).map((k) => ({
          key: k,
          label: k.replace(/_/g, " "),
          align: typeof rows[0][k] === "number" ? ("right" as const) : undefined,
          format: (v: unknown) => (k === "tick" ? `${fmtInt(v as number)}  ${tickClock(v as number)}` : fmtVal(v)),
        }))
      : [];
  return (
    <div className="rounded-xl border border-ink-700">
      <button onClick={() => setOpen((o) => !o)} className="focus-ring flex w-full items-center justify-between px-4 py-3 text-left">
        <span className="flex items-center gap-2 text-[13px] font-semibold text-fg">
          <Database size={14} className="text-fg-muted" /> Raw orders and trades in the span
        </span>
        <span className="text-[11.5px] text-fg-muted">{open ? "Hide" : "Load"}</span>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
            <div className="border-t border-ink-700 p-4">
              {q.isLoading ? (
                <Skeleton className="h-40" />
              ) : q.error ? (
                <ErrorNote error={q.error} />
              ) : q.data ? (
                <>
                  <Segmented
                    size="sm"
                    value={tab}
                    onChange={setTab}
                    options={[
                      { value: "orders", label: `Orders ${fmtInt(q.data.orders_total)}` },
                      { value: "trades", label: `Trades ${fmtInt(q.data.trades_total)}` },
                    ]}
                  />
                  <div className="mt-3">
                    <DataTable dense maxHeight={340} spec={{ columns: cols(q.data[tab]), rows: q.data[tab] }} empty="Nothing in this span." />
                  </div>
                  {(tab === "orders" ? q.data.orders_total : q.data.trades_total) > 500 && <div className="mt-2 text-[11px] text-fg-muted">Showing the first 500.</div>}
                </>
              ) : null}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export function CaseDetail({ id }: { id: string }) {
  const { run, round, market } = useSelection();
  const q = useCaseDetail(run, round, market, id);
  const d = q.data;
  const price = useMemo(() => (d ? priceChart(d) : null), [d]);
  const shap = useMemo(() => (d && d.shap.length ? shapChart(d) : null), [d]);

  if (q.error) return <ErrorNote error={q.error} />;
  if (!d || d.id !== id)
    return (
      <div className="panel space-y-3 p-5">
        <Skeleton className="h-8 w-2/3" />
        <Skeleton className="h-44" />
        <Skeleton className="h-64" />
      </div>
    );

  const st = STATUS_TEXT[d.status];
  const evidence = Object.entries(d.evidence ?? {});
  return (
    <motion.div key={d.id} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }} className="panel p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="eyebrow">Detector belief</div>
          <div className="mt-1.5 flex flex-wrap items-center gap-2">
            <PersonaBadge persona={d.persona} />
            <span className="font-mono text-[16px] font-semibold text-fg">{d.trader_id}</span>
          </div>
          <div className="mt-1.5 text-[12.5px] text-fg-muted">
            {COMMODITY_META[d.commodity]?.label ?? d.commodity} · {tickClock(d.start_tick)} to {tickClock(d.end_tick)} · {d.n_windows} windows · score{" "}
            <span className="font-mono text-fg-2">{d.score.toFixed(3)}</span>
          </div>
        </div>
        <div className="flex flex-col items-end gap-1.5">
          <StatusBadge kind={st.kind}>{d.status === "other_persona" ? `Ground truth: ${PERSONA_META[d.gt_persona ?? ""]?.label ?? d.gt_persona}` : st.label}</StatusBadge>
          <span className="font-mono text-[10.5px] text-fg-muted">fired: {d.triggered_by.join(", ").replace(/_/g, " ")}</span>
        </div>
      </div>

      <div className="mt-4 grid gap-4 2xl:grid-cols-[300px_1fr]">
        <Confidence d={d} />
        <div className="rounded-xl border border-ink-700 p-3">
          <div className="px-1 text-[12.5px] font-semibold text-fg">Price around the case</div>
          {price && <EChart option={price} height={250} />}
        </div>
      </div>

      <div className="mt-4 grid gap-4 2xl:grid-cols-2">
        <div className="rounded-xl border border-ink-700 p-4">
          <div className="flex items-center justify-between">
            <div className="text-[13px] font-semibold text-fg">Why the model fired</div>
            <div className="flex items-center gap-3 text-[11px] text-fg-muted">
              <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-sm" style={{ background: CAT[7] }} /> toward manipulation</span>
              <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-sm" style={{ background: CAT[0] }} /> away</span>
            </div>
          </div>
          <div className="mt-0.5 text-[11.5px] text-fg-muted">SHAP attribution from the {d.persona === "anomaly" ? "IsolationForest" : `${PERSONA_META[d.persona]?.label.toLowerCase()} classifier`}, strongest window.</div>
          {shap ? <EChart option={shap} height={Math.max(180, Math.min(12, d.shap.length) * 26 + 20)} /> : <div className="py-8 text-center text-[12.5px] text-fg-muted">No SHAP attribution stored for this case.</div>}
        </div>
        <div className="rounded-xl border border-ink-700 p-4">
          <div className="text-[13px] font-semibold text-fg">What the account did</div>
          <div className="mt-0.5 text-[11.5px] text-fg-muted">Measured from its own orders and trades in the flagged span.</div>
          <dl className="mt-3 divide-y divide-ink-700/70">
            {evidence.length ? (
              evidence.map(([k, v]) => (
                <div key={k} className="flex items-start justify-between gap-4 py-2">
                  <dt className="text-[12.5px] text-fg-2">{evidenceLabel(k)}</dt>
                  <dd className="num shrink-0 text-right font-mono text-[12.5px] text-fg">{fmtVal(v)}</dd>
                </div>
              ))
            ) : (
              <div className="py-4 text-[12.5px] text-fg-muted">No evidence summary for this persona.</div>
            )}
          </dl>
        </div>
      </div>

      {d.shap.length > 0 && (
        <details className="group mt-4 rounded-xl border border-ink-700">
          <summary className="focus-ring flex cursor-pointer list-none items-center justify-between px-4 py-3 text-[13px] font-semibold text-fg">
            Feature glossary for this case
            <span className="text-[11.5px] font-normal text-fg-muted group-open:hidden">Show</span>
            <span className="hidden text-[11.5px] font-normal text-fg-muted group-open:inline">Hide</span>
          </summary>
          <div className="border-t border-ink-700 p-4">
            <DataTable
              dense
              maxHeight={320}
              spec={{
                columns: [
                  { key: "feature", label: "Feature", format: (v) => <span className="font-mono text-[11.5px] text-fg">{String(v)}</span> },
                  { key: "value", label: "SHAP", align: "right", format: (v) => `${(v as number) > 0 ? "+" : ""}${fmtNum(v as number, 4)}` },
                  { key: "meaning", label: "Meaning", format: (v) => <span className="whitespace-normal">{String(v)}</span> },
                ],
                rows: d.shap.map((s) => ({ ...s, meaning: FEATURES[s.feature] ?? "-" })),
              }}
            />
          </div>
        </details>
      )}

      <div className="mt-4 space-y-4">
        <Narrative id={d.id} />
        <RawData id={d.id} />
      </div>
      <div className={clsx("mt-3 flex items-center gap-1.5 text-[11px] text-fg-muted")}>
        <BadgeCheck size={12} /> Ground truth comes from the simulator's episode log and is never shown to the detector.
      </div>
    </motion.div>
  );
}
