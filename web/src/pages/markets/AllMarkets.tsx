import { motion } from "framer-motion";
import { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { EChart } from "@/components/charts/EChart";
import { axisCommon, base, grid, hexA, legend, tooltip, ttRow, ttTitle, valueAxis, MONO } from "@/components/charts/base";
import { ChartCard, Chip, PersonaMark, StatGrid, StatTile, fadeUp, stagger } from "@/components/ui";
import type { Overview } from "@/lib/api";
import { fmtCompact, fmtInt, fmtNum, fmtPrice, fmtSignedPct, tickClock } from "@/lib/format";
import { COMMODITY_META, INK, PERSONAS, PERSONA_META, DIVERGING, personaColor, sortCommodities } from "@/lib/theme";

function Spark({ values, color }: { values: number[]; color: string }) {
  if (values.length < 2) return null;
  const w = 120;
  const h = 34;
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;
  const pts = values.map((v, i) => [(i / (values.length - 1)) * w, h - 2 - ((v - lo) / span) * (h - 4)]);
  const d = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="h-[34px] w-full" preserveAspectRatio="none" aria-hidden>
      <defs>
        <linearGradient id={`g-${color}`} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor={color} stopOpacity={0.28} />
          <stop offset="1" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
      <path d={`${d} L${w},${h} L0,${h} Z`} fill={`url(#g-${color})`} />
      <motion.path d={d} fill="none" stroke={color} strokeWidth={1.6} initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.1, ease: "easeOut" }} />
    </svg>
  );
}

export function AllMarkets({ data, onPick }: { data: Overview; onPick: (c: string) => void }) {
  // Off by default: with every book's episodes on one timeline the shading covers most of the session.
  const [showEpisodes, setShowEpisodes] = useState(false);
  const commodities = sortCommodities(data.commodities);
  const ticks = data.indexed?.ticks ?? [];
  const events = data.events ?? [];

  const indexed: EChartsOption = useMemo(
    () =>
      base({
        grid: grid({ top: 44, bottom: 30 }),
        legend: legend({ data: commodities.map((c) => COMMODITY_META[c]?.label ?? c), type: "scroll", left: 0, right: 0 }),
        tooltip: tooltip({
          formatter: (ps: unknown) => {
            const arr = (ps as { axisValue: number; seriesName: string; value: [number, number]; color: string }[]).filter((p) => p.value);
            arr.sort((a, b) => b.value[1] - a.value[1]);
            return ttTitle(tickClock(arr[0]?.axisValue ?? 0)) + arr.map((p) => ttRow(p.color, p.seriesName, `${fmtNum(p.value[1], 2)}  ${fmtSignedPct(p.value[1] - 100)}`)).join("");
          },
        }),
        dataZoom: [{ type: "inside" }],
        xAxis: { type: "value", min: "dataMin", max: "dataMax", ...axisCommon, splitLine: { show: false }, axisLabel: { ...axisCommon.axisLabel, formatter: (v: number) => tickClock(v) } },
        yAxis: valueAxis({ scale: true, axisLabel: { ...axisCommon.axisLabel, formatter: (v: number) => v.toFixed(1) } }),
        series: commodities.map((c, i) => ({
          name: COMMODITY_META[c]?.label ?? c,
          type: "line",
          showSymbol: false,
          smooth: 0.2,
          data: (data.indexed?.series[c] ?? []).map((v, j) => [ticks[j], v]),
          lineStyle: { width: 1.8, color: COMMODITY_META[c]?.color },
          itemStyle: { color: COMMODITY_META[c]?.color },
          emphasis: { focus: "series", lineStyle: { width: 2.6 } },
          ...(i === 0
            ? {
                markLine: { silent: true, symbol: "none", data: [{ yAxis: 100 }], lineStyle: { color: INK.ref, type: "solid", width: 1 }, label: { show: true, formatter: "Base 100", color: INK.muted, fontSize: 10, position: "insideEndTop" } },
                markArea: showEpisodes
                  ? { silent: true, data: events.map((e) => [{ xAxis: e.start_tick, itemStyle: { color: hexA(personaColor(e.persona), 0.07) } }, { xAxis: e.end_tick }]) }
                  : undefined,
              }
            : {}),
        })),
      }),
    [data, commodities, ticks, events, showEpisodes],
  );

  const multiples = useMemo(() => {
    const cols = 4;
    const rows = Math.ceil(commodities.length / cols);
    const pticks = data.prices?.ticks ?? [];
    const g = commodities.map((_, i) => ({
      left: `${(i % cols) * 25 + 1.5}%`,
      width: "22%",
      containLabel: true,
      top: `${Math.floor(i / cols) * (100 / rows) + 9}%`,
      height: `${100 / rows - 17}%`,
    }));
    return base({
      grid: g,
      title: commodities.map((c, i) => ({
        text: COMMODITY_META[c]?.label ?? c,
        left: `${(i % cols) * 25 + 1.5}%`,
        top: `${Math.floor(i / cols) * (100 / rows) + 1}%`,
        textStyle: { color: INK.secondary, fontSize: 11.5, fontWeight: 500, fontFamily: '"IBM Plex Sans"' },
      })),
      tooltip: tooltip({
        trigger: "axis",
        formatter: (ps: unknown) => {
          const p = (ps as { seriesName: string; value: [number, number]; color: string }[])[0];
          return p ? ttTitle(tickClock(p.value[0])) + ttRow(p.color, p.seriesName, fmtPrice(p.value[1])) : "";
        },
      }),
      xAxis: commodities.map((_, i) => ({ type: "value", gridIndex: i, min: "dataMin", max: "dataMax", show: false })),
      yAxis: commodities.map((_, i) => ({ type: "value", gridIndex: i, scale: true, ...axisCommon, axisLabel: { ...axisCommon.axisLabel, fontSize: 9 }, splitNumber: 2 })),
      series: commodities.map((c, i) => ({
        name: COMMODITY_META[c]?.label ?? c,
        type: "line",
        xAxisIndex: i,
        yAxisIndex: i,
        showSymbol: false,
        data: (data.prices?.series[c] ?? []).map((v, j) => [pticks[j], v]),
        lineStyle: { width: 1.5, color: COMMODITY_META[c]?.color },
        itemStyle: { color: COMMODITY_META[c]?.color },
        markArea: {
          silent: true,
          data: events.filter((e) => e.commodity === c).map((e) => [{ xAxis: e.start_tick, itemStyle: { color: hexA(personaColor(e.persona), 0.16) } }, { xAxis: e.end_tick }]),
        },
      })),
    }) as EChartsOption;
  }, [data, commodities, events]);

  const corr: EChartsOption = useMemo(() => {
    const labels = data.correlation?.labels ?? [];
    const order = sortCommodities(labels);
    const idx = order.map((c) => labels.indexOf(c));
    const cells: [number, number, number][] = [];
    idx.forEach((ri, y) => idx.forEach((ci, x) => cells.push([x, y, data.correlation!.matrix[ri][ci]])));
    const short = order.map((c) => COMMODITY_META[c]?.ticker ?? c);
    return base({
      grid: grid({ top: 10, right: 70, bottom: 10 }),
      tooltip: { ...tooltip({ trigger: "item" }), formatter: (p: unknown) => {
        const v = (p as { value: [number, number, number] }).value;
        return ttTitle("Return correlation") + ttRow(INK.muted, `${COMMODITY_META[order[v[1]]]?.label} vs ${COMMODITY_META[order[v[0]]]?.label}`, fmtNum(v[2], 3));
      } },
      xAxis: { type: "category", data: short, ...axisCommon, splitLine: { show: false }, axisLine: { show: false } },
      yAxis: { type: "category", data: short, ...axisCommon, splitLine: { show: false }, axisLine: { show: false }, inverse: true },
      visualMap: { min: -1, max: 1, calculable: false, orient: "vertical", right: 0, top: "middle", itemHeight: 140, itemWidth: 10, inRange: { color: DIVERGING }, textStyle: { color: INK.muted, fontFamily: MONO, fontSize: 10 }, text: ["+1", "-1"] },
      series: [{ type: "heatmap", data: cells, itemStyle: { borderColor: INK.surface, borderWidth: 2, borderRadius: 3 }, label: { show: true, color: INK.primary, fontSize: 10, fontFamily: MONO, formatter: (p: { value: unknown }) => fmtNum(Math.abs((p.value as number[])[2]) < 0.005 ? 0 : (p.value as number[])[2], 2) }, emphasis: { itemStyle: { borderColor: INK.primary } } }],
    });
  }, [data]);

  const volume: EChartsOption = useMemo(() => {
    const rows = sortCommodities(commodities).map((c) => data.volume_by_commodity?.find((v) => v.commodity === c));
    return base({
      grid: grid({ top: 10, left: 4, right: 52 }),
      tooltip: tooltip({ trigger: "item", formatter: (p: unknown) => {
        const q = p as { name: string; value: number; color: string; dataIndex: number };
        return ttRow(q.color, q.name, fmtInt(q.value)) + ttRow(INK.muted, "Trades", fmtInt(rows[q.dataIndex]?.trades));
      } }),
      xAxis: valueAxis({ axisLabel: { ...axisCommon.axisLabel, formatter: (v: number) => fmtCompact(v) } }),
      yAxis: { type: "category", data: commodities.map((c) => COMMODITY_META[c]?.label ?? c), inverse: true, ...axisCommon, splitLine: { show: false }, axisLabel: { color: INK.secondary, fontSize: 11.5 } },
      series: [{
        type: "bar",
        barWidth: 12,
        data: rows.map((r, i) => ({ value: r?.volume ?? 0, itemStyle: { color: COMMODITY_META[commodities[i]]?.color, borderRadius: [0, 4, 4, 0] } })),
        label: { show: true, position: "right", color: INK.secondary, fontFamily: MONO, fontSize: 10.5, formatter: (p: { value: unknown }) => fmtCompact(p.value as number) },
      }],
    });
  }, [data, commodities]);

  const manip: EChartsOption = useMemo(() => {
    const rows = commodities.map((c) => data.events_by_commodity?.find((r) => r.commodity === c));
    const present = PERSONAS.filter((p) => rows.some((r) => Number(r?.[p] ?? 0) > 0));
    return base({
      grid: grid({ top: 34 }),
      legend: legend({ data: present.map((p) => PERSONA_META[p].label) }),
      tooltip: tooltip({ axisPointer: { type: "shadow", shadowStyle: { color: INK.shade } } }),
      xAxis: { type: "category", data: commodities.map((c) => COMMODITY_META[c]?.ticker ?? c), ...axisCommon, splitLine: { show: false } },
      yAxis: valueAxis({ minInterval: 1, name: "Episodes" }),
      series: present.map((p, i) => ({
        name: PERSONA_META[p].label,
        type: "bar",
        stack: "e",
        barWidth: 22,
        data: rows.map((r) => Number(r?.[p] ?? 0)),
        itemStyle: { color: PERSONA_META[p].color, borderColor: INK.surface, borderWidth: 2, borderRadius: i === present.length - 1 ? [4, 4, 0, 0] : 0 },
      })),
    });
  }, [data, commodities]);

  return (
    <div>
      <StatGrid className="grid-cols-2 lg:grid-cols-4">
        <StatTile label="Trades" value={data.kpi.trades} format={fmtInt} sub="All eight books" />
        <StatTile label="Volume" value={data.kpi.volume} format={fmtInt} sub="Contracts traded" />
        <StatTile label="Active accounts" value={data.kpi.traders} format={fmtInt} sub="Buyers and sellers" />
        <StatTile label="Ground-truth episodes" value={data.kpi.events} format={fmtInt} tone={data.kpi.events ? "warn" : "good"} sub="Injected by the simulator" />
      </StatGrid>

      <motion.div variants={stagger(0.04)} initial="hidden" animate="show" className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {sortCommodities(data.snapshot.map((s) => s.commodity)).map((c) => {
          const s = data.snapshot.find((x) => x.commodity === c)!;
          const meta = COMMODITY_META[c];
          const up = s.pct >= 0;
          return (
            <motion.button
              key={c}
              variants={fadeUp}
              onClick={() => onPick(c)}
              whileHover={{ y: -2 }}
              className="focus-ring panel group p-4 text-left transition-colors hover:border-ink-600"
            >
              <div className="flex items-start justify-between gap-2">
                <div>
                  <div className="flex items-center gap-2 text-[13px] font-semibold text-fg">
                    <span className="h-2 w-2 rounded-full" style={{ background: meta?.color }} />
                    {meta?.label}
                  </div>
                  <div className="mt-0.5 font-mono text-[10.5px] text-fg-muted">
                    {meta?.ticker}=F · {meta?.sector}
                  </div>
                </div>
                {s.events > 0 ? (
                  <span className="rounded-md border border-warn/30 bg-warn/10 px-1.5 py-0.5 text-[10.5px] font-medium text-warn">{s.events} episodes</span>
                ) : (
                  <span className="rounded-md border border-good/30 bg-good/10 px-1.5 py-0.5 text-[10.5px] font-medium text-good">Clean</span>
                )}
              </div>
              <div className="mt-3 flex items-baseline justify-between">
                <span className="text-[20px] font-semibold tracking-tight text-fg">{fmtPrice(s.last)}</span>
                <span className={up ? "font-mono text-[12px] text-good" : "font-mono text-[12px] text-critical"}>
                  {up ? "▲" : "▼"} {fmtSignedPct(s.pct)}
                </span>
              </div>
              <div className="mt-2">
                <Spark values={s.spark} color={meta?.color ?? INK.muted} />
              </div>
              <div className="mt-2 flex justify-between border-t border-ink-700 pt-2 font-mono text-[10.5px] text-fg-muted">
                <span>Vol {fmtCompact(s.volume)}</span>
                <span>{fmtCompact(s.trades)} trades</span>
              </div>
            </motion.button>
          );
        })}
      </motion.div>

      <div className="mt-4">
        <ChartCard
          title="Indexed price paths"
          subtitle="Every contract rebased to 100 at the session open, so moves are comparable across very different price levels. Scroll to zoom."
          actions={
            <Chip active={showEpisodes} onClick={() => setShowEpisodes((s) => !s)} title="Shade ground-truth manipulation spans">
              Episodes
            </Chip>
          }
          table={{
            columns: [
              { key: "label", label: "Commodity" },
              { key: "first", label: "Open", align: "right", format: (v) => fmtPrice(v as number) },
              { key: "last", label: "Close", align: "right", format: (v) => fmtPrice(v as number) },
              { key: "pct", label: "Change", align: "right", format: (v) => fmtSignedPct(v as number) },
              { key: "events", label: "Episodes", align: "right" },
            ],
            rows: commodities.map((c) => {
              const s = data.snapshot.find((x) => x.commodity === c)!;
              return { label: COMMODITY_META[c]?.label ?? c, ...s };
            }),
          }}
        >
          <EChart option={indexed} height={380} />
        </ChartCard>
      </div>

      <div className="mt-4">
        <ChartCard title="Small multiples" subtitle="Each book on its own price scale. Shading marks manipulation spans, coloured by persona.">
          <EChart option={multiples} height={420} />
          <div className="mt-2 flex flex-wrap gap-3">
            {PERSONAS.map((p) => (
              <span key={p} className="flex items-center gap-1.5 text-[11.5px] text-fg-2">
                <PersonaMark persona={p} /> {PERSONA_META[p].label}
              </span>
            ))}
          </div>
        </ChartCard>
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-2">
        <ChartCard title="Return correlation" subtitle="Pearson correlation of bucketed returns. Blue = move together, red = move apart, grey = unrelated.">
          <EChart option={corr} height={340} />
        </ChartCard>
        <ChartCard
          title="Volume by book"
          subtitle="Contracts traded per commodity in this test market."
          table={{
            columns: [
              { key: "label", label: "Commodity" },
              { key: "volume", label: "Volume", align: "right", format: (v) => fmtInt(v as number) },
              { key: "trades", label: "Trades", align: "right", format: (v) => fmtInt(v as number) },
            ],
            rows: (data.volume_by_commodity ?? []).map((v) => ({ label: COMMODITY_META[v.commodity]?.label ?? v.commodity, ...v })),
          }}
        >
          <EChart option={volume} height={340} />
        </ChartCard>
      </div>

      <div className="mt-4">
        <ChartCard
          title="Where the manipulation happened"
          subtitle="Ground-truth episodes per commodity, split by persona."
          table={{
            columns: [{ key: "label", label: "Commodity" }, ...PERSONAS.map((p) => ({ key: p, label: PERSONA_META[p].label, align: "right" as const }))],
            rows: (data.events_by_commodity ?? []).map((r) => ({ ...r, label: COMMODITY_META[String(r.commodity)]?.label ?? r.commodity })),
          }}
        >
          <EChart option={manip} height={300} />
        </ChartCard>
      </div>
    </div>
  );
}
