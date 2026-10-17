import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import { EChart } from "@/components/charts/EChart";
import { axisCommon, base, grid, hexA, legend, tooltip, ttRow, ttTitle, valueAxis, MONO } from "@/components/charts/base";
import { ChartCard, ErrorNote, PersonaBadge, Skeleton, StatGrid, StatTile } from "@/components/ui";
import { useCommodity, type CommodityView } from "@/lib/api";
import { fmtCompact, fmtInt, fmtNum, fmtPrice, fmtSignedPct, tickClock } from "@/lib/format";
import { useSelection } from "@/lib/selection";
import { CAT, COMMODITY_META, INK, MARKS, PERSONA_META, SEQ_BLUE, STATUS, personaColor } from "@/lib/theme";


function episodeAreas(view: CommodityView, toX: (t: number) => number | string, alpha = 0.1) {
  return view.events.map((e) => [
    { xAxis: toX(e.start_tick), itemStyle: { color: hexA(personaColor(e.persona), alpha) }, name: PERSONA_META[e.persona]?.label },
    { xAxis: toX(e.end_tick) },
  ]);
}

export function CommodityTerminal({ commodity }: { commodity: string }) {
  const { run, round, market } = useSelection();
  const q = useCommodity(run, round, market, commodity);
  const v = q.data && q.data.commodity === commodity ? q.data : undefined;
  const meta = COMMODITY_META[commodity];

  const candles: EChartsOption | null = useMemo(() => {
    if (!v || v.empty) return null;
    const bar = v.bucket_ticks.bar;
    const cats = v.candles.map((c) => String(c.tick));
    const snap = (t: number) => String(Math.min(v.candles[v.candles.length - 1].tick, Math.max(v.candles[0].tick, Math.floor(t / bar) * bar)));
    return base({
      animation: true,
      legend: legend({ data: ["Price", "VWAP", "Volume"], right: 0 }),
      axisPointer: { link: [{ xAxisIndex: "all" }], label: { backgroundColor: INK.grid } },
      tooltip: tooltip({
        axisPointer: { type: "cross" },
        formatter: (ps: unknown) => {
          const arr = ps as { seriesName: string; data: unknown; axisValue: string; color: string }[];
          const k = arr.find((p) => p.seriesName === "Price");
          const i = cats.indexOf(arr[0]?.axisValue);
          const c = v.candles[i];
          if (!c) return "";
          const ev = v.events.filter((e) => e.start_tick <= c.tick + bar && e.end_tick >= c.tick);
          return (
            ttTitle(`${tickClock(c.tick)} · ${bar}s bar`) +
            ttRow(k ? (c.close >= c.open ? MARKS.up : MARKS.down) : INK.muted, "Open / Close", `${fmtPrice(c.open)} / ${fmtPrice(c.close)}`) +
            ttRow(INK.muted, "High / Low", `${fmtPrice(c.high)} / ${fmtPrice(c.low)}`) +
            ttRow(INK.secondary, "VWAP", fmtPrice(c.vwap)) +
            ttRow(MARKS.volume, "Volume", fmtInt(c.volume)) +
            ev.map((e) => ttRow(personaColor(e.persona), "Episode", `${PERSONA_META[e.persona]?.label} ${e.trader_ids.length} acct`)).join("")
          );
        },
      }),
      grid: [grid({ top: 34, bottom: 110 }), grid({ top: undefined, height: 64, bottom: 34 })],
      xAxis: [
        { type: "category", data: cats, boundaryGap: true, ...axisCommon, splitLine: { show: false }, axisLabel: { show: false } },
        { type: "category", gridIndex: 1, data: cats, boundaryGap: true, ...axisCommon, splitLine: { show: false }, axisLabel: { ...axisCommon.axisLabel, formatter: (x: string) => tickClock(+x) } },
      ],
      yAxis: [
        valueAxis({ scale: true, position: "right", axisLabel: { ...axisCommon.axisLabel, formatter: (x: number) => fmtPrice(x) } }),
        valueAxis({ gridIndex: 1, position: "right", splitNumber: 2, axisLabel: { ...axisCommon.axisLabel, formatter: (x: number) => fmtCompact(x) } }),
      ],
      dataZoom: [{ type: "inside", xAxisIndex: [0, 1] }, { type: "slider", xAxisIndex: [0, 1], height: 18, bottom: 6, borderColor: INK.axis, backgroundColor: INK.shade, fillerColor: hexA(INK.accent, 0.08), handleStyle: { color: INK.axis, borderColor: INK.accent }, dataBackground: { lineStyle: { color: INK.axis }, areaStyle: { color: INK.shade } }, textStyle: { color: INK.muted, fontFamily: MONO, fontSize: 10 }, labelFormatter: (_: number, s: string) => tickClock(+s) }],
      series: [
        {
          name: "Price",
          type: "candlestick",
          data: v.candles.map((c) => [c.open, c.close, c.low, c.high]),
          itemStyle: { color: MARKS.up, color0: MARKS.down, borderColor: MARKS.up, borderColor0: MARKS.down, borderWidth: 1 },
          barMaxWidth: 9,
          markArea: { silent: true, data: episodeAreas(v, snap, 0.12) as never, label: { show: true, color: INK.muted, fontSize: 10, position: "insideTop" } },
        },
        { name: "VWAP", type: "line", data: v.candles.map((c) => c.vwap), showSymbol: false, smooth: true, lineStyle: { width: 1.6, color: MARKS.vwap, opacity: 0.7 }, itemStyle: { color: MARKS.vwap } },
        {
          name: "Volume",
          type: "bar",
          itemStyle: { color: MARKS.volume },
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: v.candles.map((c) => ({ value: c.volume, itemStyle: { color: c.close >= c.open ? hexA(MARKS.up, 0.55) : hexA(MARKS.down, 0.55), borderRadius: [2, 2, 0, 0] } })),
          barMaxWidth: 9,
        },
      ],
    });
  }, [v]);

  const flow: EChartsOption | null = useMemo(() => {
    if (!v || v.empty) return null;
    const x = v.flow.map((f) => f.tick);
    return base({
      grid: grid({ top: 34 }),
      legend: legend({ data: ["Aggressive buys", "Aggressive sells"] }),
      tooltip: tooltip({
        axisPointer: { type: "shadow", shadowStyle: { color: INK.shade } },
        formatter: (ps: unknown) => {
          const arr = ps as { dataIndex: number }[];
          const f = v.flow[arr[0]?.dataIndex];
          return f ? ttTitle(`${tickClock(f.tick)} · 1 min`) + ttRow(MARKS.up, "Aggressive buys", fmtInt(f.buy)) + ttRow(MARKS.down, "Aggressive sells", fmtInt(f.sell)) + ttRow(INK.muted, "Trades", fmtInt(f.trades)) : "";
        },
      }),
      xAxis: { type: "category", data: x.map(String), ...axisCommon, splitLine: { show: false }, axisLabel: { ...axisCommon.axisLabel, formatter: (t: string) => tickClock(+t) } },
      yAxis: valueAxis({ axisLabel: { ...axisCommon.axisLabel, formatter: (n: number) => fmtCompact(Math.abs(n)) } }),
      series: [
        { name: "Aggressive buys", type: "bar", stack: "f", data: v.flow.map((f) => f.buy), itemStyle: { color: hexA(MARKS.up, 0.75), borderRadius: [3, 3, 0, 0] }, barCategoryGap: "25%" },
        { name: "Aggressive sells", type: "bar", stack: "f", data: v.flow.map((f) => -f.sell), itemStyle: { color: hexA(MARKS.down, 0.75), borderRadius: [0, 0, 3, 3] } },
      ],
    });
  }, [v]);

  const profile: EChartsOption | null = useMemo(() => {
    if (!v || v.empty) return null;
    const maxV = Math.max(...v.profile.map((p) => p.volume));
    const poc = v.profile.find((p) => p.volume === maxV);
    return base({
      grid: grid({ top: 10, left: 4, right: 10 }),
      tooltip: tooltip({ trigger: "item", formatter: (p: unknown) => {
        const q2 = p as { dataIndex: number };
        const r = v.profile[q2.dataIndex];
        return ttTitle(`Price ${fmtPrice(r.price)}`) + ttRow(meta?.color ?? INK.muted, "Volume", fmtInt(r.volume));
      } }),
      xAxis: valueAxis({ axisLabel: { ...axisCommon.axisLabel, formatter: (n: number) => fmtCompact(n) } }),
      yAxis: { type: "category", data: v.profile.map((p) => fmtPrice(p.price)), ...axisCommon, splitLine: { show: false }, axisLabel: { ...axisCommon.axisLabel, interval: 4 } },
      series: [{
        type: "bar",
        barCategoryGap: "18%",
        data: v.profile.map((p) => ({ value: p.volume, itemStyle: { color: p === poc ? meta?.color : hexA(meta?.color ?? CAT[0], 0.45), borderRadius: [0, 3, 3, 0] } })),
      }],
    });
  }, [v, meta]);

  const micro: EChartsOption | null = useMemo(() => {
    if (!v || v.empty) return null;
    const x = v.flow.map((f) => f.tick);
    const toX = (t: number) => x.reduce((best, c) => (Math.abs(c - t) < Math.abs(best - t) ? c : best), x[0]);
    const tops = ["7%", "39%", "71%"];
    const heights = "22%";
    const names = ["Cancel rate", "Order-to-trade ratio", "Aggressor imbalance"];
    return base({
      title: names.map((n, i) => ({ text: n, top: `${parseInt(tops[i]) - 6}%`, left: 4, textStyle: { color: INK.secondary, fontSize: 11.5, fontWeight: 500 } })),
      axisPointer: { link: [{ xAxisIndex: "all" }] },
      tooltip: tooltip({
        formatter: (ps: unknown) => {
          const arr = ps as { dataIndex: number }[];
          const f = v.flow[arr[0]?.dataIndex];
          return f
            ? ttTitle(`${tickClock(f.tick)} · 1 min`) +
                ttRow(STATUS.critical, "Cancel rate", `${(f.cancel_rate * 100).toFixed(1)}%`) +
                ttRow(STATUS.warning, "Orders per trade", fmtNum(f.otr, 1)) +
                ttRow(f.imbalance >= 0 ? MARKS.up : MARKS.down, "Imbalance", fmtNum(f.imbalance, 2)) +
                ttRow(INK.muted, "Placed / cancelled", `${fmtInt(f.placed)} / ${fmtInt(f.cancelled)}`)
            : "";
        },
      }),
      grid: tops.map((t) => ({ left: 48, right: 14, top: t, height: heights })),
      xAxis: [0, 1, 2].map((i) => ({ type: "category", gridIndex: i, data: x.map(String), ...axisCommon, splitLine: { show: false }, axisLabel: i === 2 ? { ...axisCommon.axisLabel, formatter: (t: string) => tickClock(+t) } : { show: false } })),
      yAxis: [
        valueAxis({ gridIndex: 0, max: 1, splitNumber: 2, axisLabel: { ...axisCommon.axisLabel, formatter: (n: number) => `${Math.round(n * 100)}%` } }),
        valueAxis({ gridIndex: 1, splitNumber: 2 }),
        valueAxis({ gridIndex: 2, min: -1, max: 1, splitNumber: 2 }),
      ],
      dataZoom: [{ type: "inside", xAxisIndex: [0, 1, 2] }],
      series: [
        {
          type: "line", xAxisIndex: 0, yAxisIndex: 0, data: v.flow.map((f) => f.cancel_rate), showSymbol: false, lineStyle: { width: 1.6, color: STATUS.critical },
          areaStyle: { color: hexA(STATUS.critical, 0.1) },
          markLine: { silent: true, symbol: "none", data: [{ yAxis: 0.5 }], lineStyle: { color: INK.ref, type: "solid" }, label: { formatter: "50%", color: INK.muted, fontSize: 10, position: "insideEndTop" } },
          markArea: { silent: true, data: v.events.map((e) => [{ xAxis: String(toX(e.start_tick)), itemStyle: { color: hexA(personaColor(e.persona), 0.1) } }, { xAxis: String(toX(e.end_tick)) }]) as never },
        },
        {
          type: "line", xAxisIndex: 1, yAxisIndex: 1, data: v.flow.map((f) => f.otr), showSymbol: false, lineStyle: { width: 1.6, color: STATUS.warning }, areaStyle: { color: hexA(STATUS.warning, 0.08) },
          markArea: { silent: true, data: v.events.map((e) => [{ xAxis: String(toX(e.start_tick)), itemStyle: { color: hexA(personaColor(e.persona), 0.1) } }, { xAxis: String(toX(e.end_tick)) }]) as never },
        },
        {
          type: "bar", xAxisIndex: 2, yAxisIndex: 2, data: v.flow.map((f) => ({ value: f.imbalance, itemStyle: { color: f.imbalance >= 0 ? hexA(MARKS.up, 0.75) : hexA(MARKS.down, 0.75), borderRadius: f.imbalance >= 0 ? [2, 2, 0, 0] : [0, 0, 2, 2] } })),
          markArea: { silent: true, data: v.events.map((e) => [{ xAxis: String(toX(e.start_tick)), itemStyle: { color: hexA(personaColor(e.persona), 0.1) } }, { xAxis: String(toX(e.end_tick)) }]) as never },
        },
      ],
    });
  }, [v]);

  const returns: EChartsOption | null = useMemo(() => {
    if (!v?.returns) return null;
    const r = v.returns;
    return base({
      grid: grid({ top: 34, bottom: 24 }),
      legend: legend({ data: ["Observed", "Normal fit"] }),
      tooltip: tooltip({ formatter: (ps: unknown) => {
        const arr = ps as { dataIndex: number }[];
        const i = arr[0]?.dataIndex ?? 0;
        return ttTitle(`${fmtNum(r.centers[i], 2)} bp`) + ttRow(meta?.color ?? INK.muted, "Observed density", fmtNum(r.density[i], 4)) + ttRow(INK.secondary, "Normal fit", fmtNum(r.normal[i], 4));
      } }),
      xAxis: { type: "category", data: r.centers.map((c) => c.toFixed(1)), ...axisCommon, splitLine: { show: false }, name: "30-second return (bp)", nameLocation: "middle", nameGap: 24, axisLabel: { ...axisCommon.axisLabel, interval: 9 } },
      yAxis: valueAxis({ splitNumber: 3 }),
      series: [
        { name: "Observed", type: "bar", data: r.density, barCategoryGap: "8%", itemStyle: { color: hexA(meta?.color ?? CAT[0], 0.7), borderRadius: [2, 2, 0, 0] } },
        { name: "Normal fit", type: "line", data: r.normal, showSymbol: false, smooth: true, lineStyle: { width: 1.6, color: INK.secondary } },
      ],
    });
  }, [v, meta]);

  const heat: EChartsOption | null = useMemo(() => {
    if (!v || v.empty || !v.heatmap.traders.length) return null;
    const h = v.heatmap;
    const cells: [number, number, number][] = [];
    h.values.forEach((row, y) => row.forEach((val, x) => cells.push([x, y, val])));
    const max = Math.max(1, ...cells.map((c) => c[2]));
    return base({
      grid: grid({ top: 8, right: 64, bottom: 8, left: 4 }),
      tooltip: { ...tooltip({ trigger: "item" }), formatter: (p: unknown) => {
        const val = (p as { value: [number, number, number] }).value;
        return ttTitle(`${tickClock(h.buckets[val[0]])} · 5 min`) + ttRow(SEQ_BLUE[5], h.traders[val[1]], `${fmtInt(val[2])} contracts`);
      } },
      xAxis: { type: "category", data: h.buckets.map((b) => tickClock(b).slice(0, 5)), ...axisCommon, splitLine: { show: false }, axisLine: { show: false } },
      yAxis: { type: "category", data: h.traders, ...axisCommon, splitLine: { show: false }, axisLine: { show: false }, inverse: true, axisLabel: { ...axisCommon.axisLabel, fontSize: 10 } },
      visualMap: { min: 0, max, calculable: false, orient: "vertical", right: 0, top: "middle", itemHeight: 160, itemWidth: 10, inRange: { color: SEQ_BLUE }, textStyle: { color: INK.muted, fontFamily: MONO, fontSize: 10 }, formatter: (n: unknown) => fmtCompact(n as number) },
      series: [{ type: "heatmap", data: cells, itemStyle: { borderColor: INK.surface, borderWidth: 1.5, borderRadius: 2 }, emphasis: { itemStyle: { borderColor: INK.primary } } }],
    });
  }, [v]);

  if (q.error) return <ErrorNote error={q.error} />;
  if (!v) {
    return (
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-6">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
        <Skeleton className="h-[480px]" />
      </div>
    );
  }
  if (v.empty) return <ErrorNote error={`No trades for ${commodity} in this test market.`} />;

  const personas = [...new Set(v.events.map((e) => e.persona))];
  const up = v.kpi.pct >= 0;

  return (
    <div className={q.isFetching ? "opacity-70 transition-opacity" : "transition-opacity"}>
      <div className="panel mb-4 flex flex-wrap items-center gap-x-8 gap-y-3 px-5 py-4">
        <div>
          <div className="eyebrow">{meta?.ticker}=F · {meta?.sector}</div>
          <div className="mt-1 flex items-baseline gap-3">
            <span className="text-[30px] font-semibold tracking-tight text-fg">{fmtPrice(v.kpi.last)}</span>
            <span className={up ? "font-mono text-[14px] text-good" : "font-mono text-[14px] text-critical"}>
              {up ? "▲" : "▼"} {fmtSignedPct(v.kpi.pct)}
            </span>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-x-8 gap-y-1 font-mono text-[12px] sm:grid-cols-4">
          <div><span className="text-fg-muted">Open </span><span className="text-fg">{fmtPrice(v.kpi.first)}</span></div>
          <div><span className="text-fg-muted">High </span><span className="text-fg">{fmtPrice(v.kpi.high)}</span></div>
          <div><span className="text-fg-muted">Low </span><span className="text-fg">{fmtPrice(v.kpi.low)}</span></div>
          <div><span className="text-fg-muted">Range </span><span className="text-fg">{fmtNum(((v.kpi.high - v.kpi.low) / v.kpi.first) * 1e4, 0)} bp</span></div>
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-2">
          {personas.length ? personas.map((p) => <PersonaBadge key={p} persona={p} />) : <span className="text-[12px] text-fg-muted">No manipulation in this book</span>}
        </div>
      </div>

      <StatGrid className="grid-cols-2 lg:grid-cols-4">
        <StatTile label="Trades" value={v.kpi.trades} format={fmtInt} />
        <StatTile label="Volume" value={v.kpi.volume} format={fmtInt} />
        <StatTile label="Active accounts" value={v.kpi.traders} format={fmtInt} />
        <StatTile label="Episodes in book" value={v.kpi.events} format={fmtInt} tone={v.kpi.events ? "warn" : "good"} />
      </StatGrid>

      <div className="mt-4">
        <ChartCard
          title="Price action"
          subtitle={`${v.bucket_ticks.bar}-second candles with session VWAP. Shaded spans are manipulation episodes. Drag the slider or scroll to zoom.`}
          table={{
            columns: [
              { key: "time", label: "Time" },
              { key: "open", label: "Open", align: "right", format: (x) => fmtPrice(x as number) },
              { key: "high", label: "High", align: "right", format: (x) => fmtPrice(x as number) },
              { key: "low", label: "Low", align: "right", format: (x) => fmtPrice(x as number) },
              { key: "close", label: "Close", align: "right", format: (x) => fmtPrice(x as number) },
              { key: "vwap", label: "VWAP", align: "right", format: (x) => fmtPrice(x as number) },
              { key: "volume", label: "Volume", align: "right", format: (x) => fmtInt(x as number) },
            ],
            rows: v.candles.map((c) => ({ ...c, time: tickClock(c.tick) })),
          }}
        >
          {candles && <EChart option={candles} height={480} />}
        </ChartCard>
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-[1.6fr_1fr]">
        <ChartCard title="Order flow" subtitle="Volume that crossed the spread each minute. Buyers above the line, sellers below.">
          {flow && <EChart option={flow} height={300} />}
        </ChartCard>
        <ChartCard title="Volume profile" subtitle="Where in the price range the session traded. The brightest bar is the point of control.">
          {profile && <EChart option={profile} height={300} />}
        </ChartCard>
      </div>

      <div className="mt-4">
        <ChartCard
          title="Microstructure signals"
          subtitle="One-minute cancel rate, orders per trade and aggressor imbalance. Cancel rates above half and order-to-trade spikes are classic spoofing and layering footprints."
          table={{
            columns: [
              { key: "time", label: "Time" },
              { key: "cancel_rate", label: "Cancel rate", align: "right", format: (x) => `${((x as number) * 100).toFixed(1)}%` },
              { key: "otr", label: "Orders / trade", align: "right", format: (x) => fmtNum(x as number, 1) },
              { key: "imbalance", label: "Imbalance", align: "right", format: (x) => fmtNum(x as number, 2) },
              { key: "placed", label: "Placed", align: "right", format: (x) => fmtInt(x as number) },
              { key: "cancelled", label: "Cancelled", align: "right", format: (x) => fmtInt(x as number) },
            ],
            rows: v.flow.map((f) => ({ ...f, time: tickClock(f.tick) })),
          }}
        >
          {micro && <EChart option={micro} height={430} group={`micro-${commodity}`} />}
        </ChartCard>
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-2">
        <ChartCard
          title="Return distribution"
          subtitle={v.returns ? `30-second returns against a normal fit. Excess kurtosis ${fmtNum(v.returns.excess_kurtosis, 1)}: real futures returns are fat-tailed too.` : "Not enough trades."}
        >
          {returns ? <EChart option={returns} height={320} /> : <div className="py-10 text-center text-[12.5px] text-fg-muted">Not enough price changes for a distribution.</div>}
        </ChartCard>
        <ChartCard title="Who traded when" subtitle="Top 20 accounts by volume, in five-minute buckets.">
          {heat && <EChart option={heat} height={320} />}
        </ChartCard>
      </div>
    </div>
  );
}
