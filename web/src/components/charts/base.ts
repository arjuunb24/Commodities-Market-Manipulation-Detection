import type { EChartsOption } from "echarts";
import { INK } from "@/lib/theme";

/* eslint-disable @typescript-eslint/no-explicit-any */
// Option fragments are plain objects; ECharts' literal-union types are enforced where `base` returns.
type Opt = Record<string, any>;

/** House style for every chart: recessive hairline grid, muted axes, text in ink tokens (never series colours). */
export const FONT = '"IBM Plex Sans", system-ui, sans-serif';
export const MONO = '"IBM Plex Mono", ui-monospace, monospace';

/** Phone-portrait widths: charts thin out labels and keep legends to one scrolling row. */
export const narrow = () => typeof window !== "undefined" && window.innerWidth < 640;

export const tooltip = (extra: Opt = {}): any => ({
  trigger: "axis",
  confine: true,
  backgroundColor: INK.tooltip,
  borderColor: INK.axis,
  borderWidth: 1,
  padding: [8, 10],
  textStyle: { color: INK.primary, fontFamily: FONT, fontSize: 12 },
  extraCssText: "border-radius:10px; box-shadow:0 12px 32px -12px rgba(0,0,0,.7); backdrop-filter: blur(6px);",
  axisPointer: {
    type: "line",
    lineStyle: { color: INK.ref, width: 1 },
    crossStyle: { color: INK.ref },
    label: { backgroundColor: INK.grid, color: INK.primary, fontFamily: MONO, fontSize: 11 },
  },
  ...extra,
});

// Getters, so a spread (...axisCommon) always reads the current theme's ink.
export const axisCommon: any = {
  get axisLine() {
    return { lineStyle: { color: INK.axis } };
  },
  axisTick: { show: false },
  get axisLabel() {
    return { color: INK.muted, fontFamily: MONO, fontSize: narrow() ? 9.5 : 10.5, hideOverlap: true };
  },
  get splitLine() {
    return { lineStyle: { color: INK.grid, width: 1 } };
  },
  get nameTextStyle() {
    return { color: INK.muted, fontFamily: FONT, fontSize: 11 };
  },
};

export const valueAxis = (extra: Opt = {}): any => ({ type: "value", scale: false, ...axisCommon, ...extra });

export const legend = (extra: Opt = {}): any => ({
  top: 0,
  right: 0,
  icon: "roundRect",
  itemWidth: 10,
  itemHeight: 10,
  itemGap: 14,
  textStyle: { color: INK.secondary, fontFamily: FONT, fontSize: 11.5 },
  inactiveColor: INK.inactive,
  ...extra,
  // A wrapped legend would spill over the plot on a phone; one swipeable row keeps it clear.
  ...(narrow() ? { type: "scroll", left: 0, right: 0, itemGap: 10, pageIconSize: 10, pageTextStyle: { color: INK.muted, fontSize: 10 } } : {}),
});

export const grid = (extra: Opt = {}): any => ({ left: 8, right: 12, top: 34, bottom: 8, containLabel: true, ...extra });

export const base = (o: Opt): EChartsOption => ({
  backgroundColor: "transparent",
  textStyle: { fontFamily: FONT, color: INK.secondary },
  animationDuration: 700,
  animationDurationUpdate: 450,
  animationEasing: "cubicOut",
  animationEasingUpdate: "cubicInOut",
  ...o,
}) as EChartsOption;

/** Tooltip row: coloured mark carries identity, text stays in ink. */
export const ttRow = (color: string, label: string, value: string) =>
  `<div style="display:flex;align-items:center;gap:8px;justify-content:space-between;min-width:170px;margin-top:3px">` +
  `<span style="display:flex;align-items:center;gap:6px;color:${INK.secondary}"><span style="width:8px;height:8px;border-radius:2px;background:${color};display:inline-block"></span>${label}</span>` +
  `<span style="font-family:${MONO};color:${INK.primary}">${value}</span></div>`;

export const ttTitle = (t: string) => `<div style="font-family:${MONO};font-size:11px;color:${INK.muted};margin-bottom:2px">${t}</div>`;

export const hexA = (hex: string, a: number) => {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
};
