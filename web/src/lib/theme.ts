/**
 * Design tokens for charts and data marks.
 *
 * Categorical colours are the validated dark-surface steps of the reference palette,
 * assigned in fixed order (checked with the dataviz validator against the panel
 * surface #0f151c: lightness band, chroma floor, adjacent CVD separation >= 8,
 * normal-vision >= 15, contrast >= 3:1). Personas carry a marker shape as well,
 * because four hues cannot all be told apart pairwise under every colour-vision type.
 */

/*
 * Every colour below is a mutable token: applyDataPalette() (called by the theme store in
 * lib/themes.ts) rewrites them in place when the viewer switches theme, and the shell remounts
 * the charts so their options are rebuilt from the new values.
 */
export const INK = {
  primary: "#e7ecf2",
  secondary: "#a7b1bf",
  muted: "#6e7a8a",
  grid: "#1b2430",
  axis: "#26313f",
  /** chart surface: the 2px gap ring around marks */
  surface: "#0f151c",
  tooltip: "rgba(12,17,24,0.96)",
  inactive: "#3a4554",
  /** reference lines and crosshairs */
  ref: "rgba(231,236,242,0.25)",
  /** hover shading behind a bar group */
  shade: "rgba(255,255,255,0.03)",
  accent: "#4fe3b5",
};

/** Non-identity marks: candle direction, VWAP, volume bars, a neutral price line, the frozen-model grey. */
export const MARKS = { up: "#22b14c", down: "#e04848", vwap: "#e7ecf2", volume: "#3a4a5e", price: "#8da2bd", neutral: "#8a94a3" };

export const PERSONAS = ["spoofing", "wash_trading", "pump_and_dump", "layering"] as const;
export type Persona = (typeof PERSONAS)[number];

export const PERSONA_META: Record<string, { label: string; short: string; color: string; symbol: string; blurb: string }> = {
  spoofing: {
    label: "Spoofing",
    short: "SPF",
    color: "#3987e5",
    symbol: "circle",
    blurb: "Large non-bona-fide orders on one side to move the price toward a small genuine order on the other.",
  },
  wash_trading: {
    label: "Wash trading",
    short: "WSH",
    color: "#d95926",
    symbol: "rect",
    blurb: "Colluding accounts trade with each other to print artificial volume without changing ownership.",
  },
  pump_and_dump: {
    label: "Pump and dump",
    short: "PND",
    color: "#199e70",
    symbol: "triangle",
    blurb: "Quiet accumulation, an aggressive ignition burst to draw in trend followers, then distribution.",
  },
  layering: {
    label: "Layering",
    short: "LYR",
    color: "#c98500",
    symbol: "diamond",
    blurb: "Moderate orders stacked on consecutive price levels to fake depth, cancelled together after a fill.",
  },
  anomaly: {
    label: "Anomaly",
    short: "ANM",
    color: "#8a94a3",
    symbol: "emptyCircle",
    blurb: "Unusual for the IsolationForest, with no persona classifier behind it.",
  },
};

export const personaLabel = (p?: string | null) => (p ? PERSONA_META[p]?.label ?? p.replace(/_/g, " ") : "-");
export const personaColor = (p?: string | null) => (p ? PERSONA_META[p]?.color ?? INK.muted : INK.muted);

/** Commodity identity colours: all eight validated slots, legend order = slot order. */
export const COMMODITIES = [
  "crude_oil_wti",
  "copper",
  "natural_gas",
  "gold",
  "silver",
  "corn",
  "brent_crude",
  "wheat",
] as const;

export const COMMODITY_META: Record<string, { label: string; ticker: string; color: string; sector: string }> = {
  crude_oil_wti: { label: "WTI Crude", ticker: "CL", color: "#3987e5", sector: "Energy" },
  copper: { label: "Copper", ticker: "HG", color: "#d95926", sector: "Metals" },
  natural_gas: { label: "Natural Gas", ticker: "NG", color: "#199e70", sector: "Energy" },
  gold: { label: "Gold", ticker: "GC", color: "#c98500", sector: "Metals" },
  silver: { label: "Silver", ticker: "SI", color: "#d55181", sector: "Metals" },
  corn: { label: "Corn", ticker: "ZC", color: "#008300", sector: "Grains" },
  brent_crude: { label: "Brent Crude", ticker: "BZ", color: "#9085e9", sector: "Energy" },
  wheat: { label: "Wheat", ticker: "ZW", color: "#e66767", sector: "Grains" },
};

export const commodityLabel = (c: string) => COMMODITY_META[c]?.label ?? c.replace(/_/g, " ");
export const commodityColor = (c: string) => COMMODITY_META[c]?.color ?? INK.muted;
export const sortCommodities = (cs: string[]) =>
  [...cs].sort((a, b) => (COMMODITIES as readonly string[]).indexOf(a) - (COMMODITIES as readonly string[]).indexOf(b));

/** The eight categorical slots in fixed order (commodity identity, two-series comparisons). */
export const CAT = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];

/** Status palette: reserved for state, always shipped with an icon and a label. */
export const STATUS = { good: "#22b14c", warning: "#fab219", serious: "#ec835a", critical: "#e04848" };

/** Sequential (one hue, dark surface: low values recede toward the surface). */
export const SEQ_BLUE = ["#13243b", "#184f95", "#1c5cab", "#256abf", "#3987e5", "#5598e7", "#86b6ef", "#b7d3f6"];
/** Diverging blue <-> red with a neutral grey midpoint. */
export const DIVERGING = ["#e66767", "#a8504f", "#383835", "#2c5f9e", "#3987e5"];

export interface DataPalette {
  personas: Record<string, string>;
  cat: string[];
  status: typeof STATUS;
  seq: string[];
  diverging: string[];
  marks: typeof MARKS;
}

/** Rewrite every data token in place (module consumers keep their references). */
export function applyDataPalette(d: DataPalette, ink: typeof INK) {
  Object.assign(INK, ink);
  Object.assign(MARKS, d.marks);
  Object.assign(STATUS, d.status);
  CAT.splice(0, CAT.length, ...d.cat);
  SEQ_BLUE.splice(0, SEQ_BLUE.length, ...d.seq);
  DIVERGING.splice(0, DIVERGING.length, ...d.diverging);
  for (const [p, c] of Object.entries(d.personas)) if (PERSONA_META[p]) PERSONA_META[p].color = c;
  COMMODITIES.forEach((c, i) => (COMMODITY_META[c].color = d.cat[i]));
}
