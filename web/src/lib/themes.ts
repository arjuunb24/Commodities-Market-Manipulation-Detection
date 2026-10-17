/**
 * Theme catalogue and store.
 *
 * A theme is two layers: interface tokens (written to CSS variables, so Tailwind classes
 * update without a re-render) and a data palette (written into the chart tokens in
 * lib/theme.ts). Data palettes only re-step the validated categorical hues for the
 * theme's surface; they never re-order or invent hues. Checked with the dataviz validator:
 * the dark set passes on every dark panel here, the amber set swaps layering to the violet
 * slot so it cannot be read as the accent, and the light set passes on the light panels
 * (two light steps sit under 3:1, relieved by persona marker shapes and legends).
 */
import { useSyncExternalStore } from "react";
import { applyDataPalette, type DataPalette, type INK } from "./theme";

export interface Ui {
  bg: string;
  bg2: string;
  panel: string;
  panel2: string;
  line: string;
  line2: string;
  text: string;
  text2: string;
  muted: string;
  accent: string;
  accentInk: string;
  good: string;
  warn: string;
  serious: string;
  critical: string;
  /** two soft light pools in the page backdrop */
  glow: [string, string];
}

export interface Theme {
  id: string;
  name: string;
  tagline: string;
  mode: "dark" | "light";
  ui: Ui;
  data: DataPalette;
}

const DARK_DATA: DataPalette = {
  personas: { spoofing: "#3987e5", wash_trading: "#d95926", pump_and_dump: "#199e70", layering: "#c98500", anomaly: "#8a94a3" },
  cat: ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
  status: { good: "#22b14c", warning: "#fab219", serious: "#ec835a", critical: "#e04848" },
  seq: ["#13243b", "#184f95", "#1c5cab", "#256abf", "#3987e5", "#5598e7", "#86b6ef", "#b7d3f6"],
  diverging: ["#e66767", "#a8504f", "#383835", "#2c5f9e", "#3987e5"],
  marks: { up: "#22b14c", down: "#e04848", vwap: "#e7ecf2", volume: "#3a4a5e", price: "#8da2bd", neutral: "#8a94a3" },
};

const LIGHT_DATA: DataPalette = {
  personas: { spoofing: "#2a78d6", wash_trading: "#eb6834", pump_and_dump: "#1baf7a", layering: "#eda100", anomaly: "#898781" },
  cat: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
  status: { good: "#0ca30c", warning: "#e0a000", serious: "#ec835a", critical: "#d03b3b" },
  seq: ["#e3eefc", "#b7d3f6", "#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#1c5cab", "#104281"],
  diverging: ["#e34948", "#f09b9a", "#e6e4de", "#8ab8ee", "#2a78d6"],
  marks: { up: "#0ca30c", down: "#d03b3b", vwap: "#1f2733", volume: "#c3ccd7", price: "#5a6f8a", neutral: "#8a8f98" },
};

const dark = (ui: Omit<Ui, "good" | "warn" | "serious" | "critical"> & Partial<Ui>): Ui => ({
  good: "#22b14c",
  warn: "#fab219",
  serious: "#ec835a",
  critical: "#e04848",
  ...ui,
});
const light = (ui: Omit<Ui, "good" | "warn" | "serious" | "critical"> & Partial<Ui>): Ui => ({
  good: "#0a7d2c",
  warn: "#9a6400",
  serious: "#b9502a",
  critical: "#c22f2f",
  ...ui,
});

export const THEMES: Theme[] = [
  {
    id: "terminal",
    name: "Veridex Terminal",
    tagline: "Slate and signal mint. The house look.",
    mode: "dark",
    ui: dark({
      bg: "#080b10", bg2: "#0b1016", panel: "#0f151c", panel2: "#141b24", line: "#1b2430", line2: "#26313f",
      text: "#e7ecf2", text2: "#a7b1bf", muted: "#6e7a8a", accent: "#4fe3b5", accentInk: "#04140f",
      glow: ["rgba(79,227,181,0.07)", "rgba(57,135,229,0.08)"],
    }),
    data: DARK_DATA,
  },
  {
    id: "amber",
    name: "Amber Terminal",
    tagline: "Phosphor amber on warm black, trading-floor heritage.",
    mode: "dark",
    ui: dark({
      bg: "#0b0a08", bg2: "#100e0b", panel: "#15130f", panel2: "#1b1813", line: "#2a2620", line2: "#3a342b",
      text: "#f2ead8", text2: "#c2b8a3", muted: "#8a8170", accent: "#ffb000", accentInk: "#1a1100",
      glow: ["rgba(255,176,0,0.07)", "rgba(255,120,40,0.05)"],
    }),
    data: { ...DARK_DATA, personas: { ...DARK_DATA.personas, layering: "#9085e9" } },
  },
  {
    id: "ledger",
    name: "Midnight Ledger",
    tagline: "Deep navy and brushed gold. Private-bank calm.",
    mode: "dark",
    ui: dark({
      bg: "#070b14", bg2: "#0a101c", panel: "#0e1524", panel2: "#131c2e", line: "#1c2740", line2: "#283654",
      text: "#e6e9f0", text2: "#aab3c7", muted: "#7d8aa6", accent: "#d4af6a", accentInk: "#1a1405",
      glow: ["rgba(212,175,106,0.06)", "rgba(70,110,200,0.09)"],
    }),
    data: DARK_DATA,
  },
  {
    id: "copper",
    name: "Oxidised Copper",
    tagline: "Verdigris teal over charcoal green, copper highlights.",
    mode: "dark",
    ui: dark({
      bg: "#0a0f0e", bg2: "#0d1412", panel: "#121a18", panel2: "#17211e", line: "#22302c", line2: "#2e3f3a",
      text: "#e4ebe8", text2: "#a9b8b2", muted: "#72847d", accent: "#5fb8a5", accentInk: "#04120e",
      glow: ["rgba(95,184,165,0.07)", "rgba(200,127,79,0.07)"],
    }),
    data: DARK_DATA,
  },
  {
    id: "redteam",
    name: "Red Team",
    tagline: "Graphite with a cold detector cyan and a hot adversary coral.",
    mode: "dark",
    ui: dark({
      bg: "#0b0d10", bg2: "#0f1216", panel: "#13161b", panel2: "#191d23", line: "#252a32", line2: "#323843",
      text: "#e8ecf1", text2: "#aeb5bf", muted: "#757d89", accent: "#5cc8ff", accentInk: "#021320",
      critical: "#e5383b",
      glow: ["rgba(92,200,255,0.07)", "rgba(255,107,92,0.07)"],
    }),
    data: DARK_DATA,
  },
  {
    id: "obsidian",
    name: "Obsidian Violet",
    tagline: "Ink-black glass with an electric violet edge.",
    mode: "dark",
    ui: dark({
      bg: "#09080e", bg2: "#0d0c14", panel: "#121019", panel2: "#18151f", line: "#241f33", line2: "#332c47",
      text: "#ebe8f2", text2: "#b1aac2", muted: "#7c7590", accent: "#a98bff", accentInk: "#0e0620",
      glow: ["rgba(169,139,255,0.08)", "rgba(90,200,250,0.05)"],
    }),
    data: DARK_DATA,
  },
  {
    id: "carbon",
    name: "Carbon",
    tagline: "True black for OLED desks, a single lime signal.",
    mode: "dark",
    ui: dark({
      bg: "#000000", bg2: "#050505", panel: "#0b0b0c", panel2: "#111113", line: "#1e1e21", line2: "#2b2b30",
      text: "#f4f4f5", text2: "#b4b4bb", muted: "#7a7a84", accent: "#c6f432", accentInk: "#121800",
      glow: ["rgba(198,244,50,0.04)", "rgba(255,255,255,0.025)"],
    }),
    data: DARK_DATA,
  },
  {
    id: "paper",
    name: "Paper Tape",
    tagline: "Newsprint, ink and oxblood. Reads like a research note.",
    mode: "light",
    ui: light({
      bg: "#f5f2eb", bg2: "#efebe2", panel: "#fffdf8", panel2: "#f8f4ec", line: "#e4ddcf", line2: "#d3cab8",
      text: "#1a1814", text2: "#4d473d", muted: "#7d7466", accent: "#8c1c2c", accentInk: "#fff8f0",
      glow: ["rgba(140,28,44,0.05)", "rgba(42,120,214,0.05)"],
    }),
    data: LIGHT_DATA,
  },
  {
    id: "glacier",
    name: "Glacier",
    tagline: "Cool daylight white with a deep-water teal.",
    mode: "light",
    ui: light({
      bg: "#eef2f6", bg2: "#e7ecf2", panel: "#ffffff", panel2: "#f5f8fb", line: "#dde3ea", line2: "#c9d2dd",
      text: "#0f1720", text2: "#3f4b5a", muted: "#6b7787", accent: "#0a7f86", accentInk: "#f2fffe",
      glow: ["rgba(10,127,134,0.06)", "rgba(42,120,214,0.06)"],
    }),
    data: LIGHT_DATA,
  },
];

const KEY = "vx-theme";
const byId = (id: string | null) => THEMES.find((t) => t.id === id) ?? THEMES[0];

const channels = (hex: string) => {
  const n = parseInt(hex.slice(1), 16);
  return `${(n >> 16) & 255} ${(n >> 8) & 255} ${n & 255}`;
};
const rgba = (hex: string, a: number) => `rgb(${channels(hex)} / ${a})`;

function apply(t: Theme, animate: boolean) {
  const root = document.documentElement;
  const u = t.ui;
  const vars: Record<string, string> = {
    bg: u.bg, "bg-2": u.bg2, panel: u.panel, "panel-2": u.panel2, line: u.line, "line-2": u.line2,
    text: u.text, "text-2": u.text2, muted: u.muted, accent: u.accent, "accent-ink": u.accentInk,
    good: u.good, warn: u.warn, serious: u.serious, critical: u.critical,
  };
  for (const [k, v] of Object.entries(vars)) {
    root.style.setProperty(`--${k}`, v);
    root.style.setProperty(`--c-${k}`, channels(v));
  }
  const tint = t.mode === "dark" ? "255 255 255" : "0 0 0";
  root.style.setProperty("--c-tint", tint);
  root.style.setProperty("--accent-soft", rgba(u.accent, 0.12));
  root.style.setProperty("--glow-1", u.glow[0]);
  root.style.setProperty("--glow-2", u.glow[1]);
  root.style.setProperty("--grid-line", t.mode === "dark" ? "rgba(255,255,255,0.018)" : "rgba(0,0,0,0.035)");
  root.style.setProperty("--shadow-drop", t.mode === "dark" ? "rgba(0,0,0,0.65)" : "rgba(40,30,10,0.12)");
  root.style.colorScheme = t.mode;
  root.dataset.theme = t.mode;
  root.dataset.palette = t.id;
  document.querySelector('meta[name="theme-color"]')?.setAttribute("content", u.bg);

  const ink: typeof INK = {
    primary: u.text,
    secondary: u.text2,
    muted: u.muted,
    grid: u.line,
    axis: u.line2,
    surface: u.panel,
    tooltip: rgba(u.panel2, 0.97),
    inactive: u.line2,
    ref: rgba(u.text, 0.25),
    shade: `rgb(${tint} / ${t.mode === "dark" ? 0.03 : 0.04})`,
    accent: u.accent,
  };
  applyDataPalette(t.data, ink);

  if (animate) {
    root.classList.add("theme-anim");
    window.clearTimeout(timer);
    timer = window.setTimeout(() => root.classList.remove("theme-anim"), 400);
  }
}

let timer = 0;
let current: Theme = (() => {
  try {
    return byId(localStorage.getItem(KEY));
  } catch {
    return THEMES[0];
  }
})();
const listeners = new Set<() => void>();

/** Apply the saved theme before the first render, so the page never flashes the default. */
export function initTheme() {
  apply(current, false);
}

export function setTheme(id: string) {
  const next = byId(id);
  if (next.id === current.id) return;
  current = next;
  apply(next, true);
  try {
    localStorage.setItem(KEY, next.id);
  } catch {
    /* private window or blocked storage: the choice lasts for this visit only */
  }
  listeners.forEach((l) => l());
}

export function cycleTheme(step = 1) {
  const i = THEMES.findIndex((t) => t.id === current.id);
  setTheme(THEMES[(i + step + THEMES.length) % THEMES.length].id);
}

export function useTheme(): Theme {
  return useSyncExternalStore(
    (l) => {
      listeners.add(l);
      return () => listeners.delete(l);
    },
    () => current,
  );
}
