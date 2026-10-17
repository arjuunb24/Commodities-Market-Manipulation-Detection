export const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

export const fmtInt = (v?: number | null) => (isNum(v) ? Math.round(v).toLocaleString("en-US") : "-");

export const fmtNum = (v?: number | null, digits = 2) =>
  isNum(v) ? v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits }) : "-";

export const fmtPct = (v?: number | null, digits = 0) => (isNum(v) ? `${(v * 100).toFixed(digits)}%` : "-");

export const fmtSignedPct = (v?: number | null, digits = 2) =>
  isNum(v) ? `${v >= 0 ? "+" : ""}${v.toFixed(digits)}%` : "-";

export const fmtCompact = (v?: number | null) =>
  isNum(v) ? new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(v) : "-";

export const fmtPrice = (v?: number | null) => {
  if (!isNum(v)) return "-";
  const d = v >= 1000 ? 1 : v >= 100 ? 2 : v >= 10 ? 2 : 3;
  return v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
};

export const fmtVal = (v: unknown): string => {
  if (v === null || v === undefined) return "-";
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString("en-US") : String(+v.toPrecision(4));
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (Array.isArray(v)) return v.map(fmtVal).join(", ");
  if (typeof v === "object") return Object.entries(v as Record<string, unknown>).map(([k, x]) => `${k}: ${fmtVal(x)}`).join(", ");
  return String(v);
};

/** Ticks are simulated seconds; show them as a session clock (day starts at 00:00:00). */
export const tickClock = (t: number) => {
  const s = ((Math.floor(t) % 86400) + 86400) % 86400;
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
};

export const runLabel = (run: string) => {
  const m = run.match(/adversarial_(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z_seed(\d+)(.*)/);
  if (!m) return { date: run, time: "", seed: "", tag: "" };
  const [, y, mo, d, h, mi, , seed, tag] = m;
  const date = new Date(Date.UTC(+y, +mo - 1, +d, +h, +mi));
  return {
    date: date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }),
    time: date.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false, timeZone: "UTC" }) + " UTC",
    seed,
    tag: tag.replace(/^_/, "").replace(/_/g, " "),
  };
};

export const humanize = (s: string) => s.replace(/_/g, " ");
