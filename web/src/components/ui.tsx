import clsx from "clsx";
import { animate, motion, useMotionValue, useTransform } from "framer-motion";
import { AlertTriangle, ArrowDown, ArrowUp, ArrowUpDown, CheckCircle2, Info, Table2, XCircle } from "lucide-react";
import { useEffect, useId, useMemo, useRef, useState, type ReactNode } from "react";
import { PERSONA_META } from "@/lib/theme";

// ── Motion presets ───────────────────────────────────────────────────────────

export const fadeUp = {
  hidden: { opacity: 0, y: 10 },
  show: { opacity: 1, y: 0, transition: { duration: 0.42, ease: [0.22, 1, 0.36, 1] } },
};
export const stagger = (delay = 0.05) => ({ hidden: {}, show: { transition: { staggerChildren: delay } } });

export function Reveal({ children, className, delay = 0 }: { children: ReactNode; className?: string; delay?: number }) {
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-40px" }}
      transition={{ duration: 0.5, delay, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  );
}

// ── Animated number ──────────────────────────────────────────────────────────

/**
 * Counts up to `value` whenever it changes. Always renders the same element: swapping a
 * placeholder for the animated span (as an earlier version did) left the count-up never
 * starting, so values that arrived after the first render stuck at 0.
 */
export function AnimatedNumber({ value, format, className }: { value: number | null | undefined; format: (v: number) => string; className?: string }) {
  const valid = value !== null && value !== undefined && Number.isFinite(value);
  const mv = useMotionValue(0);
  const formatRef = useRef(format);
  formatRef.current = format;
  const text = useTransform(mv, (v) => formatRef.current(v));
  const [shown, setShown] = useState(valid);
  useEffect(() => {
    if (!valid) {
      setShown(false);
      return;
    }
    setShown(true);
    const c = animate(mv, value as number, { duration: 0.8, ease: [0.22, 1, 0.36, 1] });
    return () => c.stop();
  }, [value, valid, mv]);
  return <motion.span className={className}>{shown && valid ? text : "-"}</motion.span>;
}

// ── Panels ───────────────────────────────────────────────────────────────────

export function Panel({ children, className, padded = true }: { children: ReactNode; className?: string; padded?: boolean }) {
  return <div className={clsx("panel", padded && "p-4 sm:p-5", className)}>{children}</div>;
}

export interface TableSpec {
  columns: { key: string; label: string; align?: "left" | "right"; format?: (v: unknown, row: Record<string, unknown>) => ReactNode }[];
  rows: Record<string, unknown>[];
}

export function ChartCard({
  title,
  subtitle,
  actions,
  children,
  table,
  className,
  footer,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  table?: TableSpec;
  className?: string;
  footer?: ReactNode;
}) {
  const [showTable, setShowTable] = useState(false);
  return (
    <Reveal className={clsx("panel flex flex-col p-4 sm:p-5", className)}>
      {/* With extra controls the header stacks on phones, so the description keeps the full width. */}
      <div className={clsx("mb-3 flex gap-3", actions ? "flex-col sm:flex-row sm:items-start sm:justify-between" : "items-start justify-between")}>
        <div className="min-w-0 flex-1">
          <h3 className="text-[14px] font-semibold tracking-tight text-fg">{title}</h3>
          {subtitle && <p className="mt-0.5 max-w-3xl text-[12.5px] leading-relaxed text-fg-muted">{subtitle}</p>}
        </div>
        <div className={clsx("flex shrink-0 flex-wrap items-center gap-2", actions ? "justify-start sm:justify-end" : "justify-end")}>
          {actions}
          {table && (
            <button
              onClick={() => setShowTable((s) => !s)}
              className={clsx(
                "focus-ring inline-flex h-7 items-center gap-1.5 rounded-md border px-2 text-[11.5px] transition",
                showTable ? "border-accent/40 bg-accent-soft text-accent" : "border-ink-600 text-fg-2 hover:border-ink-600 hover:bg-ink-800",
              )}
              aria-pressed={showTable}
              title="Show the data behind the chart"
            >
              <Table2 size={13} /> Data
            </button>
          )}
        </div>
      </div>
      <div className="relative min-h-0 flex-1">
        {showTable && table ? <DataTable spec={table} maxHeight={360} dense /> : children}
      </div>
      {footer && <div className="mt-3 border-t border-ink-700 pt-3 text-[12px] text-fg-muted">{footer}</div>}
    </Reveal>
  );
}

// ── Stat tile ────────────────────────────────────────────────────────────────

export function StatTile({
  label,
  value,
  format,
  hint,
  tone,
  icon,
  sub,
}: {
  label: string;
  value: number | null | undefined;
  format: (v: number) => string;
  hint?: string;
  tone?: "good" | "warn" | "critical" | "accent";
  icon?: ReactNode;
  sub?: ReactNode;
}) {
  const toneCls =
    tone === "good" ? "text-good" : tone === "warn" ? "text-warn" : tone === "critical" ? "text-critical" : tone === "accent" ? "text-accent" : "text-fg";
  return (
    <motion.div variants={fadeUp} className="panel group relative overflow-hidden p-4" title={hint}>
      <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-tint/10 to-transparent opacity-0 transition group-hover:opacity-100" />
      <div className="flex items-center justify-between">
        <span className="eyebrow">{label}</span>
        {icon && <span className="text-fg-muted">{icon}</span>}
      </div>
      <div className={clsx("mt-2 text-[26px] font-semibold leading-none tracking-tight", toneCls)}>
        <AnimatedNumber value={value} format={format} />
      </div>
      {sub && <div className="mt-2 text-[12px] text-fg-muted">{sub}</div>}
    </motion.div>
  );
}

export function StatGrid({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <motion.div variants={stagger(0.06)} initial="hidden" animate="show" className={clsx("grid gap-3", className)}>
      {children}
    </motion.div>
  );
}

// ── Badges ───────────────────────────────────────────────────────────────────

export function PersonaMark({ persona, size = 9 }: { persona: string; size?: number }) {
  const m = PERSONA_META[persona] ?? PERSONA_META.anomaly;
  const s = size;
  const shape =
    m.symbol === "rect" ? (
      <rect x={1} y={1} width={s - 2} height={s - 2} rx={1.5} fill={m.color} />
    ) : m.symbol === "triangle" ? (
      <path d={`M${s / 2} 0.6 L${s - 0.6} ${s - 0.8} L0.6 ${s - 0.8} Z`} fill={m.color} />
    ) : m.symbol === "diamond" ? (
      <path d={`M${s / 2} 0.4 L${s - 0.4} ${s / 2} L${s / 2} ${s - 0.4} L0.4 ${s / 2} Z`} fill={m.color} />
    ) : m.symbol === "emptyCircle" ? (
      <circle cx={s / 2} cy={s / 2} r={s / 2 - 1.2} fill="none" stroke={m.color} strokeWidth={1.6} />
    ) : (
      <circle cx={s / 2} cy={s / 2} r={s / 2 - 0.6} fill={m.color} />
    );
  return (
    <svg width={s} height={s} viewBox={`0 0 ${s} ${s}`} aria-hidden className="shrink-0">
      {shape}
    </svg>
  );
}

export function PersonaBadge({ persona, className }: { persona: string; className?: string }) {
  const m = PERSONA_META[persona] ?? PERSONA_META.anomaly;
  return (
    <span className={clsx("inline-flex items-center gap-1.5 rounded-md border border-ink-600 bg-ink-800 px-2 py-0.5 text-[11.5px] font-medium text-fg", className)}>
      <PersonaMark persona={persona} />
      {m.label}
    </span>
  );
}

export type StatusKind = "good" | "warning" | "serious" | "critical" | "neutral";

export function StatusBadge({ kind, children, className }: { kind: StatusKind; children: ReactNode; className?: string }) {
  const Icon = kind === "good" ? CheckCircle2 : kind === "warning" ? AlertTriangle : kind === "neutral" ? Info : XCircle;
  const cls =
    kind === "good"
      ? "text-good border-good/30 bg-good/10"
      : kind === "warning"
        ? "text-warn border-warn/30 bg-warn/10"
        : kind === "serious"
          ? "text-serious border-serious/30 bg-serious/10"
          : kind === "critical"
            ? "text-critical border-critical/30 bg-critical/10"
            : "text-fg-2 border-ink-600 bg-ink-800";
  return (
    <span className={clsx("inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11px] font-medium", cls, className)}>
      <Icon size={12} strokeWidth={2.2} />
      {children}
    </span>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="rounded border border-ink-600 bg-ink-800 px-1.5 py-0.5 font-mono text-[10.5px] text-fg-2">{children}</kbd>;
}

// ── Segmented control (animated indicator) ───────────────────────────────────

export function Segmented<T extends string | number>({
  options,
  value,
  onChange,
  size = "md",
  className,
}: {
  options: { value: T; label: ReactNode; title?: string }[];
  value: T;
  onChange: (v: T) => void;
  size?: "sm" | "md";
  className?: string;
}) {
  const id = useId();
  return (
    <div className={clsx("inline-flex rounded-lg border border-ink-600 bg-ink-900/80 p-0.5", className)} role="tablist">
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={String(o.value)}
            role="tab"
            aria-selected={active}
            title={o.title}
            onClick={() => onChange(o.value)}
            className={clsx(
              "focus-ring relative rounded-md font-medium transition-colors",
              size === "sm" ? "px-2.5 py-1 text-[11.5px]" : "px-3 py-1.5 text-[12.5px]",
              active ? "text-fg" : "text-fg-muted hover:text-fg-2",
            )}
          >
            {active && (
              <motion.span
                layoutId={`seg-${id}`}
                className="absolute inset-0 rounded-md border border-ink-600 bg-ink-700"
                transition={{ type: "spring", stiffness: 500, damping: 38 }}
              />
            )}
            <span className="relative z-10 inline-flex items-center gap-1.5">{o.label}</span>
          </button>
        );
      })}
    </div>
  );
}

// ── Toggle chips ─────────────────────────────────────────────────────────────

export function Chip({ active, onClick, children, title }: { active: boolean; onClick: () => void; children: ReactNode; title?: string }) {
  return (
    <button
      onClick={onClick}
      title={title}
      aria-pressed={active}
      className={clsx(
        "focus-ring inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-[12px] transition",
        active ? "border-ink-600 bg-ink-700 text-fg" : "border-ink-700 bg-transparent text-fg-muted hover:border-ink-600 hover:text-fg-2",
      )}
    >
      {children}
    </button>
  );
}

// ── Callout ──────────────────────────────────────────────────────────────────

export function Callout({ kind = "neutral", title, children }: { kind?: "neutral" | "warning" | "accent"; title?: string; children: ReactNode }) {
  const cls =
    kind === "warning" ? "border-warn/25 bg-warn/[0.06]" : kind === "accent" ? "border-accent/25 bg-accent/[0.05]" : "border-ink-600 bg-ink-800/60";
  const Icon = kind === "warning" ? AlertTriangle : Info;
  return (
    <div className={clsx("flex gap-3 rounded-xl border px-4 py-3", cls)}>
      <Icon size={16} className={clsx("mt-0.5 shrink-0", kind === "warning" ? "text-warn" : kind === "accent" ? "text-accent" : "text-fg-muted")} />
      <div className="text-[12.5px] leading-relaxed text-fg-2">
        {title && <div className="mb-0.5 font-semibold text-fg">{title}</div>}
        {children}
      </div>
    </div>
  );
}

// ── Empty / loading ──────────────────────────────────────────────────────────

export function EmptyState({ title, children, icon }: { title: string; children?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-ink-600 px-6 py-14 text-center">
      {icon && <div className="mb-3 text-fg-muted">{icon}</div>}
      <div className="text-[14px] font-semibold text-fg">{title}</div>
      {children && <div className="mt-1 max-w-md text-[12.5px] leading-relaxed text-fg-muted">{children}</div>}
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={clsx("skeleton", className)} />;
}

export function ErrorNote({ error }: { error: unknown }) {
  return (
    <Callout kind="warning" title="Could not load this view">
      {error instanceof Error ? error.message : String(error)}
    </Callout>
  );
}

// ── Data table (sortable, sticky header) ─────────────────────────────────────

export function DataTable({ spec, maxHeight = 480, dense = false, empty = "No rows." }: { spec: TableSpec; maxHeight?: number; dense?: boolean; empty?: string }) {
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 } | null>(null);
  const rows = useMemo(() => {
    if (!sort) return spec.rows;
    return [...spec.rows].sort((a, b) => {
      const x = a[sort.key];
      const y = b[sort.key];
      if (typeof x === "number" && typeof y === "number") return (x - y) * sort.dir;
      return String(x ?? "").localeCompare(String(y ?? "")) * sort.dir;
    });
  }, [spec.rows, sort]);
  if (!spec.rows.length) return <div className="py-8 text-center text-[12.5px] text-fg-muted">{empty}</div>;
  return (
    <div className="overflow-auto rounded-lg border border-ink-700" style={{ maxHeight }}>
      <table className="w-full border-collapse text-left text-[12.5px]">
        <thead className="sticky top-0 z-10 bg-ink-800/95 backdrop-blur">
          <tr>
            {spec.columns.map((c) => {
              const active = sort?.key === c.key;
              return (
                <th
                  key={c.key}
                  className={clsx("whitespace-nowrap border-b border-ink-600 font-semibold text-fg-2", dense ? "px-2.5 py-1.5" : "px-3 py-2", c.align === "right" && "text-right")}
                >
                  <button
                    className="focus-ring inline-flex items-center gap-1 hover:text-fg"
                    onClick={() => setSort((s) => (s?.key === c.key ? (s.dir === 1 ? { key: c.key, dir: -1 } : null) : { key: c.key, dir: 1 }))}
                  >
                    {c.label}
                    {active ? sort!.dir === 1 ? <ArrowUp size={11} /> : <ArrowDown size={11} /> : <ArrowUpDown size={11} className="opacity-30" />}
                  </button>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-ink-700/70 transition-colors last:border-0 hover:bg-tint/[0.02]">
              {spec.columns.map((c) => (
                <td
                  key={c.key}
                  className={clsx("whitespace-nowrap text-fg-2", dense ? "px-2.5 py-1.5" : "px-3 py-2", c.align === "right" && "num text-right font-mono text-[12px]")}
                >
                  {c.format ? c.format(r[c.key], r) : String(r[c.key] ?? "-")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Page header ──────────────────────────────────────────────────────────────

export function PageHeader({ eyebrow, title, description, right }: { eyebrow: string; title: string; description?: ReactNode; right?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <motion.div initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.4 }} className="eyebrow text-accent/90">
          {eyebrow}
        </motion.div>
        <motion.h1
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, delay: 0.04 }}
          className="mt-1.5 text-[26px] font-semibold tracking-tight text-fg sm:text-[28px]"
        >
          {title}
        </motion.h1>
        {description && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="mt-1.5 max-w-3xl text-[13.5px] leading-relaxed text-fg-muted"
          >
            {description}
          </motion.p>
        )}
      </div>
      {right}
    </div>
  );
}

export function SectionTitle({ children, right }: { children: ReactNode; right?: ReactNode }) {
  return (
    <div className="mb-3 mt-8 flex items-center justify-between gap-3">
      <h2 className="text-[13px] font-semibold uppercase tracking-[0.12em] text-fg-2">{children}</h2>
      {right}
    </div>
  );
}

/** Horizontal 0-1 gauge with the decision threshold as a tick. */
export function Gauge({ value, threshold, color, height = 8 }: { value: number; threshold?: number | null; color: string; height?: number }) {
  const v = Math.max(0, Math.min(1, value));
  return (
    <div className="relative w-full rounded-full bg-tint/[0.06]" style={{ height }}>
      <motion.div
        className="h-full rounded-full"
        style={{ background: color }}
        initial={{ width: 0 }}
        animate={{ width: `${v * 100}%` }}
        transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
      />
      {threshold !== undefined && threshold !== null && (
        <div
          title={`decision threshold ${threshold.toFixed(2)}`}
          className="absolute -top-[3px] w-[2px] rounded bg-fg"
          style={{ left: `calc(${Math.max(0, Math.min(1, threshold)) * 100}% - 1px)`, height: height + 6 }}
        />
      )}
    </div>
  );
}
