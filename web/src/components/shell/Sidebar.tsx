import clsx from "clsx";
import { motion } from "framer-motion";
import { Activity, BrainCircuit, CandlestickChart, Gauge, LayoutDashboard, MessagesSquare, ScanSearch, Swords } from "lucide-react";
import { NavLink, useLocation } from "react-router-dom";
import { useId } from "react";
import { useRoundSummary } from "@/lib/api";
import { fmtNum } from "@/lib/format";
import { useSelection } from "@/lib/selection";
import { PERSONAS, PERSONA_META } from "@/lib/theme";
import { PersonaMark } from "../ui";
import { Wordmark } from "./Logo";
import { useScopeLabel } from "./ScopeSwitch";

export const NAV = [
  { to: "/", label: "Command", icon: LayoutDashboard, hint: "Run brief" },
  { to: "/markets", label: "Markets", icon: CandlestickChart, hint: "Prices, flow and microstructure" },
  { to: "/surveillance", label: "Surveillance", icon: ScanSearch, hint: "Detector cases and evidence" },
  { to: "/models", label: "Model Lab", icon: Gauge, hint: "Detection performance per round" },
  { to: "/adversary", label: "Adversary", icon: Swords, hint: "Strategist mutations and outcomes" },
  { to: "/analyst", label: "Analyst", icon: MessagesSquare, hint: "Ask the run's logs" },
];

export function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const { search, run, round, scopeMarket } = useSelection();
  // Unique per instance: the desktop sidebar stays mounted while the phone drawer is open.
  const lid = useId();
  const loc = useLocation();
  const summary = useRoundSummary(run, round, scopeMarket);
  const scope = useScopeLabel();
  const personas = summary.data?.personas ?? {};

  return (
    <aside className="flex h-full w-[248px] flex-col border-r border-ink-700 bg-ink-900/90 backdrop-blur">
      <div className="px-5 pb-5 pt-5">
        <Wordmark />
      </div>

      <nav className="px-3">
        <div className="eyebrow px-2 pb-2">Workspace</div>
        <ul className="space-y-0.5">
          {NAV.map((n) => {
            const active = n.to === "/" ? loc.pathname === "/" : loc.pathname.startsWith(n.to);
            return (
              <li key={n.to}>
                <NavLink
                  to={{ pathname: n.to, search }}
                  onClick={onNavigate}
                  title={n.hint}
                  className={clsx(
                    "focus-ring group relative flex items-center gap-3 rounded-lg px-2.5 py-2 text-[13px] font-medium transition-colors",
                    active ? "text-fg" : "text-fg-muted hover:bg-tint/[0.03] hover:text-fg-2",
                  )}
                >
                  {active && (
                    <motion.span
                      layoutId={`nav-active-${lid}`}
                      className="absolute inset-0 rounded-lg border border-ink-600 bg-ink-800"
                      transition={{ type: "spring", stiffness: 480, damping: 38 }}
                    />
                  )}
                  {active && (
                    <motion.span layoutId={`nav-bar-${lid}`} className="absolute left-0 top-2 h-[calc(100%-16px)] w-[2px] rounded-full bg-accent" />
                  )}
                  <n.icon size={16} className={clsx("relative z-10", active ? "text-accent" : "")} strokeWidth={1.9} />
                  <span className="relative z-10">{n.label}</span>
                </NavLink>
              </li>
            );
          })}
        </ul>
      </nav>

      <div className="mt-6 px-3">
        <div className="flex items-center justify-between px-2 pb-2">
          <span className="eyebrow">R{round ?? "-"} detector · {scope}</span>
          <Activity size={12} className="text-fg-muted" />
        </div>
        <div className="space-y-1.5">
          {PERSONAS.filter((p) => personas[p]).map((p) => {
            const m = personas[p];
            const f1 = m.f1 ?? 0;
            return (
              <motion.div
                key={p}
                layout
                className="rounded-lg border border-ink-700 bg-ink-850/70 px-2.5 py-2"
                title={`Precision ${fmtNum(m.precision)} · Recall ${fmtNum(m.recall)} · FPR ${fmtNum(m.fpr, 3)}`}
              >
                <div className="flex items-center justify-between">
                  <span className="flex items-center gap-1.5 text-[12px] text-fg-2">
                    <PersonaMark persona={p} /> {PERSONA_META[p].label}
                  </span>
                  <span className="num font-mono text-[12px] text-fg">{fmtNum(m.f1, 3)}</span>
                </div>
                <div className="mt-1.5 h-1 rounded-full bg-tint/[0.05]">
                  <motion.div
                    className="h-full rounded-full"
                    style={{ background: PERSONA_META[p].color }}
                    initial={{ width: 0 }}
                    animate={{ width: `${f1 * 100}%` }}
                    transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
                  />
                </div>
                <div className="mt-1.5 flex gap-3 font-mono text-[10.5px] text-fg-muted">
                  <span>P {fmtNum(m.precision)}</span>
                  <span>R {fmtNum(m.recall)}</span>
                  <span>FPR {fmtNum(m.fpr, 3)}</span>
                </div>
              </motion.div>
            );
          })}
          {!Object.keys(personas).length && <div className="px-2 text-[12px] text-fg-muted">No metrics for this round.</div>}
        </div>
      </div>

      <div className="mt-auto border-t border-ink-700 px-5 py-4">
        <div className="flex items-start gap-2 text-[11px] leading-relaxed text-fg-muted">
          <BrainCircuit size={14} className="mt-0.5 shrink-0" />
          <span>Synthetic agent-based market. No real accounts, trades or regulatory findings.</span>
        </div>
      </div>
    </aside>
  );
}
