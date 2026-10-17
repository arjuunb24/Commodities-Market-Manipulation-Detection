import clsx from "clsx";
import { motion } from "framer-motion";
import { Menu, Sparkles } from "lucide-react";
import { useAiStatus } from "@/lib/api";
import { marketLabel, useSelection } from "@/lib/selection";
import { tickClock } from "@/lib/format";
import { Segmented } from "../ui";
import { RunPicker } from "./RunPicker";
import { ThemePicker } from "./ThemePicker";

function RoundScrubber() {
  const { rounds, round, setRound } = useSelection();
  if (!rounds.length) return null;
  return (
    <div className="flex items-center gap-2">
      <span className="eyebrow hidden lg:inline">Round</span>
      <div className="relative flex items-center rounded-lg border border-ink-600 bg-ink-900/80 p-0.5">
        {rounds.map((r, i) => {
          const active = r.round === round;
          return (
            <button
              key={r.round}
              onClick={() => setRound(r.round)}
              className={clsx(
                "focus-ring relative h-7 min-w-[34px] rounded-md px-2 font-mono text-[12px] transition-colors",
                active ? "text-accent-ink" : "text-fg-muted hover:text-fg",
              )}
              title={i === 0 ? "Baseline round (no mutations)" : `Round ${r.round}: mutated personas`}
            >
              {active && (
                <motion.span layoutId="round-pill" className="absolute inset-0 rounded-md bg-accent" transition={{ type: "spring", stiffness: 520, damping: 36 }} />
              )}
              <span className="relative z-10 font-medium">R{r.round}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function MarketToggle() {
  const { roundInfo, market, setMarket } = useSelection();
  const markets = roundInfo?.markets ?? [];
  if (markets.length < 2) return null;
  return (
    <Segmented
      size="sm"
      value={market ?? markets[0].name}
      onChange={setMarket}
      options={markets.map((m) => ({
        value: m.name,
        label: marketLabel(m.name, markets.length),
        title: m.tick_range ? `Independent test market, session ${tickClock(m.tick_range[0])} to ${tickClock(m.tick_range[1])}` : undefined,
      }))}
    />
  );
}

function AiPill() {
  const { data } = useAiStatus();
  const ok = data?.analyst_assistant.available;
  return (
    <div
      className={clsx(
        "hidden items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-[11.5px] xl:flex",
        ok ? "border-accent/25 text-accent" : "border-ink-600 text-fg-muted",
      )}
      title={ok ? "LLM features available" : data?.analyst_assistant.reason ?? "Checking LLM availability"}
    >
      <Sparkles size={13} />
      {data ? (ok ? "AI online" : "AI offline") : "AI"}
    </div>
  );
}

export function TopBar({ onMenu }: { onMenu: () => void }) {
  return (
    <header className="sticky top-0 z-30 border-b border-ink-700 bg-ink-950/75 backdrop-blur-xl">
      <div className="flex h-14 items-center gap-3 px-4 sm:px-6">
        <button onClick={onMenu} className="focus-ring -ml-1 rounded-md p-1.5 text-fg-2 hover:bg-tint/5 lg:hidden" aria-label="Open navigation">
          <Menu size={18} />
        </button>
        <RunPicker />
        <div className="hidden h-6 w-px bg-ink-700 md:block" />
        <div className="hidden md:block">
          <RoundScrubber />
        </div>
        <div className="ml-auto flex items-center gap-2">
          <div className="hidden sm:block">
            <MarketToggle />
          </div>
          <AiPill />
          <ThemePicker />
          <span className="hidden rounded-md border border-ink-600 px-2 py-1 font-mono text-[10.5px] uppercase tracking-[0.12em] text-fg-muted 2xl:inline">
            Simulated data
          </span>
        </div>
      </div>
      <div className="flex items-center gap-3 overflow-x-auto px-4 pb-2.5 sm:px-6 md:hidden">
        <RoundScrubber />
        <MarketToggle />
      </div>
    </header>
  );
}
