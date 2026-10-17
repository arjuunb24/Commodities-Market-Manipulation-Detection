import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import { Check, Moon, Palette, Sun } from "lucide-react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { THEMES, cycleTheme, setTheme, useTheme, type Theme } from "@/lib/themes";

/** Miniature of the theme: its page, a panel, the accent and the first data hues. */
function Preview({ t }: { t: Theme }) {
  const u = t.ui;
  const p = t.data.personas;
  const lines = [p.spoofing, p.wash_trading, p.pump_and_dump];
  return (
    <div className="relative h-[68px] overflow-hidden rounded-md" style={{ background: u.bg, border: `1px solid ${u.line}` }}>
      <div className="absolute inset-y-2 left-2 w-[18%] rounded" style={{ background: u.panel, border: `1px solid ${u.line}` }}>
        <div className="mx-1.5 mt-1.5 h-1 rounded-full" style={{ background: u.accent }} />
        <div className="mx-1.5 mt-1 h-1 rounded-full" style={{ background: u.line2 }} />
        <div className="mx-1.5 mt-1 h-1 rounded-full" style={{ background: u.line2 }} />
      </div>
      <div className="absolute inset-y-2 left-[26%] right-2 rounded" style={{ background: u.panel, border: `1px solid ${u.line}` }}>
        <svg viewBox="0 0 100 40" preserveAspectRatio="none" className="absolute inset-0 h-full w-full">
          {lines.map((c, i) => (
            <polyline
              key={i}
              fill="none"
              stroke={c}
              strokeWidth={1.6}
              vectorEffect="non-scaling-stroke"
              points={[0, 1, 2, 3, 4, 5, 6].map((x) => `${x * 16.6},${30 - i * 6 - Math.sin(x * 1.3 + i * 1.7) * 5}`).join(" ")}
            />
          ))}
        </svg>
        <div className="absolute right-1.5 top-1.5 h-2 w-5 rounded-sm" style={{ background: u.accent }} />
      </div>
    </div>
  );
}

export function ThemePicker() {
  const theme = useTheme();
  const [open, setOpen] = useState(false);
  const btn = useRef<HTMLButtonElement>(null);
  const [anchor, setAnchor] = useState({ top: 64, right: 16 });
  const [sheet, setSheet] = useState(() => typeof window !== "undefined" && window.innerWidth < 640);

  // The panel lives in a portal (the top bar's backdrop blur would otherwise trap fixed
  // positioning), anchored under the button on wide screens and as a bottom sheet on phones.
  useLayoutEffect(() => {
    if (!open) return;
    const place = () => {
      const r = btn.current?.getBoundingClientRect();
      if (r) setAnchor({ top: r.bottom + 8, right: Math.max(8, window.innerWidth - r.right) });
      setSheet(window.innerWidth < 640);
    };
    place();
    window.addEventListener("resize", place);
    return () => window.removeEventListener("resize", place);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open]);

  // Shift+T cycles themes from anywhere outside a text field.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (el.closest("input, textarea, select, [contenteditable]")) return;
      if (e.shiftKey && !e.ctrlKey && !e.metaKey && !e.altKey && e.key.toLowerCase() === "t") {
        e.preventDefault();
        cycleTheme(1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="relative">
      <button
        ref={btn}
        onClick={() => {
          // Decide the layout before the panel mounts, so its entry animation matches it.
          setSheet(window.innerWidth < 640);
          setOpen((o) => !o);
        }}
        className={clsx(
          "focus-ring flex h-8 items-center gap-2 rounded-lg border px-2.5 text-[12px] transition-colors",
          open ? "border-accent/40 bg-accent/10 text-fg" : "border-ink-600 text-fg-2 hover:border-ink-600 hover:bg-tint/[0.04] hover:text-fg",
        )}
        aria-haspopup="dialog"
        aria-expanded={open}
        title="Theme (Shift+T to cycle)"
      >
        <Palette size={14} />
        <span className="hidden xl:inline">{theme.name}</span>
        <span className="flex -space-x-1">
          {[theme.ui.accent, theme.data.personas.spoofing, theme.data.personas.wash_trading].map((c, i) => (
            <span key={i} className="h-2.5 w-2.5 rounded-full ring-2 ring-ink-950" style={{ background: c }} />
          ))}
        </span>
      </button>

      {createPortal(
      <AnimatePresence>
        {open && (
          // A tap outside lands on this backdrop and only closes the panel: it never reaches
          // the page (or a theme card) underneath.
          <motion.div
            key="theme-backdrop"
            className={clsx("fixed inset-0 z-[60]", sheet ? "bg-black/40" : "bg-transparent")}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0, pointerEvents: "none" }}
            transition={{ duration: 0.15 }}
            onClick={() => setOpen(false)}
          />
        )}
        {open && (
          <motion.div
            key="theme-panel"
            role="dialog"
            aria-label="Choose a theme"
            initial={sheet ? { y: "100%", opacity: 1 } : { opacity: 0, y: -6, scale: 0.98 }}
            animate={sheet ? { y: 0, opacity: 1 } : { opacity: 1, y: 0, scale: 1 }}
            // Closing panels stop taking input at once, so a quick tap cannot pick a theme mid-fade.
            exit={sheet ? { y: "100%", pointerEvents: "none" } : { opacity: 0, y: -6, scale: 0.98, pointerEvents: "none" }}
            transition={sheet ? { type: "spring", stiffness: 420, damping: 40 } : { duration: 0.18, ease: [0.22, 1, 0.36, 1] }}
            style={sheet ? undefined : { top: anchor.top, right: anchor.right }}
            className={clsx(
              "panel-raised fixed z-[61] p-3 shadow-panel",
              sheet ? "inset-x-0 bottom-0 rounded-b-none pb-[max(12px,env(safe-area-inset-bottom))]" : "w-[min(92vw,540px)] origin-top-right",
            )}
          >
            {sheet && <div className="mx-auto mb-2.5 h-1 w-10 rounded-full bg-ink-600" />}
            <div className="mb-2.5 flex items-baseline justify-between px-1">
              <div>
                <div className="text-[13px] font-semibold text-fg">Theme</div>
                <div className="text-[11.5px] text-fg-muted">Applies instantly and is remembered on this device.</div>
              </div>
              <span className="hidden font-mono text-[10.5px] text-fg-muted sm:inline">
                <kbd className="rounded border border-ink-600 px-1">Shift</kbd> + <kbd className="rounded border border-ink-600 px-1">T</kbd> cycles
              </span>
            </div>
            <div className="grid max-h-[62vh] grid-cols-2 gap-2 overflow-y-auto pr-0.5">
              {THEMES.map((t, i) => {
                const active = t.id === theme.id;
                return (
                  <motion.button
                    key={t.id}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.025, duration: 0.2 }}
                    onClick={() => setTheme(t.id)}
                    className={clsx(
                      "focus-ring group relative rounded-lg border p-2 text-left transition-colors",
                      active ? "border-accent/60 bg-accent/[0.07]" : "border-ink-700 hover:border-ink-600 hover:bg-tint/[0.03]",
                    )}
                    aria-pressed={active}
                  >
                    <Preview t={t} />
                    <div className="mt-2 flex items-center justify-between gap-2 px-0.5">
                      <span className="text-[12.5px] font-medium text-fg">{t.name}</span>
                      <span className="flex items-center gap-1 text-[10.5px] uppercase tracking-[0.1em] text-fg-muted">
                        {t.mode === "dark" ? <Moon size={11} /> : <Sun size={11} />}
                        {t.mode}
                      </span>
                    </div>
                    <div className="mt-0.5 px-0.5 text-[11px] leading-snug text-fg-muted">{t.tagline}</div>
                    {active && (
                      <motion.span
                        layoutId="theme-check"
                        className="absolute right-3.5 top-3.5 flex h-5 w-5 items-center justify-center rounded-full bg-accent text-accent-ink"
                        transition={{ type: "spring", stiffness: 500, damping: 34 }}
                      >
                        <Check size={12} strokeWidth={3} />
                      </motion.span>
                    )}
                  </motion.button>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>,
      document.body,
      )}
    </div>
  );
}
