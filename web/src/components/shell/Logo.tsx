import { motion } from "framer-motion";

/** Mark: a "V" drawn as a price path dipping and recovering, with a verification tick above it. */
export function LogoMark({ size = 30 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden>
      <rect width="32" height="32" rx="8" style={{ fill: "var(--bg-2)", stroke: "var(--line-2)" }} />
      <motion.path
        d="M7.5 9 L16 24 L24.5 9"
        fill="none"
        stroke="var(--accent)"
        strokeWidth={2.8}
        strokeLinecap="round"
        strokeLinejoin="round"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 1.1, ease: [0.65, 0, 0.35, 1] }}
      />
      <motion.circle cx="16" cy="6.2" r="1.6" fill="var(--accent)" initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 0.9, type: "spring" }} />
    </svg>
  );
}

export function Wordmark() {
  return (
    <div className="flex items-center gap-2.5">
      <LogoMark />
      <div className="leading-none">
        <div className="text-[15px] font-semibold tracking-tight text-fg">Veridex</div>
        <div className="mt-1 text-[9.5px] font-medium uppercase tracking-[0.2em] text-fg-muted">Market integrity</div>
      </div>
    </div>
  );
}
