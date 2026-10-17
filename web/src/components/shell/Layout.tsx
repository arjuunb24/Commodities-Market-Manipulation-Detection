import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useState } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { useSelection } from "@/lib/selection";
import { useTheme } from "@/lib/themes";
import { ErrorBoundary } from "../ErrorBoundary";
import { ServerCheck } from "./ServerCheck";
import { EmptyState, Skeleton } from "../ui";
import { Sidebar } from "./Sidebar";
import { Ticker } from "./Ticker";
import { TopBar } from "./TopBar";

export function Layout() {
  const loc = useLocation();
  const [drawer, setDrawer] = useState(false);
  const { runs, loading, run } = useSelection();
  const theme = useTheme();
  // Charts bake colours into their options, so a theme switch remounts the views that draw
  // them (data comes straight from the query cache). The analyst chat has no charts and keeps
  // its conversation.
  const paint = theme.id;
  const pageKey = loc.pathname + (loc.pathname.startsWith("/analyst") ? "" : `~${paint}`);

  // Block bodies on purpose: newer browsers return a Promise from scrollTo, and an effect
  // that returns anything but a cleanup function crashes React on the next navigation.
  useEffect(() => {
    setDrawer(false);
  }, [loc.pathname]);
  useEffect(() => {
    window.scrollTo({ top: 0 });
  }, [loc.pathname]);

  return (
    <div className="backdrop min-h-full">
      <div className="fixed inset-y-0 left-0 z-40 hidden lg:block">
        <Sidebar key={paint} />
      </div>

      <AnimatePresence>
        {drawer && (
          <>
            <motion.div
              className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm lg:hidden"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setDrawer(false)}
            />
            <motion.div
              className="fixed inset-y-0 left-0 z-50 lg:hidden"
              initial={{ x: -260 }}
              animate={{ x: 0 }}
              exit={{ x: -260 }}
              transition={{ type: "spring", stiffness: 380, damping: 38 }}
            >
              <Sidebar key={paint} onNavigate={() => setDrawer(false)} />
            </motion.div>
          </>
        )}
      </AnimatePresence>

      <div className="relative lg:pl-[248px]">
        <TopBar onMenu={() => setDrawer(true)} />
        <ServerCheck />
        <Ticker key={paint} />
        <main className="mx-auto max-w-[1520px] px-4 pb-16 pt-6 sm:px-6 lg:px-8">
          {loading ? (
            <div className="space-y-4">
              <Skeleton className="h-10 w-72" />
              <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                {Array.from({ length: 4 }).map((_, i) => (
                  <Skeleton key={i} className="h-24" />
                ))}
              </div>
              <Skeleton className="h-80" />
            </div>
          ) : !runs.length || !run ? (
            <EmptyState title="No adversarial runs yet">
              Run the pipeline first: <code className="font-mono text-fg-2">python scripts/run_full_pipeline.py</code>. Runs appear here from
              <code className="ml-1 font-mono text-fg-2">data/runs/</code>.
            </EmptyState>
          ) : (
            // Enter-only transition. An exit animation around <Outlet> is unsafe: the outgoing
            // wrapper renders the NEW route while fading out and could stay stuck at opacity 0.
            <motion.div
              key={pageKey}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
            >
              <ErrorBoundary resetKey={loc.pathname + loc.search}>
                <Outlet />
              </ErrorBoundary>
            </motion.div>
          )}
        </main>
        <footer className="border-t border-ink-700 px-4 py-5 text-center text-[11.5px] text-fg-muted sm:px-6">
          All data is synthetic. No real trading, real accounts or real regulatory findings are represented in this application.
        </footer>
      </div>
    </div>
  );
}
