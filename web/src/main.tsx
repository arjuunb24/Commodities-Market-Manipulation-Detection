import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MotionConfig } from "framer-motion";
import { StrictMode, Suspense, lazy, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Layout } from "./components/shell/Layout";
import { Skeleton } from "./components/ui";
import { SelectionProvider } from "./lib/selection";
import { initTheme } from "./lib/themes";
import "./styles.css";

initTheme();

const Command = lazy(() => import("./pages/Command"));
const Markets = lazy(() => import("./pages/Markets"));
const Surveillance = lazy(() => import("./pages/Surveillance"));
const ModelLab = lazy(() => import("./pages/ModelLab"));
const Adversary = lazy(() => import("./pages/Adversary"));
const Analyst = lazy(() => import("./pages/Analyst"));
const NotFound = lazy(() => import("./pages/NotFound"));

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 5 * 60_000, refetchOnWindowFocus: false, retry: 1 } },
});

const Fallback = () => (
  <div className="space-y-4">
    <Skeleton className="h-10 w-72" />
    <Skeleton className="h-72" />
  </div>
);

/** Each lazy page suspends inside the shell, so the navigation never flashes. */
const Page = ({ el }: { el: ReactNode }) => <Suspense fallback={<Fallback />}>{el}</Suspense>;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <MotionConfig reducedMotion="user">
        <BrowserRouter>
          <SelectionProvider>
            <Routes>
              <Route element={<Layout />}>
                <Route index element={<Page el={<Command />} />} />
                <Route path="markets" element={<Page el={<Markets />} />} />
                <Route path="surveillance" element={<Page el={<Surveillance />} />} />
                <Route path="models" element={<Page el={<ModelLab />} />} />
                <Route path="adversary" element={<Page el={<Adversary />} />} />
                <Route path="analyst" element={<Page el={<Analyst />} />} />
                <Route path="*" element={<Page el={<NotFound />} />} />
              </Route>
            </Routes>
          </SelectionProvider>
        </BrowserRouter>
      </MotionConfig>
    </QueryClientProvider>
  </StrictMode>,
);
