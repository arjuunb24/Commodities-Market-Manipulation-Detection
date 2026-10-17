import { useQuery } from "@tanstack/react-query";
import { RotateCw } from "lucide-react";

/** API behaviour this build relies on (keep in step with API_VERSION in api/main.py). */
export const REQUIRED_API_VERSION = 3;

/**
 * Python does not reload code until the server restarts, so an old API process can keep
 * serving a newer web build: market-scoped views then silently show pooled numbers.
 */
export function ServerCheck() {
  const { data } = useQuery({
    queryKey: ["health"],
    queryFn: async () => (await fetch("/api/health")).json() as Promise<{ ok: boolean; api_version?: number }>,
    staleTime: 60_000,
    refetchInterval: 60_000,
  });
  if (!data || (data.api_version ?? 1) >= REQUIRED_API_VERSION) return null;
  return (
    <div className="flex items-center gap-3 border-b border-warn/30 bg-warn/10 px-4 py-2 text-[12.5px] text-warn sm:px-6">
      <RotateCw size={14} className="shrink-0" />
      <span>
        The API server is running older code than this page (server version {data.api_version ?? 1}, page needs {REQUIRED_API_VERSION}). Restart it so
        per-market views and other new features work:{" "}
        <code className="rounded bg-tint/10 px-1.5 py-0.5 font-mono text-[11.5px] text-fg">.\.venv\Scripts\python -m uvicorn api.main:app --port 8000</code>
      </span>
    </div>
  );
}
