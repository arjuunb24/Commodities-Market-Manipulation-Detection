"""
api/main.py
===========
Veridex web API + static host for the React app.

Dev:   .venv\\Scripts\\python -m uvicorn api.main:app --reload --port 8000
       (cd web && npm run dev)  -> http://localhost:5173 (proxies /api)
Prod:  (cd web && npm run build) then uvicorn api.main:app --port 8000
       -> http://localhost:8000 serves the built app from web/dist
"""

from __future__ import annotations

import base64
import hmac
import logging
import os
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api import bymarket, cases, data, market, performance  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("veridex.api")

app = FastAPI(title="Veridex API", version="1.0.0", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(GZipMiddleware, minimum_size=2048)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

# Optional access gate for when the app is shared beyond this PC (scripts/webapp.ps1 -Password):
# HTTP Basic auth, any user name, password from VERIDEX_PASSWORD. Unset = open (local use).
_PASSWORD = os.environ.get("VERIDEX_PASSWORD") or ""
if _PASSWORD:
    @app.middleware("http")
    async def _require_password(request: Request, call_next):
        header = request.headers.get("authorization", "")
        if header.lower().startswith("basic "):
            try:
                supplied = base64.b64decode(header[6:]).decode("utf-8").split(":", 1)[1]
            except Exception:
                supplied = ""
            if hmac.compare_digest(supplied.encode(), _PASSWORD.encode()):
                return await call_next(request)
        return Response("Password required", status_code=401, headers={"WWW-Authenticate": 'Basic realm="Veridex"'})


def _market(run: str, rnd: int, m: str | None) -> str | None:
    """Default to the round's first test market (markets are separate simulations)."""
    if m:
        return m
    ms = data.list_markets(run, rnd)
    return ms[0] if ms else None


# ── Runs & rounds ────────────────────────────────────────────────────────────

# Bump when the web app starts relying on new API behaviour; the app warns when the server is older.
API_VERSION = 3   # 2 = per-market scope (?market=); 3 = analyst context, account-level and market-comparison tools


@app.get("/api/health")
def health():
    return {"ok": True, "api_version": API_VERSION}


@app.get("/api/runs")
def runs():
    out = []
    for d in data.list_runs():
        try:
            out.append(performance.run_summary(d.name))
        except Exception as exc:   # a half-written run must not break the list
            logger.warning("run summary failed for %s: %s", d.name, exc)
            out.append({"run": d.name, "rounds": []})
    return out


_warmed: set[str] = set()


def _warm_market_metrics(run: str) -> None:
    """Compute every round x market's metrics in the background so per-market views open instantly."""
    # One worker, latest round first: parallel warmers fought the page's own requests for the GIL.
    try:
        for r in sorted(data.list_rounds(run), reverse=True):
            for m in data.list_markets(run, r):
                bymarket.market_metrics(run, r, m)
    except Exception as exc:   # warming is best effort
        logger.warning("warming per-market metrics failed for %s: %s", run, exc)


@app.get("/api/runs/{run}/rounds")
def rounds(run: str):
    if run not in _warmed:
        _warmed.add(run)
        threading.Thread(target=_warm_market_metrics, args=(run,), daemon=True).start()
    out = []
    for r in data.list_rounds(run):
        ms = []
        for m in data.list_markets(run, r):
            rng = data.tick_range(run, r, m)
            ms.append({"name": m, "tick_range": list(rng) if rng else None})
        out.append({"round": r, "markets": ms})
    return out


@app.get("/api/runs/{run}/rounds/{rnd}/summary")
def round_summary(run: str, rnd: int, market_name: str | None = Query(None, alias="market")):
    """Round metrics; with ?market= the detection and effect blocks are that test market's alone."""
    m = data.metrics(run, rnd)
    keep = ("personas", "pre_retrain", "unsupervised", "thresholds", "n_flags", "n_test_markets", "params_used",
            "mutation_source", "effectiveness", "effectiveness_ratio", "mutation_score")
    out = {k: m.get(k) for k in keep}
    if market_name:
        if market_name not in data.list_markets(run, rnd):
            raise HTTPException(404, f"test market {market_name!r} not found")
        mm = bymarket.market_metrics(run, rnd, market_name)
        out.update({k: mm.get(k) for k in ("personas", "pre_retrain", "effectiveness", "effectiveness_ratio", "mutation_score")})
        out["n_flags"] = int(data.flags_lean_for_market(run, rnd, market_name)["final_flag"].sum()) if not data.flags_lean(run, rnd).empty else None
        out.update(scope=market_name, source=mm.get("source"), frozen_by_market=bool(mm.get("has_frozen")))
    else:
        out["scope"] = "pooled"
    return data.clean(out)


# ── Market ───────────────────────────────────────────────────────────────────

@app.get("/api/runs/{run}/rounds/{rnd}/market/overview")
def market_overview(run: str, rnd: int, market_name: str | None = Query(None, alias="market")):
    return market.overview(run, rnd, _market(run, rnd, market_name))


@app.get("/api/runs/{run}/rounds/{rnd}/market/commodity/{commodity}")
def market_commodity(run: str, rnd: int, commodity: str, market_name: str | None = Query(None, alias="market")):
    return market.commodity(run, rnd, _market(run, rnd, market_name), commodity)


# ── Surveillance ─────────────────────────────────────────────────────────────

@app.get("/api/runs/{run}/rounds/{rnd}/cases")
def case_list(run: str, rnd: int, market_name: str | None = Query(None, alias="market")):
    return cases.summary(run, rnd, _market(run, rnd, market_name))


@app.get("/api/runs/{run}/rounds/{rnd}/cases/{cid}")
def case_detail(run: str, rnd: int, cid: str, market_name: str | None = Query(None, alias="market")):
    return cases.detail(run, rnd, _market(run, rnd, market_name), cid)


@app.get("/api/runs/{run}/rounds/{rnd}/cases/{cid}/raw")
def case_raw(run: str, rnd: int, cid: str, market_name: str | None = Query(None, alias="market")):
    return cases.raw(run, rnd, _market(run, rnd, market_name), cid)


@app.post("/api/runs/{run}/rounds/{rnd}/cases/{cid}/narrative")
def case_narrative(run: str, rnd: int, cid: str, market_name: str | None = Query(None, alias="market")):
    return cases.narrative(run, rnd, _market(run, rnd, market_name), cid)


# ── Performance ──────────────────────────────────────────────────────────────

@app.get("/api/runs/{run}/performance")
def perf(run: str, market_name: str | None = Query(None, alias="market")):
    return performance.performance(run, market_name)


@app.get("/api/runs/{run}/params")
def params(run: str):
    return performance.params_history(run)


@app.get("/api/runs/{run}/strategist")
def strategist(run: str, market_name: str | None = Query(None, alias="market")):
    return performance.strategist(run, market_name)


# ── AI status & analyst chat ─────────────────────────────────────────────────

@data.ttl_cache(120, maxsize=2)
def _ai_status() -> dict:
    from src.agentic.llm_client import llm_unavailable_reason
    out = {}
    for section in ("narrative_generator", "analyst_assistant"):
        reason = llm_unavailable_reason(data.llm_config(section))
        out[section] = {"available": reason is None, "reason": reason}
    return out


@app.get("/api/ai/status")
def ai_status():
    return _ai_status()


# The LLM client import takes a few seconds; do it at startup, not on the first page load.
threading.Thread(target=_ai_status, daemon=True).start()


class ChatContext(BaseModel):
    round: int | None = None
    market: str | None = Field(None, max_length=16)    # "A" / "B"
    scope: str | None = Field(None, max_length=16)     # "market" / "pooled"
    page: str | None = Field(None, max_length=40)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None
    context: ChatContext | None = None


_sessions: dict[tuple[str, str], object] = {}
_sessions_lock = threading.Lock()


@app.post("/api/runs/{run}/chat")
def chat(run: str, req: ChatRequest):
    from src.agentic.llm_client import llm_unavailable_reason
    cfg = data.llm_config("analyst_assistant")
    reason = llm_unavailable_reason(cfg)
    if reason:
        raise HTTPException(503, reason)
    rd = data.run_dir(run)
    sid = req.session_id or uuid.uuid4().hex
    with _sessions_lock:
        assistant = _sessions.get((run, sid))
        if assistant is None:
            from src.agentic.analyst_assistant import AnalystAssistant
            assistant = AnalystAssistant(cfg, run_dir=rd)
            _sessions[(run, sid)] = assistant
    try:
        result = assistant.ask(req.question, context=req.context.model_dump() if req.context else None)   # type: ignore[attr-defined]
    except Exception as exc:
        logger.warning("analyst assistant failed: %s", exc)
        raise HTTPException(502, f"The assistant could not answer: {exc}") from exc
    return data.clean({"session_id": sid, "answer": result.get("answer", ""), "sources": result.get("sources", [])})


# ── Static app (production build) ────────────────────────────────────────────

DIST = REPO_ROOT / "web" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404)
        f = DIST / path
        if path and f.is_file() and DIST in f.resolve().parents:
            return FileResponse(f)
        return FileResponse(DIST / "index.html")
