# Veridex web app

React front end for Veridex. It covers everything in the Streamlit dashboard, which still works and is unchanged. The app is served by a small FastAPI layer (`api/`) that reuses the project's Python code.

| View | What it shows |
|---|---|
| Command | Run brief: headline metrics, frozen-vs-retrained trend, this round's persona summary |
| Markets | All eight books (snapshot cards, indexed paths, small multiples, correlation, volume, episode mix) or one commodity (candles with VWAP and episode shading, order flow, volume profile, microstructure, returns, trader heatmap) |
| Surveillance | Detector cases checked against ground truth, filters, confidence gauges, SHAP, evidence, AI case note, raw orders and trades |
| Model Lab | Frozen vs retrained, F1 heatmap, radar, metric evolution, precision-recall, thresholds, full metrics, parameter history |
| Adversary | Strategist mutations per round: outcomes, knob changes, probe candidates, guard reverts, reasoning |
| Analyst | Questions answered from the run's logs through inspectable tool calls |

The run, round, test market, commodity and case are all kept in the URL, so any view can be shared as a link. Press `Ctrl+K` or `/` to open the run picker.

## Run it

Development mode (hot reload). Run each in its own terminal:

```powershell
.\.venv\Scripts\python -m uvicorn api.main:app --reload --port 8000
cd web; npm install; npm run dev        # http://localhost:5173 (proxies /api to :8000)
```

Production mode (a single process serves both the API and the built app):

```powershell
cd web; npm install; npm run build; cd ..
.\.venv\Scripts\python -m uvicorn api.main:app --host 0.0.0.0 --port 8000   # http://localhost:8000
```

API docs are at `/api/docs`. The AI features (case note and analyst) use the provider set in `configs/llm_config.yaml`. Without a key they show why they are unavailable, and everything else keeps working.

## Stack

Vite, React 18, TypeScript, Tailwind CSS, Framer Motion (page, list and number transitions), ECharts (all charts), TanStack Query, React Router, Lucide icons and react-markdown.

Chart colours come from a palette checked with a colour-vision validator. Personas also carry a marker shape, so their identity never depends on colour alone. Charts with a Data button switch to a table of their values.

## Hosting from this PC

`scripts\webapp` runs the built app in the background and turns it on or off:

```
scripts\webapp start                           # http://localhost:8000
scripts\webapp start -Lan                      # also other devices on the same Wi-Fi
scripts\webapp start -Public -Password <pw>    # an internet link via a Cloudflare quick tunnel
scripts\webapp status
scripts\webapp stop
```

`-Public` needs `cloudflared` (`winget install --id Cloudflare.cloudflared`) and a password,
because the analyst chat spends your LLM budget. `-Build` rebuilds `web/dist` first. Logs are in `.webapp\`.
