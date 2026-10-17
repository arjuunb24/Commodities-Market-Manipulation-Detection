# Running the Veridex web app

The React web app (`web/`) shows the same runs as the Streamlit dashboard: markets, detector cases, model performance, the adversarial strategist and the analyst chat. A FastAPI server (`api/`) reads the run outputs in `data/runs/` and also serves the built app, so in normal use there is only one process and one address.

This guide covers three ways to run it:

1. [On this PC only](#1-on-this-pc-only)
2. [On your private network](#2-on-your-private-network-wi-fi) (phones and laptops on the same Wi-Fi)
3. [On the public internet](#3-on-the-public-internet-cloudflare-tunnel) through a Cloudflare tunnel

All commands are for Windows and run from the project folder (`Veridex\`). The Streamlit dashboard is separate and unaffected.

---

## Before you start (once)

**You need:**

- Python 3.11+ with the project virtual environment in `.venv\`
- Node.js 18 or newer (`node -v` to check)
- At least one completed run in `data/runs/` (create one with `python scripts/run_full_pipeline.py`)
- For the analyst chat and AI case notes: an LLM key in `.env` (copy `.env.example` to `.env` and fill in `CEREBRAS_API_KEY` or `GEMINI_API_KEY`). The rest of the app works without one.

**Install the dependencies:**

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
cd web
npm install
cd ..
```

---

## 1. On this PC only

### Quick way: the start/stop script

```powershell
scripts\webapp start
```

Open **http://localhost:8000**.

The script builds the app the first time (if `web\dist` is missing), then runs the server hidden in the background. Closing the terminal does **not** stop it. Use:

```powershell
scripts\webapp status    # ON or OFF, plus the addresses
scripts\webapp stop      # turn it off
scripts\webapp restart   # stop, then start again (use this to change options)
```

| Option | What it does |
|---|---|
| `-Build` | Rebuild the React app before starting (after you change files in `web\src`) |
| `-Port 8010` | Use another port, if 8000 is taken |
| `-Password <pw>` | Ask visitors for a password (see [Password](#password)) |
| `-Lan` | Also accept devices on your Wi-Fi ([section 2](#2-on-your-private-network-wi-fi)) |
| `-Public` | Also open an internet link ([section 3](#3-on-the-public-internet-cloudflare-tunnel)) |

Logs are in `.webapp\server.log` and `.webapp\server.err.log`, a folder git ignores.

### Manual way: run the server yourself

```powershell
cd web
npm run build
cd ..
.\.venv\Scripts\python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open **http://localhost:8000**. Press `Ctrl+C` to stop.

### Developing the frontend (live reload)

Use two terminals:

```powershell
# Terminal 1: the API, restarted automatically when Python files change
.\.venv\Scripts\python -m uvicorn api.main:app --reload --port 8000

# Terminal 2: the React dev server, which forwards /api to port 8000
cd web
npm run dev
```

Open **http://localhost:5173**. Edits in `web\src` appear immediately. Run `npm run build` when you are done so port 8000 serves the new version.

---

## 2. On your private network (Wi-Fi)

Lets phones, tablets and other computers on the **same network** open the app.

```powershell
scripts\webapp start -Lan -Password <pw>
```

The script prints a **Wi-Fi** address such as `http://192.168.1.23:8000`. Open it on the other device.

- **Windows Firewall:** the first time, Windows may ask whether Python can accept connections. Allow it for **Private networks**. If you missed the prompt and other devices cannot connect, go to *Windows Security > Firewall & network protection > Allow an app through firewall* and tick *Private* for Python.
- **Network type:** your Wi-Fi must be set to *Private* in Windows (Settings > Network & internet > your network). On *Public* networks Windows blocks incoming connections.
- **Password:** optional on a home network, but recommended on shared networks such as offices or university Wi-Fi.
- **Address changes:** the IP address can change when your router restarts. Run `scripts\webapp status` to see the current one.

Manual equivalent:

```powershell
.\.venv\Scripts\python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## 3. On the public internet (Cloudflare tunnel)

Gives you an `https://….trycloudflare.com` link that anyone can open from anywhere, with no router or firewall changes. It uses a free Cloudflare *quick tunnel*: your PC makes an outgoing connection to Cloudflare, which forwards visitors to it. No Cloudflare account is needed.

### Install cloudflared (once)

```powershell
winget install --id Cloudflare.cloudflared
```

Then **open a new terminal** so Windows finds the new command.

### Start it

```powershell
scripts\webapp start -Public -Password <pick-a-real-password>
```

After a few seconds it prints:

```
Veridex web app: ON
  This PC : http://localhost:8000
  Internet: https://random-words-here.trycloudflare.com
  Password: required (any user name)
```

Share the **Internet** link and the password. Stop it with `scripts\webapp stop`, and the link stops working at once.

### Things to know

- **A password is required.** The script refuses `-Public` without one. Without it, anyone who found the link could use the analyst chat and spend your LLM credit.
- **The link changes every time** you start the tunnel. For a permanent address you need a free Cloudflare account, a domain and a *named tunnel*. See Cloudflare's [named tunnel guide](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/).
- **It only works while your PC is on**, awake and running the app. Turn off sleep for long sessions.
- **Analyst chat** works on the public link. Each answer takes about 10–40 seconds. Quick tunnels give up after about 100 seconds, so very broad questions can fail; ask about one round or market at a time.
- **All visitors share your PC's resources** and your LLM budget (capped by `cerebras_budget` in `configs/llm_config.yaml`).
- **Testing it:** open the link on your phone with **Wi-Fi turned off**, so the request really comes from outside your network.

Manual equivalent, in two terminals:

```powershell
$env:VERIDEX_PASSWORD = "<pw>"
.\.venv\Scripts\python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

cloudflared tunnel --url http://127.0.0.1:8000
```

The link appears in the `cloudflared` output.

---

## Password

When the server has a password, the browser shows a login box. Type **anything** as the user name and the password as the password. The browser remembers it until it is closed.

- With the script: `-Password <pw>`
- Manually: set `VERIDEX_PASSWORD` before starting uvicorn
- No password set: the app is open to anyone who can reach it (fine for `localhost`)

The password covers the whole app, including the data API and the analyst chat. It is HTTP Basic authentication: the tunnel link is HTTPS, so it is encrypted there, but on plain-HTTP Wi-Fi addresses it is not encrypted, so do not reuse an important password.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Port 8000 is already in use` | Another server is running (often a manual uvicorn). Stop it, run `scripts\webapp stop`, or use `-Port 8010`. |
| Yellow banner: "The API server is running older code" | The server started before the code changed. Run `scripts\webapp restart` (or restart uvicorn), then press `Ctrl+F5`. |
| Page shows an old version after an update | Rebuild with `scripts\webapp restart -Build`, then `Ctrl+F5`. On phones, close and reopen the tab. |
| "No adversarial runs yet" | There is nothing in `data/runs/`. Run the pipeline first. |
| Top bar says "AI offline" | No LLM key in `.env`, or the budget is used up. Hover over the label for the reason. The rest of the app still works. |
| `cloudflared is not installed` | Install it (see above) and open a new terminal. |
| Tunnel starts but prints no link | Check `.webapp\tunnel.log`. Usually a network or firewall block on outgoing connections. |
| Other devices cannot reach the Wi-Fi address | Check the firewall and network-type notes in [section 2](#2-on-your-private-network-wi-fi). |
| `running scripts is disabled on this system` | Use `scripts\webapp` (the `.cmd` wrapper), not `scripts\webapp.ps1`. |
| Server fails to start | Read `.webapp\server.err.log`. The script also prints its last lines. |

---

## How it fits together

```
browser ──► uvicorn :8000 ──► api/main.py ──► data/runs/<run>/...   (parquet, metrics.json)
               │                   └────────► src/agentic/...         (analyst chat, case notes)
               └── serves web/dist/ (the built React app)

-Lan     binds 0.0.0.0, so other devices on the network can connect
-Public  cloudflared ──► Cloudflare edge ──► https://….trycloudflare.com
```

| Path | What it is |
|---|---|
| `web/` | React + TypeScript source (Vite, Tailwind, ECharts) |
| `web/dist/` | The built app (git-ignored; created by `npm run build`) |
| `api/` | FastAPI server |
| `scripts/webapp.ps1`, `scripts/webapp.cmd` | Start/stop script |
| `.webapp/` | Script state and logs (git-ignored) |
