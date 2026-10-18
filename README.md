# Commodities-Market-Manipulation-Detection

## Web app

A React web app (`web/`, served by the FastAPI server in `api/`) shows the run results in the browser. See [docs/WEB_APP.md](docs/WEB_APP.md) to run it on this PC, on your Wi-Fi, or publicly through a Cloudflare tunnel.

## Testing on real market data

Every adversarial round saves its models to `data/runs/<run>/round_N/models/`. `scripts/fetch_market_data.py` saves recent real 1-minute bars for the eight commodities, and `scripts/predict_live.py` scores them and checks each flagged window against the four manipulation mechanisms. See [docs/LIVE_TESTING.md](docs/LIVE_TESTING.md), including what anonymous public data can and cannot reveal.
