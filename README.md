# FSELX NAV Estimator

A production-ready web application that estimates the **intraday NAV** and **user P&L** for the **Fidelity Select Semiconductors Portfolio (FSELX)** using live top-10 holdings and real-time stock prices.

---

## Features

- **Live holdings fetch** — scrapes Fidelity → Yahoo Finance → hardcoded fallback
- **Real-time prices** via `yfinance` (1-minute bars, 2-day window)
- **NAV estimation** using weighted-return calculation
- **Mobile-first UI** with terminal/financial aesthetic
- **60-second price cache** + **24-hour holdings cache**
- **Graceful degradation** at every layer

---

## Architecture

```
GET /              → Serves the HTML/JS frontend
GET /estimate?investment=10000  → NAV + P&L calculation
GET /holdings      → Currently loaded holdings + source
```

### NAV Formula

```
return_i         = (current_price - prev_close) / prev_close
weighted_return  = Σ(weight_i × return_i)
adjusted_return  = weighted_return / Σ(weight_i)
NAV_estimated    = NAV_prev × (1 + adjusted_return)
```

---

## Local Development

```bash
cd app
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000

---

## Railway Deployment

1. Push this repo to GitHub
2. Create a new Railway project → **Deploy from GitHub repo**
3. Railway auto-detects `Procfile` → sets `PORT` env variable
4. Done — app available at your Railway URL

> **Note**: Update `FSELX_PREV_NAV` in `main.py` to the most recent official NAV daily, or add a scraper for Fidelity's published NAV.

---

## Holdings Fetch Priority

| Priority | Source | Method |
|----------|--------|--------|
| 1 | Fidelity Investments | HTML scrape |
| 2 | Yahoo Finance | HTML scrape |
| 3 | Hardcoded fallback | JSON dict |

Holdings are cached for 24 hours and refreshed on restart.

---

## File Structure

```
app/
├── main.py          # FastAPI app (backend + embedded frontend)
├── requirements.txt
├── Procfile
└── README.md
```

---

## Disclaimer

This app provides **estimates only** and is **not financial advice**. Estimated NAV is computed from the top-10 holdings only and may differ significantly from the actual published NAV. Prices are delayed. Always verify with official Fidelity data.
