"""
FSELX Intraday NAV Estimator
Production-ready FastAPI app with live holdings scraping, fallback, and price data.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import httpx
import yfinance as yf
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
FSELX_PREV_NAV = 27.12          # Update this daily or fetch from Fidelity
CACHE_TTL_HOURS = 24
PRICE_CACHE_TTL_SECONDS = 60    # Re-fetch prices at most every 60s

FALLBACK_HOLDINGS: Dict[str, float] = {
    "NVDA": 0.18,
    "AMD":  0.10,
    "AVGO": 0.09,
    "TSM":  0.08,
    "ASML": 0.07,
    "AMAT": 0.06,
    "LRCX": 0.05,
    "KLAC": 0.04,
    "MU":   0.03,
    "ADI":  0.03,
}

# ── In-memory cache ──────────────────────────────────────────────────────────
holdings_cache: Dict[str, float] = {}
holdings_source: str = "uninitialized"
holdings_timestamp: Optional[float] = None

price_cache: Dict[str, Tuple[float, float]] = {}   # ticker → (prev_close, current)
price_cache_ts: float = 0.0

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="FSELX NAV Estimator", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# HOLDINGS FETCHING
# ══════════════════════════════════════════════════════════════════════════════

async def fetch_fidelity_holdings() -> Dict[str, float]:
    """Scrape top 10 holdings from Fidelity's fund page."""
    url = "https://fundresearch.fidelity.com/mutual-funds/composition/316390590"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://fundresearch.fidelity.com/",
    }

    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Fidelity renders holdings in a table with class containing "equity-holdings"
    holdings: Dict[str, float] = {}

    # Try multiple selector patterns Fidelity has used
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            texts = [c.get_text(strip=True) for c in cells]
            # Look for ticker-like cell followed by a percentage
            for i, text in enumerate(texts):
                if text and len(text) <= 5 and text.isupper() and i + 1 < len(texts):
                    pct_text = texts[i + 1].replace("%", "").strip()
                    try:
                        pct = float(pct_text)
                        if 0.1 < pct < 50:
                            holdings[text] = pct / 100.0
                    except ValueError:
                        pass
            if len(holdings) >= 10:
                break
        if len(holdings) >= 10:
            break

    if len(holdings) < 5:
        raise ValueError(f"Only found {len(holdings)} holdings from Fidelity page — too few")

    # Keep top 10 by weight
    top10 = dict(sorted(holdings.items(), key=lambda x: x[1], reverse=True)[:10])
    log.info("✅ Fidelity scrape success: %s", list(top10.keys()))
    return top10


async def fetch_yahoo_holdings() -> Dict[str, float]:
    """Fetch FSELX top holdings via Yahoo Finance fund profile page."""
    url = "https://finance.yahoo.com/quote/FSELX/holdings/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    holdings: Dict[str, float] = {}

    # Yahoo Finance uses a table for top holdings
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for row in rows[1:]:  # skip header
            cells = row.find_all("td")
            if len(cells) >= 2:
                ticker_cell = cells[0].get_text(strip=True)
                pct_cell = cells[-1].get_text(strip=True).replace("%", "")
                if ticker_cell and len(ticker_cell) <= 6:
                    try:
                        pct = float(pct_cell)
                        if 0.1 < pct < 50:
                            holdings[ticker_cell] = pct / 100.0
                    except ValueError:
                        pass

    # Also try JSON embedded in page (Yahoo sometimes embeds data)
    for script in soup.find_all("script"):
        txt = script.string or ""
        if "topHoldings" in txt and "holdingPercent" in txt:
            try:
                start = txt.find('"topHoldings"')
                chunk = txt[start:start + 2000]
                import re
                tickers = re.findall(r'"symbol"\s*:\s*"([A-Z]{1,6})"', chunk)
                pcts = re.findall(r'"holdingPercent"\s*:\s*([\d.]+)', chunk)
                for t, p in zip(tickers, pcts):
                    holdings[t] = float(p)
            except Exception:
                pass

    if len(holdings) < 5:
        raise ValueError(f"Only {len(holdings)} holdings from Yahoo — too few")

    top10 = dict(sorted(holdings.items(), key=lambda x: x[1], reverse=True)[:10])
    log.info("✅ Yahoo Finance scrape success: %s", list(top10.keys()))
    return top10


async def load_holdings() -> Tuple[Dict[str, float], str]:
    """
    Try live sources in order; fall back to hardcoded data.
    Returns (holdings_dict, source_label).
    """
    # 1. Fidelity
    try:
        h = await fetch_fidelity_holdings()
        return h, "Fidelity Investments (live)"
    except Exception as e:
        log.warning("Fidelity fetch failed: %s", e)

    # 2. Yahoo Finance
    try:
        h = await fetch_yahoo_holdings()
        return h, "Yahoo Finance (live)"
    except Exception as e:
        log.warning("Yahoo Finance fetch failed: %s", e)

    # 3. Hardcoded fallback
    log.warning("All live sources failed — using hardcoded fallback holdings")
    return FALLBACK_HOLDINGS.copy(), "Fallback (hardcoded — live sources unavailable)"


async def refresh_holdings_if_needed():
    """Refresh holdings cache if stale or empty."""
    global holdings_cache, holdings_source, holdings_timestamp

    now = time.time()
    if holdings_timestamp and (now - holdings_timestamp) < CACHE_TTL_HOURS * 3600:
        return  # still fresh

    log.info("Refreshing FSELX holdings cache…")
    h, src = await load_holdings()
    holdings_cache = h
    holdings_source = src
    holdings_timestamp = now
    log.info("Holdings cached from: %s", src)


# ══════════════════════════════════════════════════════════════════════════════
# PRICE FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def fetch_prices(tickers: list) -> Dict[str, Tuple[float, float]]:
    """
    Returns {ticker: (prev_close, current_price)} for each ticker.
    Works during market hours AND when market is closed (weekends/after-hours).
    Strategy:
      - Try 1-minute intraday bars (period=5d) for live session
      - Fall back to daily bars (period=5d) when market is closed
    """
    results: Dict[str, Tuple[float, float]] = {}

    def _parse_df(df, ticker):
        """Extract (prev_close, current) from a dataframe."""
        if df is None or df.empty:
            return None
        df = df.dropna(subset=["Close"])
        if len(df) < 2:
            return None

        df.index = df.index.tz_convert("UTC")
        today = datetime.now(timezone.utc).date()
        today_rows = df[df.index.date == today]
        prev_rows  = df[df.index.date < today]

        if not today_rows.empty and not prev_rows.empty:
            # Market is open — normal intraday case
            prev_close = float(prev_rows["Close"].iloc[-1])
            current    = float(today_rows["Close"].iloc[-1])
        elif today_rows.empty and not prev_rows.empty:
            # Market closed today (weekend / after-hours / holiday)
            # Use last two available closing prices
            days = sorted(set(prev_rows.index.date))
            if len(days) < 2:
                return None
            day_n   = days[-1]   # most recent session
            day_n1  = days[-2]   # session before that
            prev_close = float(prev_rows[prev_rows.index.date == day_n1]["Close"].iloc[-1])
            current    = float(prev_rows[prev_rows.index.date == day_n]["Close"].iloc[-1])
        else:
            # Fallback: just use last two rows
            prev_close = float(df["Close"].iloc[-2])
            current    = float(df["Close"].iloc[-1])

        return (prev_close, current)

    # ── Attempt 1: 1-minute intraday bars ────────────────────────────────────
    try:
        data = yf.download(
            tickers,
            period="5d",
            interval="1m",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        for ticker in tickers:
            try:
                df = data if len(tickers) == 1 else (
                    data[ticker] if ticker in data.columns.get_level_values(0) else None
                )
                parsed = _parse_df(df, ticker)
                if parsed:
                    results[ticker] = parsed
                else:
                    log.warning("1m: no usable data for %s", ticker)
            except Exception as e:
                log.warning("1m parse error for %s: %s", ticker, e)
    except Exception as e:
        log.warning("1m download failed: %s", e)

    # ── Attempt 2: daily bars for any tickers that failed ────────────────────
    missing = [t for t in tickers if t not in results]
    if missing:
        log.info("Falling back to daily bars for: %s", missing)
        try:
            data_d = yf.download(
                missing,
                period="5d",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            for ticker in missing:
                try:
                    df = data_d if len(missing) == 1 else (
                        data_d[ticker] if ticker in data_d.columns.get_level_values(0) else None
                    )
                    parsed = _parse_df(df, ticker)
                    if parsed:
                        results[ticker] = parsed
                        log.info("Daily fallback OK for %s: prev=%.2f cur=%.2f", ticker, *parsed)
                    else:
                        log.warning("Daily: no usable data for %s", ticker)
                except Exception as e:
                    log.warning("Daily parse error for %s: %s", ticker, e)
        except Exception as e:
            log.error("Daily download failed: %s", e)

    return results


async def get_prices(tickers: list) -> Dict[str, Tuple[float, float]]:
    """Async wrapper with short-lived cache."""
    global price_cache, price_cache_ts

    now = time.time()
    if now - price_cache_ts < PRICE_CACHE_TTL_SECONDS and all(t in price_cache for t in tickers):
        return price_cache

    loop = asyncio.get_event_loop()
    prices = await loop.run_in_executor(None, fetch_prices, tickers)
    price_cache = prices
    price_cache_ts = now
    return prices


# ══════════════════════════════════════════════════════════════════════════════
# NAV CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def calculate_nav(
    holdings: Dict[str, float],
    prices: Dict[str, Tuple[float, float]],
    prev_nav: float,
) -> Tuple[float, float, list]:
    """
    Returns (estimated_nav, adjusted_return_pct, holding_details).
    """
    weighted_return = 0.0
    total_weight = 0.0
    details = []

    for ticker, weight in holdings.items():
        if ticker not in prices:
            log.warning("Skipping %s — no price data", ticker)
            continue

        prev_close, current = prices[ticker]
        if prev_close == 0:
            continue

        ret = (current - prev_close) / prev_close
        weighted_return += weight * ret
        total_weight += weight

        details.append({
            "ticker": ticker,
            "weight": round(weight * 100, 2),
            "prev_close": round(prev_close, 4),
            "current": round(current, 4),
            "return_pct": round(ret * 100, 3),
            "contribution_pct": round(weight * ret * 100, 4),
        })

    if total_weight == 0:
        raise ValueError(
            "No valid price data — if the market is closed, try again during "
            "US market hours (Mon–Fri 9:30am–4pm ET). "
            "Weekend estimates use the most recent two closing prices."
        )

    adjusted_return = weighted_return / total_weight
    nav_est = prev_nav * (1 + adjusted_return)

    return nav_est, adjusted_return * 100, details


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    log.info("🚀 FSELX NAV Estimator starting up…")
    await refresh_holdings_if_needed()


# ══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/holdings")
async def get_holdings():
    """Return current holdings used for NAV estimation."""
    await refresh_holdings_if_needed()
    ts = datetime.fromtimestamp(holdings_timestamp, tz=timezone.utc).isoformat() if holdings_timestamp else None
    return JSONResponse({
        "source": holdings_source,
        "last_updated": ts,
        "holdings": {k: round(v * 100, 2) for k, v in holdings_cache.items()},
        "count": len(holdings_cache),
    })


@app.get("/estimate")
async def estimate(investment: float = Query(..., gt=0, description="Investment amount in USD")):
    """Estimate FSELX intraday NAV and user P&L."""
    await refresh_holdings_if_needed()

    if not holdings_cache:
        raise HTTPException(status_code=503, detail="Holdings data unavailable")

    tickers = list(holdings_cache.keys())
    try:
        prices = await get_prices(tickers)
    except Exception as e:
        log.error("Price fetch error: %s", e)
        raise HTTPException(status_code=502, detail=f"Price data unavailable: {e}")

    try:
        nav_est, return_pct, details = calculate_nav(holdings_cache, prices, FSELX_PREV_NAV)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    shares = investment / FSELX_PREV_NAV
    estimated_value = shares * nav_est
    dollar_gain = estimated_value - investment

    ts = datetime.fromtimestamp(holdings_timestamp, tz=timezone.utc).isoformat() if holdings_timestamp else None

    return JSONResponse({
        "fund": "FSELX",
        "prev_nav": FSELX_PREV_NAV,
        "estimated_nav": round(nav_est, 4),
        "return_pct": round(return_pct, 3),
        "investment": round(investment, 2),
        "estimated_value": round(estimated_value, 2),
        "dollar_gain": round(dollar_gain, 2),
        "shares": round(shares, 4),
        "holdings_source": holdings_source,
        "holdings_last_updated": ts,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "holdings_detail": details,
    })


# ══════════════════════════════════════════════════════════════════════════════
# FRONTEND  (served from /)
# ══════════════════════════════════════════════════════════════════════════════

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0"/>
<title>FSELX · NAV Estimator</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;600;700;900&display=swap" rel="stylesheet"/>
<style>
/* ── Reset & base ─────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:          #090b0f;
  --surface:     #0d1117;
  --surface2:    #121820;
  --border:      #1e2a38;
  --border2:     #243040;
  --accent:      #00d4ff;
  --accent2:     #0091cc;
  --green:       #00e5a0;
  --red:         #ff4560;
  --amber:       #ffb800;
  --text:        #e8f0f8;
  --text-muted:  #5a7a9a;
  --text-dim:    #3a5570;
  --mono:        'Share Tech Mono', monospace;
  --display:     'Barlow Condensed', sans-serif;
}

html { background: var(--bg); color: var(--text); font-family: var(--display); min-height: 100vh; }
body { min-height: 100vh; display: flex; flex-direction: column; overflow-x: hidden; }

/* ── Scanline overlay ─────────────────────────────────────────────── */
body::before {
  content: '';
  position: fixed; inset: 0; z-index: 999; pointer-events: none;
  background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,.06) 2px, rgba(0,0,0,.06) 4px);
}

/* ── Animated grid background ─────────────────────────────────────── */
.grid-bg {
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image:
    linear-gradient(rgba(0,212,255,.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,255,.03) 1px, transparent 1px);
  background-size: 40px 40px;
  animation: gridpan 60s linear infinite;
}
@keyframes gridpan { from{background-position:0 0} to{background-position:40px 40px} }

/* ── Layout ───────────────────────────────────────────────────────── */
.wrapper {
  position: relative; z-index: 1;
  max-width: 540px; margin: 0 auto;
  padding: 20px 16px 60px;
  display: flex; flex-direction: column; gap: 16px;
}

/* ── Header ───────────────────────────────────────────────────────── */
header {
  display: flex; flex-direction: column; align-items: flex-start;
  border-bottom: 1px solid var(--border); padding-bottom: 16px;
}
.fund-badge {
  font-family: var(--mono); font-size: 11px; letter-spacing: .2em;
  color: var(--accent); background: rgba(0,212,255,.08);
  border: 1px solid rgba(0,212,255,.25); border-radius: 3px;
  padding: 3px 10px; margin-bottom: 8px;
}
header h1 {
  font-size: clamp(28px, 8vw, 44px); font-weight: 900; letter-spacing: -.02em;
  line-height: 1; color: var(--text);
  text-transform: uppercase;
}
header h1 span { color: var(--accent); }
.subtitle {
  font-size: 13px; font-weight: 300; letter-spacing: .08em;
  color: var(--text-muted); margin-top: 4px; text-transform: uppercase;
}

/* ── Status bar ───────────────────────────────────────────────────── */
.status-bar {
  display: flex; align-items: center; gap: 8px;
  font-family: var(--mono); font-size: 10px; color: var(--text-dim);
  border: 1px solid var(--border); border-radius: 4px;
  padding: 8px 12px; background: var(--surface);
}
.status-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--text-dim); flex-shrink: 0;
  transition: background .3s;
}
.status-dot.live { background: var(--green); box-shadow: 0 0 6px var(--green); animation: pulse 2s infinite; }
.status-dot.error { background: var(--red); }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
#status-text { flex: 1; color: var(--text-muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ── Input card ───────────────────────────────────────────────────── */
.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 20px;
}
.card-label {
  font-size: 10px; font-weight: 600; letter-spacing: .18em;
  text-transform: uppercase; color: var(--text-muted);
  margin-bottom: 10px; display: block;
}

.input-row { display: flex; gap: 10px; align-items: stretch; }
.input-wrap {
  flex: 1; position: relative;
  border: 1px solid var(--border2); border-radius: 5px;
  background: var(--bg); overflow: hidden;
  transition: border-color .2s;
}
.input-wrap:focus-within { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(0,212,255,.12); }
.input-prefix {
  position: absolute; left: 12px; top: 50%; transform: translateY(-50%);
  font-family: var(--mono); font-size: 16px; color: var(--text-muted);
  pointer-events: none;
}
#investment {
  width: 100%; height: 52px; background: transparent; border: none; outline: none;
  color: var(--text); font-family: var(--mono); font-size: 20px;
  padding: 0 14px 0 30px;
}
#investment::placeholder { color: var(--text-dim); }

button#go-btn {
  height: 52px; padding: 0 20px; border: none; border-radius: 5px; cursor: pointer;
  background: var(--accent); color: var(--bg);
  font-family: var(--display); font-size: 15px; font-weight: 700;
  letter-spacing: .06em; text-transform: uppercase; white-space: nowrap;
  transition: all .15s; position: relative; overflow: hidden;
}
button#go-btn::after {
  content: ''; position: absolute; inset: 0;
  background: rgba(255,255,255,0); transition: background .15s;
}
button#go-btn:hover::after { background: rgba(255,255,255,.12); }
button#go-btn:active { transform: scale(.97); }
button#go-btn:disabled { opacity: .5; cursor: not-allowed; }
button#go-btn .btn-icon { margin-right: 4px; }

/* ── Quick amounts ────────────────────────────────────────────────── */
.quick-amounts {
  display: flex; gap: 6px; margin-top: 10px; flex-wrap: wrap;
}
.qa-btn {
  font-family: var(--mono); font-size: 11px; padding: 4px 10px;
  background: transparent; border: 1px solid var(--border2);
  border-radius: 3px; color: var(--text-muted); cursor: pointer;
  transition: all .15s;
}
.qa-btn:hover { border-color: var(--accent); color: var(--accent); background: rgba(0,212,255,.06); }

/* ── Results ──────────────────────────────────────────────────────── */
#results { display: none; flex-direction: column; gap: 12px; }
#results.visible { display: flex; }

/* Big NAV display */
.nav-display {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 24px 20px 20px;
  position: relative; overflow: hidden;
}
.nav-display::before {
  content: 'FSELX';
  position: absolute; right: 16px; top: 16px;
  font-family: var(--mono); font-size: 10px; letter-spacing: .2em;
  color: var(--text-dim);
}
.nav-label { font-size: 10px; font-weight: 600; letter-spacing: .18em; text-transform: uppercase; color: var(--text-muted); }
.nav-value {
  font-family: var(--mono); font-size: clamp(36px, 10vw, 52px);
  color: var(--text); line-height: 1.1; margin: 6px 0 4px;
  letter-spacing: -.02em;
}
.nav-return {
  font-family: var(--mono); font-size: 14px; font-weight: 600;
  display: inline-flex; align-items: center; gap: 5px;
  padding: 3px 10px; border-radius: 3px; letter-spacing: .05em;
}
.nav-return.up   { background: rgba(0,229,160,.1); color: var(--green); }
.nav-return.down { background: rgba(255,69,96,.1);  color: var(--red); }
.nav-return.flat { background: rgba(255,184,0,.1);  color: var(--amber); }

/* ── Stats grid ───────────────────────────────────────────────────── */
.stats-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
}
.stat-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 14px 16px;
}
.stat-label { font-size: 9px; font-weight: 600; letter-spacing: .18em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 6px; }
.stat-value { font-family: var(--mono); font-size: 20px; color: var(--text); letter-spacing: -.01em; }
.stat-value.up   { color: var(--green); }
.stat-value.down { color: var(--red); }

/* ── Holdings table ───────────────────────────────────────────────── */
.holdings-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; overflow: hidden;
}
.holdings-header {
  padding: 12px 16px; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
}
.holdings-title {
  font-size: 10px; font-weight: 600; letter-spacing: .18em; text-transform: uppercase; color: var(--text-muted);
}
.holdings-source {
  font-family: var(--mono); font-size: 9px; color: var(--text-dim);
  background: var(--bg); border: 1px solid var(--border); border-radius: 3px;
  padding: 2px 6px; max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

table { width: 100%; border-collapse: collapse; }
thead th {
  font-family: var(--mono); font-size: 9px; letter-spacing: .12em; text-transform: uppercase;
  color: var(--text-dim); padding: 8px 12px; text-align: left;
  border-bottom: 1px solid var(--border); background: var(--bg);
}
thead th:not(:first-child) { text-align: right; }
tbody tr { border-bottom: 1px solid var(--border); transition: background .1s; }
tbody tr:last-child { border-bottom: none; }
tbody tr:hover { background: rgba(255,255,255,.02); }
tbody td { font-family: var(--mono); font-size: 12px; padding: 9px 12px; color: var(--text); }
tbody td:not(:first-child) { text-align: right; }
td.ticker { font-weight: 600; color: var(--accent); letter-spacing: .05em; }
td.up { color: var(--green); }
td.down { color: var(--red); }

/* Weight bar */
.weight-cell { display: flex; align-items: center; gap: 8px; justify-content: flex-end; }
.weight-bar {
  height: 3px; border-radius: 2px;
  background: linear-gradient(90deg, var(--accent2), var(--accent));
  min-width: 2px;
}

/* ── Timestamp footer ─────────────────────────────────────────────── */
.ts-footer {
  font-family: var(--mono); font-size: 9px; color: var(--text-dim);
  text-align: center; letter-spacing: .1em;
  padding: 8px 0 0;
}

/* ── Loading shimmer ──────────────────────────────────────────────── */
.shimmer {
  background: linear-gradient(90deg, var(--surface) 25%, var(--surface2) 50%, var(--surface) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.4s infinite;
  border-radius: 4px; height: 20px;
}
@keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }

/* ── Error banner ─────────────────────────────────────────────────── */
.error-banner {
  background: rgba(255,69,96,.08); border: 1px solid rgba(255,69,96,.3);
  border-radius: 6px; padding: 12px 16px;
  font-family: var(--mono); font-size: 12px; color: var(--red); display: none;
}
.error-banner.show { display: block; }

/* ── Disclaimer ───────────────────────────────────────────────────── */
.disclaimer {
  font-size: 10px; color: var(--text-dim); line-height: 1.6;
  border-top: 1px solid var(--border); padding-top: 12px;
  font-family: var(--mono);
}

/* ── Responsive ───────────────────────────────────────────────────── */
@media(max-width:360px){
  .stats-grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div class="grid-bg"></div>

<div class="wrapper">

  <!-- Header -->
  <header>
    <div class="fund-badge">MUTUAL FUND · SEMICONDUCTOR</div>
    <h1>FS<span>ELX</span></h1>
    <div class="subtitle">Fidelity Select Semiconductors · NAV Estimator</div>
  </header>

  <!-- Status bar -->
  <div class="status-bar">
    <div class="status-dot" id="status-dot"></div>
    <span id="status-text">Initializing…</span>
  </div>

  <!-- Input card -->
  <div class="card">
    <span class="card-label">Your Investment</span>
    <div class="input-row">
      <div class="input-wrap">
        <span class="input-prefix">$</span>
        <input id="investment" type="number" min="1" step="any" placeholder="10000" autocomplete="off" inputmode="decimal"/>
      </div>
      <button id="go-btn" onclick="runEstimate()">
        <span class="btn-icon">⚡</span> Estimate
      </button>
    </div>
    <div class="quick-amounts">
      <button class="qa-btn" onclick="setAmount(1000)">$1K</button>
      <button class="qa-btn" onclick="setAmount(5000)">$5K</button>
      <button class="qa-btn" onclick="setAmount(10000)">$10K</button>
      <button class="qa-btn" onclick="setAmount(25000)">$25K</button>
      <button class="qa-btn" onclick="setAmount(50000)">$50K</button>
      <button class="qa-btn" onclick="setAmount(100000)">$100K</button>
    </div>
  </div>

  <!-- Error -->
  <div class="error-banner" id="error-banner"></div>

  <!-- Results -->
  <div id="results">

    <!-- NAV display -->
    <div class="nav-display">
      <div class="nav-label">Estimated NAV</div>
      <div class="nav-value" id="r-nav">—</div>
      <div id="r-return" class="nav-return flat">— %</div>
    </div>

    <!-- Stats grid -->
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-label">Est. Value</div>
        <div class="stat-value" id="r-value">—</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Gain / Loss</div>
        <div class="stat-value" id="r-gain">—</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Prev. NAV</div>
        <div class="stat-value" id="r-prev">—</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Shares Est.</div>
        <div class="stat-value" id="r-shares">—</div>
      </div>
    </div>

    <!-- Holdings table -->
    <div class="holdings-card">
      <div class="holdings-header">
        <span class="holdings-title">Top 10 Holdings · Live Weights</span>
        <span class="holdings-source" id="holdings-source-badge">—</span>
      </div>
      <table>
        <thead>
          <tr>
            <th>Ticker</th>
            <th>Weight</th>
            <th>Price</th>
            <th>Return</th>
            <th>Contrib.</th>
          </tr>
        </thead>
        <tbody id="holdings-tbody"></tbody>
      </table>
    </div>

    <div class="ts-footer" id="r-timestamp"></div>

    <div class="disclaimer">
      ⚠ ESTIMATED ONLY · Not financial advice · NAV computed from top-10 holdings weights only ·
      Actual FSELX NAV may differ · Data from yfinance &amp; public sources · Prices delayed
    </div>

  </div><!-- /results -->

</div><!-- /wrapper -->

<script>
const $ = id => document.getElementById(id);
const fmt  = (n, d=2) => n.toLocaleString('en-US', {minimumFractionDigits:d, maximumFractionDigits:d});
const fmtD = n => (n>=0?'+':'')+fmt(n,2);

function setStatus(text, state='idle') {
  $('status-text').textContent = text;
  const dot = $('status-dot');
  dot.className = 'status-dot' + (state==='live'?' live':state==='error'?' error':'');
}

function setAmount(v) {
  $('investment').value = v;
  $('investment').focus();
}

function setError(msg) {
  const b = $('error-banner');
  b.textContent = '⚠ ' + msg;
  b.classList.add('show');
}

function clearError() {
  $('error-banner').classList.remove('show');
}

async function runEstimate() {
  const inv = parseFloat($('investment').value);
  if (!inv || inv <= 0) {
    setError('Please enter a valid investment amount.');
    return;
  }
  clearError();

  const btn = $('go-btn');
  btn.disabled = true;
  btn.innerHTML = '<span style="display:inline-block;animation:spin .7s linear infinite">⏳</span>';

  setStatus('Fetching live prices…', 'idle');

  try {
    const res = await fetch(`/estimate?investment=${inv}`);
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Server error');
    }
    const d = await res.json();
    renderResults(d, inv);
    setStatus('Live · Updated ' + new Date(d.as_of).toLocaleTimeString(), 'live');
  } catch(e) {
    setError(e.message || 'Request failed. Check connection.');
    setStatus('Error', 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">⚡</span> Estimate';
  }
}

function renderResults(d, inv) {
  const r = d.return_pct;
  const isUp   = r > 0.005;
  const isDown = r < -0.005;
  const cls    = isUp ? 'up' : isDown ? 'down' : 'flat';
  const arrow  = isUp ? '▲' : isDown ? '▼' : '─';

  $('results').classList.add('visible');

  // NAV
  $('r-nav').textContent = '$' + fmt(d.estimated_nav, 4);
  const retEl = $('r-return');
  retEl.textContent = `${arrow} ${Math.abs(r).toFixed(3)}%`;
  retEl.className = 'nav-return ' + cls;

  // Stats
  const valEl   = $('r-value');
  const gainEl  = $('r-gain');
  valEl.textContent  = '$' + fmt(d.estimated_value);
  gainEl.textContent = (d.dollar_gain >= 0 ? '+$' : '-$') + fmt(Math.abs(d.dollar_gain));
  valEl.className  = 'stat-value ' + cls;
  gainEl.className = 'stat-value ' + cls;
  $('r-prev').textContent   = '$' + fmt(d.prev_nav, 4);
  $('r-shares').textContent = fmt(d.shares, 4);

  // Source badge
  const src = d.holdings_source || '—';
  $('holdings-source-badge').textContent = src.replace('(live)', '').trim();
  $('holdings-source-badge').title = src;

  // Holdings table
  const tbody = $('holdings-tbody');
  tbody.innerHTML = '';
  const maxW = Math.max(...d.holdings_detail.map(h => h.weight));
  d.holdings_detail.forEach(h => {
    const ret = h.return_pct;
    const hCls = ret > 0.05 ? 'up' : ret < -0.05 ? 'down' : '';
    const barW = Math.round((h.weight / maxW) * 80);
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="ticker">${h.ticker}</td>
      <td>
        <div class="weight-cell">
          <div class="weight-bar" style="width:${barW}px"></div>
          ${h.weight.toFixed(1)}%
        </div>
      </td>
      <td>$${fmt(h.current,2)}</td>
      <td class="${hCls}">${fmtD(ret)}%</td>
      <td class="${hCls}">${h.contribution_pct >= 0 ? '+' : ''}${h.contribution_pct.toFixed(3)}%</td>
    `;
    tbody.appendChild(tr);
  });

  // Timestamp
  const ts = new Date(d.as_of);
  $('r-timestamp').textContent =
    `DATA AS OF ${ts.toLocaleDateString()} ${ts.toLocaleTimeString()} · HOLDINGS: ${d.holdings_last_updated ? new Date(d.holdings_last_updated).toLocaleDateString() : '—'}`;
}

// Auto-run on Enter
$('investment').addEventListener('keydown', e => { if(e.key==='Enter') runEstimate(); });

// Init status
(async () => {
  try {
    const res = await fetch('/holdings');
    const d = await res.json();
    setStatus(`Holdings loaded · ${d.source.replace('(live)','').trim()} · ${d.count} tickers`, 'live');
  } catch(e) {
    setStatus('Could not reach server', 'error');
  }
})();

// CSS spin keyframe
const style = document.createElement('style');
style.textContent = '@keyframes spin{to{transform:rotate(360deg)}}';
document.head.appendChild(style);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=HTML)
