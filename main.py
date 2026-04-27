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
import pandas as pd
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
    "NVDA": 0.2507,
    "AVGO": 0.1294,
    "MRVL": 0.1148,
    "MPWR": 0.0571,
    "NXPI": 0.0568,
    "ON":   0.0476,
    "GFS":  0.0430,
    "LRCX": 0.0423,
    "ASML": 0.0412,
    "MU":   0.0402,
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

def _fetch_single_ticker(ticker: str) -> Optional[Tuple[float, float]]:
    """
    Fetch (prev_close, current_price) for one ticker.

    Strategy (in order of preference):
      1. fast_info — single quote API call, fastest & most reliable
      2. yf.download daily bars — falls back to last 2 closing prices

    Returns None only if both methods fail.
    """
    import math

    def _is_valid(*vals):
        for v in vals:
            if v is None:
                return False
            try:
                f = float(v)
                if math.isnan(f) or f <= 0:
                    return False
            except (TypeError, ValueError):
                return False
        return True

    # ── Method 1: fast_info (single quote API call) ──────────────────────────
    try:
        t = yf.Ticker(ticker)
        fi = t.fast_info

        # fast_info exposes both attribute and dict-style access; try both.
        prev = getattr(fi, "regular_market_previous_close", None) or \
               getattr(fi, "previous_close", None)
        cur  = getattr(fi, "last_price", None)

        if _is_valid(prev, cur):
            return (float(prev), float(cur))
        else:
            log.warning("%s: fast_info returned invalid values prev=%s cur=%s",
                        ticker, prev, cur)
    except Exception as e:
        log.warning("%s: fast_info error — %s", ticker, e)

    # ── Method 2: yf.download daily bars (fallback) ──────────────────────────
    try:
        df = yf.download(
            ticker,
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if df is not None and not df.empty:
            # Handle both flat and multi-level column DataFrames
            if isinstance(df.columns, pd.MultiIndex):
                if ("Close", ticker) in df.columns:
                    closes = df[("Close", ticker)]
                else:
                    closes = df.xs("Close", axis=1, level=0).iloc[:, 0]
            else:
                closes = df["Close"]

            closes = closes.dropna()
            if len(closes) >= 2 and _is_valid(closes.iloc[-2], closes.iloc[-1]):
                log.info("%s: using daily-bar fallback", ticker)
                return (float(closes.iloc[-2]), float(closes.iloc[-1]))
    except Exception as e:
        log.warning("%s: download fallback error — %s", ticker, e)

    return None


def fetch_prices(tickers: list) -> Dict[str, Tuple[float, float]]:
    """
    Fetch all tickers in parallel using a thread pool.
    One ticker's failure can't break the others.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: Dict[str, Tuple[float, float]] = {}

    with ThreadPoolExecutor(max_workers=min(len(tickers), 10)) as pool:
        future_to_ticker = {pool.submit(_fetch_single_ticker, t): t for t in tickers}
        try:
            for future in as_completed(future_to_ticker, timeout=25):
                ticker = future_to_ticker[future]
                try:
                    val = future.result()
                    if val:
                        results[ticker] = val
                        log.info("✅ %-5s prev=%.4f cur=%.4f Δ=%+.2f%%",
                                 ticker, val[0], val[1],
                                 100 * (val[1] - val[0]) / val[0])
                    else:
                        log.warning("❌ %s: no price data", ticker)
                except Exception as e:
                    log.warning("❌ %s: future error — %s", ticker, e)
        except TimeoutError:
            log.error("Price fetch timed out at 25s — partial results returned")

    log.info("Price fetch summary: %d/%d OK", len(results), len(tickers))
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
    missing = []

    for ticker, weight in holdings.items():
        if ticker not in prices:
            missing.append(ticker)
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
            f"No valid price data for any of the {len(holdings)} tickers: "
            f"{', '.join(holdings.keys())}. "
            f"Yahoo Finance may be rate-limiting the server. "
            f"Try /diagnostics for details."
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


@app.get("/diagnostics")
async def diagnostics():
    """Diagnose price-fetching health for each ticker."""
    await refresh_holdings_if_needed()

    tickers = list(holdings_cache.keys())
    if not tickers:
        return JSONResponse({"error": "no holdings loaded"}, status_code=503)

    loop = asyncio.get_event_loop()
    prices = await loop.run_in_executor(None, fetch_prices, tickers)

    report = []
    for ticker in tickers:
        if ticker in prices:
            prev, cur = prices[ticker]
            report.append({
                "ticker": ticker,
                "status": "✅ OK",
                "prev_close": round(prev, 4),
                "current": round(cur, 4),
                "return_pct": round(100 * (cur - prev) / prev, 3),
            })
        else:
            report.append({"ticker": ticker, "status": "❌ FAILED", "prev_close": None, "current": None})

    return JSONResponse({
        "yfinance_version": yf.__version__,
        "tickers_total": len(tickers),
        "tickers_ok": len(prices),
        "tickers_failed": len(tickers) - len(prices),
        "report": report,
        "as_of": datetime.now(timezone.utc).isoformat(),
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
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover"/>
<title>FSELX · NAV Estimator</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;600;700;900&display=swap" rel="stylesheet"/>
<style>
/* ── Reset & base ─────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:          #ffffff;
  --surface:     #f5f7f5;
  --surface2:    #eaf2ec;
  --border:      #d4e2d6;
  --border2:     #b8cfbb;
  --accent:      #00a650;
  --accent2:     #007a3d;
  --green:       #00a650;
  --red:         #d0021b;
  --amber:       #b85c00;
  --text:        #1a2e1c;
  --text-muted:  #4a6b4e;
  --text-dim:    #8aab8e;
  --mono:        'Share Tech Mono', monospace;
  --display:     'Barlow Condensed', sans-serif;
}

html { background: var(--bg); color: var(--text); font-family: var(--display); min-height: 100vh; width: 100%; overflow-x: hidden; }
body { min-height: 100vh; width: 100%; display: flex; flex-direction: column; overflow-x: hidden; }

/* ── Scanline overlay (removed for light theme) ───────────────────── */

/* ── Subtle dot grid background ───────────────────────────────────── */
.grid-bg {
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image: radial-gradient(circle, rgba(0,166,80,.12) 1px, transparent 1px);
  background-size: 24px 24px;
}

/* ── Layout ───────────────────────────────────────────────────────── */
.wrapper {
  position: relative; z-index: 1;
  width: 100%; max-width: 540px; margin: 0 auto;
  padding: 20px 16px 60px;
  box-sizing: border-box;
  display: flex; flex-direction: column; gap: 16px;
}

/* ── Header ───────────────────────────────────────────────────────── */
header {
  display: flex; flex-direction: column; align-items: flex-start;
  border-bottom: 1px solid var(--border); padding-bottom: 16px;
}
.fund-badge {
  font-family: var(--mono); font-size: 11px; letter-spacing: .2em;
  color: var(--accent); background: rgba(0,166,80,.08);
  border: 1px solid rgba(0,166,80,.3); border-radius: 3px;
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
.input-wrap:focus-within { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(0,166,80,.15); }
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
  background: var(--accent); color: #ffffff;
  font-family: var(--display); font-size: 15px; font-weight: 700;
  letter-spacing: .06em; text-transform: uppercase; white-space: nowrap;
  transition: all .15s; position: relative; overflow: hidden;
}
button#go-btn::after {
  content: ''; position: absolute; inset: 0;
  background: rgba(0,0,0,0); transition: background .15s;
}
button#go-btn:hover::after { background: rgba(0,0,0,.1); }
button#go-btn:active { transform: scale(.97); }
button#go-btn:disabled { opacity: .5; cursor: not-allowed; }

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
.qa-btn:hover { border-color: var(--accent); color: var(--accent); background: rgba(0,166,80,.07); }

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
.nav-return.up   { background: rgba(0,166,80,.1);  color: var(--green); }
.nav-return.down { background: rgba(208,2,27,.08); color: var(--red); }
.nav-return.flat { background: rgba(184,92,0,.08); color: var(--amber); }

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
  width: 100%; max-width: 100%;
}
.holdings-header {
  padding: 12px 16px; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 6px;
}
.holdings-title {
  font-size: 10px; font-weight: 600; letter-spacing: .18em; text-transform: uppercase; color: var(--text-muted);
}
.holdings-source {
  font-family: var(--mono); font-size: 9px; color: var(--text-dim);
  background: var(--bg); border: 1px solid var(--border); border-radius: 3px;
  padding: 2px 6px; max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

.table-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; width: 100%; }
table { width: 100%; border-collapse: collapse; min-width: 340px; }
thead th {
  font-family: var(--mono); font-size: 9px; letter-spacing: .12em; text-transform: uppercase;
  color: var(--text-dim); padding: 8px 12px; text-align: left;
  border-bottom: 1px solid var(--border); background: var(--bg);
}
thead th:not(:first-child) { text-align: right; }
tbody tr { border-bottom: 1px solid var(--border); transition: background .1s; }
tbody tr:last-child { border-bottom: none; }
tbody tr:hover { background: rgba(0,166,80,.04); }
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
@media(max-width:480px){
  .wrapper { padding: 14px 12px 48px; gap: 12px; }
  .input-row { flex-wrap: nowrap; }
  #investment { font-size: 18px; }
  button#go-btn { font-size: 13px; padding: 0 14px; }
  .nav-value { font-size: clamp(28px, 9vw, 42px); }
  .stat-value { font-size: 17px; }
  thead th, tbody td { padding: 7px 8px; font-size: 11px; }
}
@media(max-width:360px){
  .stats-grid { grid-template-columns: 1fr; }
  .wrapper { padding: 12px 10px 40px; }
  button#go-btn { font-size: 12px; padding: 0 10px; }
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
      <button id="go-btn" onclick="runEstimate()">Update Estimate</button>
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
      <div class="table-scroll">
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
    btn.innerHTML = 'Update Estimate';
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
