"""
FSELX Intraday NAV Estimator
Production-ready FastAPI app with live holdings scraping, fallback, and price data.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

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
FSELX_NAV_FALLBACK = 57.86      # Used only if live NAV fetch fails
NAV_CACHE_TTL_SECONDS = 4 * 3600   # Mutual funds price once/day at 4pm ET
CACHE_TTL_HOURS = 24
PRICE_CACHE_TTL_SECONDS = 60    # Re-fetch prices at most every 60s

# SEC EDGAR — N-PORT-P filing source (quarterly portfolio holdings, public ~60-day lag)
# SEC fair-use policy requires a User-Agent with real contact info.
# Override via Railway env var: SEC_USER_AGENT="YourApp your-email@example.com"
SEC_USER_AGENT  = os.environ.get("SEC_USER_AGENT", "FSELX-NAV-Estimator contact@example.com")
EDGAR_SEARCH    = "https://efts.sec.gov/LATEST/search-index"
SEC_ARCHIVES    = "https://www.sec.gov/Archives/edgar/data"
FSELX_SEARCH_Q  = '"Fidelity Select Semiconductors"'

# Holdings cap — top N by weight kept for performance
MAX_HOLDINGS = 30

# ── Company name → ticker mapping ────────────────────────────────────────────
# N-PORT XML uses legal names ("NVIDIA CORP"), not tickers.
# Mapping covers the universe a semiconductor sector fund typically holds.
NAME_TO_TICKER: Dict[str, str] = {
    # ── Top FSELX semiconductor names ──
    "NVIDIA CORP": "NVDA",
    "BROADCOM INC": "AVGO",
    "MARVELL TECHNOLOGY INC": "MRVL",
    "MONOLITHIC POWER SYSTEMS INC": "MPWR",
    "NXP SEMICONDUCTORS NV": "NXPI",
    "ON SEMICONDUCTOR CORP": "ON",
    "ONSEMI": "ON",
    "GLOBALFOUNDRIES INC": "GFS",
    "LAM RESEARCH CORP": "LRCX",
    "ASML HOLDING NV": "ASML",
    "ASML HOLDING N V": "ASML",
    "MICRON TECHNOLOGY INC": "MU",
    "ADVANCED MICRO DEVICES INC": "AMD",
    "TAIWAN SEMICONDUCTOR MANUFACTURING CO LTD": "TSM",
    "TAIWAN SEMICONDUCTOR MFG CO LTD": "TSM",
    "APPLIED MATERIALS INC": "AMAT",
    "KLA CORP": "KLAC",
    "ANALOG DEVICES INC": "ADI",
    "INTEL CORP": "INTC",
    "TEXAS INSTRUMENTS INC": "TXN",
    "QUALCOMM INC": "QCOM",
    "ARM HOLDINGS PLC": "ARM",
    "ARM HOLDINGS PLC ADR": "ARM",
    "MICROCHIP TECHNOLOGY INC": "MCHP",
    "SYNOPSYS INC": "SNPS",
    "CADENCE DESIGN SYSTEMS INC": "CDNS",
    "ENTEGRIS INC": "ENTG",
    "ALLEGRO MICROSYSTEMS INC": "ALGM",
    "SKYWORKS SOLUTIONS INC": "SWKS",
    "QORVO INC": "QRVO",
    "ASE TECHNOLOGY HOLDING CO LTD": "ASX",
    "UNITED MICROELECTRONICS CORP": "UMC",
    "TOWER SEMICONDUCTOR LTD": "TSEM",
    "AMKOR TECHNOLOGY INC": "AMKR",
    "POWER INTEGRATIONS INC": "POWI",
    "RAMBUS INC": "RMBS",
    "CIRRUS LOGIC INC": "CRUS",
    "SILICON LABORATORIES INC": "SLAB",
    "LATTICE SEMICONDUCTOR CORP": "LSCC",
    "WOLFSPEED INC": "WOLF",
    "FORMFACTOR INC": "FORM",
    "ONTO INNOVATION INC": "ONTO",
    "AXCELIS TECHNOLOGIES INC": "ACLS",
    "ASTERA LABS INC": "ALAB",
    "CREDO TECHNOLOGY GROUP HOLDING LTD": "CRDO",
    "MACOM TECHNOLOGY SOLUTIONS HOLDINGS INC": "MTSI",
    "DIODES INC": "DIOD",
    "NAVITAS SEMICONDUCTOR CORP": "NVTS",
    "SITIME CORP": "SITM",
    "PHOTRONICS INC": "PLAB",
    "VEECO INSTRUMENTS INC": "VECO",
    "ULTRA CLEAN HOLDINGS INC": "UCTT",
    "IMPINJ INC": "PI",
    "INDIE SEMICONDUCTOR INC": "INDI",
    "ICHOR HOLDINGS LTD": "ICHR",
    "MKS INSTRUMENTS INC": "MKSI",
    "RENESAS ELECTRONICS CORP": "6723.T",
    "MEDIATEK INC": "2454.TW",
    "ARISTA NETWORKS INC": "ANET",
    "INTERNATIONAL BUSINESS MACHINES CORP": "IBM",
}

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
holdings_metadata: dict = {}

price_cache: Dict[str, Tuple[float, float]] = {}   # ticker → (prev_close, current)
price_cache_ts: float = 0.0

# FSELX previous NAV cache
fselx_nav_cache: Optional[float] = None
fselx_nav_source: str = "uninitialized"
fselx_nav_ts: float = 0.0

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
    url = "https://fundresearch.fidelity.com/mutual-funds/composition/316390863"
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


# ══════════════════════════════════════════════════════════════════════════════
# SEC N-PORT FETCHER (full holdings, quarterly)
# ══════════════════════════════════════════════════════════════════════════════

def _name_to_ticker(name: str) -> Optional[str]:
    """Map a security name (from N-PORT) to a ticker. Returns None if no match."""
    if not name:
        return None
    upper = name.strip().upper()
    # Strip trailing legal suffixes that vary across filings
    for suffix in [" - CLASS A", " CLASS A", " ADR", " - ADR", " SPONSORED"]:
        if upper.endswith(suffix):
            upper = upper[:-len(suffix)].strip()

    # Direct match
    if upper in NAME_TO_TICKER:
        return NAME_TO_TICKER[upper]

    # Bidirectional substring match (handles minor wording differences)
    for known_name, ticker in NAME_TO_TICKER.items():
        if known_name in upper or upper in known_name:
            return ticker

    # Token overlap match: if the first 2 significant words match, accept it
    upper_tokens = [t for t in upper.split() if len(t) > 2]
    for known_name, ticker in NAME_TO_TICKER.items():
        known_tokens = [t for t in known_name.split() if len(t) > 2]
        if len(upper_tokens) >= 2 and len(known_tokens) >= 2:
            if upper_tokens[:2] == known_tokens[:2]:
                return ticker
    return None


async def fetch_sec_nport_holdings() -> Tuple[Dict[str, float], dict]:
    """
    Fetch FSELX's full holdings list from the most recent SEC N-PORT-P filing.
    Returns ({ticker: weight_decimal}, metadata_dict).

    Strategy:
      1. Use EDGAR full-text search to locate the most recent NPORT-P
         filing for "Fidelity Select Semiconductors".
      2. Fetch the primary_doc.xml from EDGAR archives.
      3. Parse <invstOrSec> entries → name, CUSIP, pctVal.
      4. Map each name to a ticker via NAME_TO_TICKER + fuzzy matching.
    """
    headers = {
        "User-Agent": SEC_USER_AGENT,
        "Accept": "application/json",
        "Host": "efts.sec.gov",
    }

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        # 1. Search EDGAR full-text for FSELX's recent N-PORT-P filings
        params = {"q": FSELX_SEARCH_Q, "forms": "NPORT-P"}
        resp = await client.get(EDGAR_SEARCH, params=params, headers=headers)
        resp.raise_for_status()
        search_data = resp.json()

        hits = search_data.get("hits", {}).get("hits", [])
        if not hits:
            raise ValueError("No N-PORT-P filings found in EDGAR")

        latest = hits[0]
        src = latest.get("_source", {})
        accession = src.get("adsh") or latest.get("_id", "").split(":")[0]
        ciks      = src.get("ciks", [])
        file_date = src.get("file_date") or src.get("period_of_report", "unknown")

        if not accession or not ciks:
            raise ValueError("EDGAR result missing accession/CIK")

        cik_int = int(ciks[0])
        accession_clean = accession.replace("-", "")

        # 2. Fetch the primary document (the actual portfolio XML)
        doc_url = f"{SEC_ARCHIVES}/{cik_int}/{accession_clean}/primary_doc.xml"
        doc_headers = {"User-Agent": SEC_USER_AGENT, "Accept": "application/xml"}
        doc_resp = await client.get(doc_url, headers=doc_headers)
        doc_resp.raise_for_status()
        xml_content = doc_resp.text

    # 3. Parse XML — N-PORT uses <invstOrSec> entries with <title> and <pctVal>
    soup = BeautifulSoup(xml_content, "lxml-xml")

    holdings: Dict[str, float] = {}
    unmapped: List[Tuple[str, float]] = []
    total_pct_mapped = 0.0
    total_pct_seen   = 0.0

    for sec in soup.find_all("invstOrSec"):
        title_el = sec.find("title")
        pct_el   = sec.find("pctVal")
        if not title_el or not pct_el:
            continue

        name = title_el.get_text(strip=True)
        try:
            pct = float(pct_el.get_text(strip=True))
        except ValueError:
            continue

        total_pct_seen += pct

        ticker = _name_to_ticker(name)
        if ticker:
            # Sum if multiple share classes resolve to same ticker
            holdings[ticker] = holdings.get(ticker, 0.0) + pct / 100.0
            total_pct_mapped += pct
        else:
            unmapped.append((name, pct))

    if len(holdings) < 5:
        raise ValueError(
            f"Only {len(holdings)} holdings mapped from N-PORT (saw {len(unmapped)} unmapped)"
        )

    metadata = {
        "filing_date": file_date,
        "accession": accession,
        "total_holdings_in_filing": int(len(holdings) + len(unmapped)),
        "mapped_count": len(holdings),
        "unmapped_count": len(unmapped),
        "coverage_pct": round(total_pct_mapped, 2),
        "filing_total_pct": round(total_pct_seen, 2),
        "top_unmapped": [{"name": n, "pct": p}
                         for n, p in sorted(unmapped, key=lambda x: -x[1])[:5]],
    }

    log.info("✅ N-PORT scrape: %d/%d holdings mapped, %.1f%% coverage, filed %s",
             len(holdings), len(holdings) + len(unmapped),
             total_pct_mapped, file_date)

    return holdings, metadata


async def fetch_blended_holdings() -> Tuple[Dict[str, float], dict]:
    """
    Combine N-PORT (full holdings list) with current Fidelity top-10 weights.

    This produces the most accurate holdings estimate available:
      • N-PORT contributes the long tail (positions 11+) which Fidelity's
        marketing page omits — typically 25-30% of the portfolio.
      • Fidelity's current top-10 weights override the (~3 month older)
        N-PORT weights for the largest positions, which change most.
      • Final weights are renormalized so they sum to 1.0.
    """
    nport, meta = await fetch_sec_nport_holdings()

    current_top10: Dict[str, float] = {}
    try:
        current_top10 = await fetch_fidelity_holdings()
    except Exception as e:
        log.warning("Blend: Fidelity top-10 fetch failed (%s) — using N-PORT alone", e)

    blended = dict(nport)
    overridden = []
    if current_top10:
        for ticker, weight in current_top10.items():
            if ticker in blended:
                overridden.append(ticker)
            blended[ticker] = weight  # add or override

    # Renormalize so weights sum to 1.0
    total = sum(blended.values())
    if total > 0:
        blended = {k: v / total for k, v in blended.items()}

    meta = {
        **meta,
        "blended_with_fidelity_top10": bool(current_top10),
        "weights_overridden": overridden,
        "total_holdings_after_blend": len(blended),
    }
    return blended, meta


async def load_holdings() -> Tuple[Dict[str, float], str, dict]:
    """
    Try live sources in order; fall back to hardcoded data.
    Returns (holdings_dict, source_label, metadata_dict).
    Holdings are capped at top MAX_HOLDINGS by weight for performance.
    """
    def _cap(h: Dict[str, float]) -> Dict[str, float]:
        if len(h) <= MAX_HOLDINGS:
            return h
        return dict(sorted(h.items(), key=lambda x: x[1], reverse=True)[:MAX_HOLDINGS])

    # 1. Blended (SEC N-PORT full list + current Fidelity top-10) — most accurate
    try:
        h, meta = await fetch_blended_holdings()
        label = (
            f"SEC N-PORT (filed {meta.get('filing_date', '?')}) + Fidelity top-10"
            if meta.get("blended_with_fidelity_top10")
            else f"SEC N-PORT (filed {meta.get('filing_date', '?')})"
        )
        return _cap(h), label, meta
    except Exception as e:
        log.warning("Blended/N-PORT source failed: %s", e)

    # 2. Fidelity alone (top-10, current)
    try:
        h = await fetch_fidelity_holdings()
        return _cap(h), "Fidelity Investments (live, top-10 only)", {}
    except Exception as e:
        log.warning("Fidelity fetch failed: %s", e)

    # 3. Yahoo Finance (top-10, current)
    try:
        h = await fetch_yahoo_holdings()
        return _cap(h), "Yahoo Finance (live, top-10 only)", {}
    except Exception as e:
        log.warning("Yahoo Finance fetch failed: %s", e)

    # 4. Hardcoded fallback
    log.warning("All live sources failed — using hardcoded fallback")
    return FALLBACK_HOLDINGS.copy(), "Fallback (hardcoded — live sources unavailable)", {}


async def refresh_holdings_if_needed():
    """Refresh holdings cache if stale or empty."""
    global holdings_cache, holdings_source, holdings_timestamp, holdings_metadata

    now = time.time()
    if holdings_timestamp and (now - holdings_timestamp) < CACHE_TTL_HOURS * 3600:
        return  # still fresh

    log.info("Refreshing FSELX holdings cache…")
    h, src, meta = await load_holdings()
    holdings_cache    = h
    holdings_source   = src
    holdings_metadata = meta
    holdings_timestamp = now
    log.info("Holdings cached: %d positions from %s", len(h), src)


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
# FSELX NAV FETCH (live)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_fselx_nav_sync() -> Optional[Tuple[float, str]]:
    """
    Fetch the most recent published NAV for FSELX from Yahoo via yfinance.
    Returns (nav, source_label) or None on failure.

    Mutual funds publish one NAV per day at 4pm ET.
      - Pre-4pm:  fast_info.last_price = yesterday's NAV  (= "previous NAV" we want)
      - Post-4pm: fast_info.last_price = today's NAV      (= the "previous NAV" we use to estimate tomorrow's intraday)
    Either way, last_price is the most recent published NAV — exactly what we want.
    """
    import math

    def _ok(v):
        if v is None: return False
        try:
            f = float(v)
            return not math.isnan(f) and f > 0
        except (TypeError, ValueError):
            return False

    # ── Method 1: fast_info ──────────────────────────────────────────────────
    try:
        t = yf.Ticker("FSELX")
        fi = t.fast_info
        nav = getattr(fi, "last_price", None)
        if _ok(nav):
            return float(nav), "Yahoo Finance (live)"

        prev = getattr(fi, "regular_market_previous_close", None) or \
               getattr(fi, "previous_close", None)
        if _ok(prev):
            return float(prev), "Yahoo Finance (previous_close)"
    except Exception as e:
        log.warning("FSELX fast_info error: %s", e)

    # ── Method 2: daily-bar fallback ─────────────────────────────────────────
    try:
        df = yf.download("FSELX", period="5d", interval="1d",
                         auto_adjust=True, progress=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                closes = df.xs("Close", axis=1, level=0).iloc[:, 0]
            else:
                closes = df["Close"]
            closes = closes.dropna()
            if len(closes) >= 1 and _ok(closes.iloc[-1]):
                return float(closes.iloc[-1]), "Yahoo Finance (daily bar)"
    except Exception as e:
        log.warning("FSELX download fallback error: %s", e)

    return None


async def get_fselx_prev_nav() -> Tuple[float, str]:
    """
    Returns (prev_nav, source_label). Cached for NAV_CACHE_TTL_SECONDS.
    Falls back to FSELX_NAV_FALLBACK only if the live fetch fails the very first time.
    """
    global fselx_nav_cache, fselx_nav_source, fselx_nav_ts

    now = time.time()
    if fselx_nav_cache and (now - fselx_nav_ts) < NAV_CACHE_TTL_SECONDS:
        return fselx_nav_cache, fselx_nav_source

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _fetch_fselx_nav_sync)

    if result:
        nav, src = result
        fselx_nav_cache = nav
        fselx_nav_source = src
        fselx_nav_ts = now
        log.info("✅ FSELX NAV refreshed: $%.4f from %s", nav, src)
        return nav, src

    if fselx_nav_cache:
        log.warning("FSELX NAV refresh failed — using stale cache: $%.4f", fselx_nav_cache)
        return fselx_nav_cache, fselx_nav_source + " (stale)"

    log.warning("FSELX NAV fetch failed — using hardcoded fallback: $%.4f", FSELX_NAV_FALLBACK)
    return FSELX_NAV_FALLBACK, "Fallback (hardcoded)"


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
    nav, src = await get_fselx_prev_nav()
    log.info("Initial FSELX NAV: $%.4f (%s)", nav, src)


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
        "metadata": holdings_metadata,
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
    prev_nav, nav_source = await get_fselx_prev_nav()

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
        "fselx_prev_nav": round(prev_nav, 4),
        "fselx_nav_source": nav_source,
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

    # Fetch live previous NAV in parallel with prices
    tickers = list(holdings_cache.keys())
    try:
        prev_nav, nav_source = await get_fselx_prev_nav()
        prices = await get_prices(tickers)
    except Exception as e:
        log.error("Data fetch error: %s", e)
        raise HTTPException(status_code=502, detail=f"Data unavailable: {e}")

    try:
        nav_est, return_pct, details = calculate_nav(holdings_cache, prices, prev_nav)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    shares = investment / prev_nav
    estimated_value = shares * nav_est
    dollar_gain = estimated_value - investment

    ts = datetime.fromtimestamp(holdings_timestamp, tz=timezone.utc).isoformat() if holdings_timestamp else None
    nav_ts = datetime.fromtimestamp(fselx_nav_ts, tz=timezone.utc).isoformat() if fselx_nav_ts else None

    return JSONResponse({
        "fund": "FSELX",
        "prev_nav": round(prev_nav, 4),
        "prev_nav_source": nav_source,
        "prev_nav_last_updated": nav_ts,
        "estimated_nav": round(nav_est, 4),
        "return_pct": round(return_pct, 3),
        "investment": round(investment, 2),
        "estimated_value": round(estimated_value, 2),
        "dollar_gain": round(dollar_gain, 2),
        "shares": round(shares, 4),
        "holdings_source": holdings_source,
        "holdings_last_updated": ts,
        "holdings_count": len(holdings_cache),
        "holdings_coverage_pct": holdings_metadata.get("coverage_pct"),
        "nport_filing_date": holdings_metadata.get("filing_date"),
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
        <span class="holdings-title" id="holdings-title">Holdings · Live Weights</span>
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

  // Source badge + title
  const src = d.holdings_source || '—';
  $('holdings-source-badge').textContent = src.replace('(live)', '').trim();
  $('holdings-source-badge').title = src;

  const titleEl = $('holdings-title');
  let title = `Top ${d.holdings_count} Holdings · Live Prices`;
  if (d.holdings_coverage_pct) {
    title += ` · ${d.holdings_coverage_pct.toFixed(0)}% coverage`;
  }
  titleEl.textContent = title;

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
  let footer = `DATA AS OF ${ts.toLocaleDateString()} ${ts.toLocaleTimeString()}`;
  if (d.nport_filing_date) {
    footer += ` · N-PORT FILED ${d.nport_filing_date}`;
  } else if (d.holdings_last_updated) {
    footer += ` · HOLDINGS ${new Date(d.holdings_last_updated).toLocaleDateString()}`;
  }
  $('r-timestamp').textContent = footer;
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
