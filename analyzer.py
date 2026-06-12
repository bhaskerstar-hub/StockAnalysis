"""
Stock Analysis Engine
=====================
Data sources (all free, no API key, no registration required):
  • NASDAQ Official API  — real-time OHLCV price history
  • SEC EDGAR XBRL API   — official company financials (revenue, assets, cash …)
  • FINRA API            — short interest / short volume data
"""

import time
import os
import re
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

_NASDAQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json,*/*",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}
_SEC_UA = "Mozilla/5.0 StockAnalysis chaurasiabhasker@gmail.com"

# Valid US ticker symbols: 1-6 letters, optional .X / -X share-class suffix (e.g. BRK.B)
_TICKER_RE = re.compile(r"^[A-Z]{1,6}([.\-][A-Z]{1,3})?$")

_CIK_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cik_cache.json")
_CIK_CACHE_TTL = 7 * 24 * 3600  # 1 week — SEC's ticker map changes rarely

# Full per-ticker analysis cache (budget-independent data only)
_CORE_CACHE: dict = {}
_CORE_CACHE_TTL = 300  # 5 minutes


def _request_with_retry(method: str, url: str, *, retries: int = 2, backoff: float = 0.6, **kwargs) -> requests.Response:
    """requests.request with retries on connection errors and 429/5xx responses."""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            r = requests.request(method, url, **kwargs)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(backoff * (attempt + 1))
                continue
            return r
        except requests.RequestException as e:
            last_exc = e
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
    raise last_exc


# Cached CIK lookup (loaded once per process, persisted to disk)
_CIK_MAP: dict = {}


def _load_cik_map() -> dict:
    """Load SEC EDGAR ticker→CIK mapping from disk cache, or download once and persist it."""
    global _CIK_MAP
    if _CIK_MAP:
        return _CIK_MAP

    if os.path.exists(_CIK_CACHE_FILE) and time.time() - os.path.getmtime(_CIK_CACHE_FILE) < _CIK_CACHE_TTL:
        try:
            with open(_CIK_CACHE_FILE) as f:
                _CIK_MAP = json.load(f)
            if _CIK_MAP:
                return _CIK_MAP
        except Exception:
            _CIK_MAP = {}

    try:
        r = _request_with_retry(
            "GET", "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": _SEC_UA},
            timeout=20,
        )
        if r.ok:
            for _, v in r.json().items():
                _CIK_MAP[v["ticker"].upper()] = {
                    "cik": str(v["cik_str"]).zfill(10),
                    "name": v["title"],
                }
            try:
                with open(_CIK_CACHE_FILE, "w") as f:
                    json.dump(_CIK_MAP, f)
            except Exception:
                pass
    except Exception:
        pass
    return _CIK_MAP


def _clean_num(s) -> float | None:
    """Strip $, commas, +, % from a string and return float."""
    if s is None:
        return None
    try:
        return float(str(s).replace("$", "").replace(",", "")
                     .replace("+", "").replace("%", "").strip())
    except Exception:
        return None


def _fmt_large(n) -> str:
    if n is None:
        return "N/A"
    try:
        n = float(n)
        if n >= 1e9:
            return f"${n/1e9:.2f}B"
        if n >= 1e6:
            return f"${n/1e6:.2f}M"
        if n >= 1e3:
            return f"${n/1e3:.2f}K"
        return f"${n:.2f}"
    except Exception:
        return "N/A"


# ─────────────────────────────────────────────────────────────────────────────
# NASDAQ Official API — price history + current quote
# ─────────────────────────────────────────────────────────────────────────────

def fetch_price_history(ticker: str, days: int = 400) -> pd.DataFrame:
    """
    Fetch daily OHLCV from NASDAQ's official historical data API.
    Works for all US-listed stocks (NYSE, NASDAQ, AMEX, OTC).
    """
    today = datetime.now()
    start = today - timedelta(days=days)

    def _fetch_rows(assetclass: str) -> list:
        r = _request_with_retry(
            "GET", f"https://api.nasdaq.com/api/quote/{ticker}/historical",
            params={
                "assetclass": assetclass,
                "fromdate": start.strftime("%Y-%m-%d"),
                "limit": "400",
                "todate": today.strftime("%Y-%m-%d"),
                "type": "annual",
            },
            headers=_NASDAQ_HEADERS,
            timeout=25,
        )
        r.raise_for_status()
        return ((r.json().get("data") or {}).get("tradesTable") or {}).get("rows") or []

    # NASDAQ's API requires the correct assetclass — stocks vs. ETFs return
    # null "data" when queried under the wrong one, so try both.
    rows = _fetch_rows("stocks") or _fetch_rows("etf")
    if not rows:
        raise ValueError(
            f"NASDAQ returned no price data for '{ticker}'. "
            "Verify the ticker symbol is correct."
        )

    records, dates = [], []
    for row in rows:
        try:
            date = pd.to_datetime(row["date"], format="%m/%d/%Y")
            record = {
                "Open":   _clean_num(row.get("open", 0)),
                "High":   _clean_num(row.get("high", 0)),
                "Low":    _clean_num(row.get("low", 0)),
                "Close":  _clean_num(row.get("close", 0)),
                "Volume": _clean_num(row.get("volume", 0)) or 0,
            }
        except Exception:
            continue
        dates.append(date)
        records.append(record)

    df = pd.DataFrame(records, index=pd.DatetimeIndex(dates))
    df = df.sort_index()  # chronological order
    df = df.dropna(subset=["Close"])
    df["Volume"] = df["Volume"].fillna(0).astype(float)
    return df


def fetch_current_quote(ticker: str) -> dict:
    """Fetch current price, change, volume from NASDAQ chart endpoint."""
    r = _request_with_retry(
        "GET", f"https://api.nasdaq.com/api/quote/{ticker}/chart",
        params={"assetclass": "stocks"},
        headers=_NASDAQ_HEADERS,
        timeout=15,
    )
    if not r.ok:
        return {}
    d = r.json().get("data") or {}
    return {
        "company_name": d.get("company", ticker),
        "exchange":     d.get("exchange", ""),
        "last_price":   _clean_num(d.get("lastSalePrice")),
        "net_change":   _clean_num(d.get("netChange")),
        "pct_change":   _clean_num(d.get("percentageChange")),
        "volume":       int(str(d.get("volume", "0")).replace(",", "")),
        "prev_close":   _clean_num(d.get("previousClose")),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SEC EDGAR XBRL API — official company financials
# ─────────────────────────────────────────────────────────────────────────────

def _sec_concept(facts: dict, *concepts) -> float | None:
    """
    Return the most recent USD value for any of the given XBRL concepts.
    Prefers 10-K (annual) for income/flow items, 10-Q (quarterly) for
    balance-sheet items.  Falls back to either if preferred form not found.
    """
    us_gaap = (facts.get("facts") or {}).get("us-gaap", {})
    for concept in concepts:
        entries = (us_gaap.get(concept) or {}).get("units", {}).get("USD", [])
        if not entries:
            continue
        entries_sorted = sorted(entries, key=lambda x: x.get("end", ""), reverse=True)
        # Try 10-K first (annual; income statement / cash flow)
        for form in ("10-K", "10-Q", "10-KSB", "20-F"):
            match = next((e for e in entries_sorted if e.get("form") == form), None)
            if match:
                return match["val"]
        # Fallback: any entry
        return entries_sorted[0]["val"]
    return None


def _sec_annual_pair(facts: dict, *concepts):
    """Return (recent_val, prior_val) from two consecutive 10-K filings."""
    us_gaap = (facts.get("facts") or {}).get("us-gaap", {})
    for concept in concepts:
        entries = (us_gaap.get(concept) or {}).get("units", {}).get("USD", [])
        annual = sorted(
            [e for e in entries if e.get("form") in ("10-K", "10-KSB", "20-F")],
            key=lambda x: x.get("end", ""),
        )
        if len(annual) >= 2:
            return annual[-1]["val"], annual[-2]["val"]
        if len(annual) == 1:
            return annual[-1]["val"], None
    return None, None


def fetch_sec_fundamentals(ticker: str) -> dict:
    """
    Pull financial data from SEC EDGAR's free XBRL API.
    Returns a unified dict of fundamental metrics.
    """
    cik_map = _load_cik_map()
    info = cik_map.get(ticker.upper(), {})
    cik = info.get("cik")
    if not cik:
        return {"company_name_sec": ticker, "sec_available": False}

    try:
        r = _request_with_retry(
            "GET", f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
            headers={"User-Agent": _SEC_UA},
            timeout=25,
        )
    except Exception:
        return {"company_name_sec": info.get("name", ticker), "sec_available": False}

    if not r.ok:
        return {"company_name_sec": info.get("name", ticker), "sec_available": False}

    facts = r.json()

    # Revenue (try multiple GAAP names)
    rev_now, rev_prior = _sec_annual_pair(
        facts,
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "RevenueFromContractWithCustomer",
    )

    # Balance sheet (most recent quarter)
    total_assets  = _sec_concept(facts, "Assets")
    total_liab    = _sec_concept(facts, "Liabilities")
    equity        = _sec_concept(facts, "StockholdersEquity",
                                 "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
    cash          = _sec_concept(facts, "CashAndCashEquivalentsAtCarryingValue",
                                 "CashAndCashEquivalentsPeriodIncreaseDecrease")
    long_term_debt = _sec_concept(facts, "LongTermDebt", "LongTermDebtNoncurrent")

    # Income
    net_income    = _sec_concept(facts, "NetIncomeLoss")
    gross_profit  = _sec_concept(facts, "GrossProfit")
    op_cash_flow  = _sec_concept(facts, "NetCashProvidedByUsedInOperatingActivities")

    # Shares outstanding
    shares = _sec_concept(facts, "CommonStockSharesOutstanding",
                          "CommonStockSharesIssued")
    if shares:
        shares = int(shares)

    # Derived metrics
    revenue_growth = None
    if rev_now and rev_prior and rev_prior != 0:
        revenue_growth = (rev_now - rev_prior) / abs(rev_prior)

    profit_margin = None
    if net_income and rev_now and rev_now != 0:
        profit_margin = net_income / rev_now

    gross_margin = None
    if gross_profit and rev_now and rev_now != 0:
        gross_margin = gross_profit / rev_now

    debt_to_equity = None
    if long_term_debt and equity and equity != 0:
        debt_to_equity = long_term_debt / abs(equity)

    return {
        "company_name_sec": facts.get("entityName", info.get("name", ticker)),
        "sec_available":    True,
        "cik":              cik,
        "revenue":          rev_now,
        "revenue_prior":    rev_prior,
        "revenue_growth":   revenue_growth,
        "net_income":       net_income,
        "gross_profit":     gross_profit,
        "total_assets":     total_assets,
        "total_liabilities": total_liab,
        "equity":           equity,
        "cash":             cash,
        "long_term_debt":   long_term_debt,
        "op_cash_flow":     op_cash_flow,
        "shares_outstanding": shares,
        "profit_margin":    profit_margin,
        "gross_margin":     gross_margin,
        "debt_to_equity":   debt_to_equity,
        # formatted
        "revenue_fmt":      _fmt_large(rev_now),
        "assets_fmt":       _fmt_large(total_assets),
        "cash_fmt":         _fmt_large(cash),
        "net_income_fmt":   _fmt_large(net_income),
        "debt_fmt":         _fmt_large(long_term_debt),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FINRA API — short selling / short interest data
# ─────────────────────────────────────────────────────────────────────────────

def fetch_short_interest(ticker: str) -> dict:
    """Pull short-selling data from FINRA's free public API."""
    try:
        r = _request_with_retry(
            "GET", "https://api.finra.org/data/group/otcMarket/name/regShoDaily",
            params={
                "limit": 10,
                "securities.symbolCode": ticker,
                "sortFields": ["-tradeReportDate"],
            },
            headers={"User-Agent": _SEC_UA, "Accept": "application/json"},
            timeout=15,
        )
        if not r.ok or not r.json():
            return {}

        rows = r.json()
        # Aggregate across all reporting facilities
        total, short_vol = 0, 0
        for row in rows:
            d = row.get("tradeReportDate", "")
            total    += row.get("totalParQuantity", 0)
            short_vol += row.get("shortParQuantity", 0)

        short_pct = round(short_vol / total, 4) if total > 0 else None
        return {
            "short_vol": short_vol,
            "total_vol": total,
            "short_pct": short_pct,
            "short_pct_display": f"{short_pct*100:.1f}%" if short_pct else "N/A",
            "last_date": rows[0].get("tradeReportDate", "") if rows else "",
        }
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Technical Analysis
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _safe(series: pd.Series, decimals: int = 2):
    v = series.iloc[-1]
    return round(float(v), decimals) if not (np.isnan(v) or np.isinf(v)) else None


def calculate_technical_indicators(hist: pd.DataFrame) -> dict:
    close  = hist["Close"]
    volume = hist["Volume"]
    high   = hist["High"]
    low    = hist["Low"]

    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    ema_12 = close.ewm(span=12).mean()
    ema_26 = close.ewm(span=26).mean()

    macd_line  = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9).mean()
    macd_hist   = macd_line - signal_line

    rsi = _rsi(close)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_up  = bb_mid + 2 * bb_std
    bb_dn  = bb_mid - 2 * bb_std

    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    stoch_k = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    stoch_d = stoch_k.rolling(3).mean()

    avg_vol = volume.rolling(20).mean()

    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    current = float(close.iloc[-1])
    n = len(close)

    def chg(days):
        return round((current - float(close.iloc[-(days+1)])) / float(close.iloc[-(days+1)]) * 100, 2) \
            if n > days else 0

    # Detect bullish MACD crossover in last 5 bars
    bullish_cross = False
    for i in range(1, min(6, n - 1)):
        if (macd_line.iloc[-i] > signal_line.iloc[-i] and
                macd_line.iloc[-(i+1)] <= signal_line.iloc[-(i+1)]):
            bullish_cross = True
            break

    vol_ratio = float(volume.iloc[-1] / avg_vol.iloc[-1]) if avg_vol.iloc[-1] > 0 else 1.0

    return {
        "current_price":    round(current, 2),
        "sma_20":           _safe(sma_20),
        "sma_50":           _safe(sma_50),
        "sma_200":          _safe(sma_200),
        "macd":             _safe(macd_line, 4),
        "macd_signal":      _safe(signal_line, 4),
        "macd_histogram":   _safe(macd_hist, 4),
        "macd_bullish_cross": bullish_cross,
        "rsi":              _safe(rsi, 1),
        "bb_upper":         _safe(bb_up),
        "bb_middle":        _safe(bb_mid),
        "bb_lower":         _safe(bb_dn),
        "stoch_k":          _safe(stoch_k, 1),
        "stoch_d":          _safe(stoch_d, 1),
        "volume":           int(volume.iloc[-1]),
        "avg_volume":       int(avg_vol.iloc[-1]) if not np.isnan(avg_vol.iloc[-1]) else 0,
        "volume_ratio":     round(vol_ratio, 2),
        "price_change_1d":  chg(1),
        "price_change_5d":  chg(5),
        "price_change_20d": chg(20),
        "high_52w":         round(float(high.tail(252).max()), 2),
        "low_52w":          round(float(low.tail(252).min()), 2),
        "atr_14":           _safe(atr, 4),
        "atr_pct":          round(float(atr.iloc[-1] / current * 100), 1)
                            if current > 0 and not np.isnan(atr.iloc[-1]) else 0,
    }


def calculate_technical_score(ind: dict):
    score, signals = 0, []
    rsi = ind["rsi"] or 50

    # RSI (0-25)
    if 35 <= rsi <= 55:
        score += 23; signals.append(("RSI", f"Optimal zone ({rsi:.0f}) — recovering/building momentum", "bullish"))
    elif 55 < rsi <= 70:
        score += 19; signals.append(("RSI", f"Strong momentum ({rsi:.0f})", "bullish"))
    elif 20 <= rsi < 35:
        score += 15; signals.append(("RSI", f"Oversold ({rsi:.0f}) — bounce potential", "bullish"))
    elif rsi > 70:
        score +=  8; signals.append(("RSI", f"Overbought ({rsi:.0f}) — pullback risk", "bearish"))
    else:
        score +=  5; signals.append(("RSI", f"Deeply oversold ({rsi:.0f})", "neutral"))

    # MACD (0-25)
    macd, sig, hist_v = ind.get("macd") or 0, ind.get("macd_signal") or 0, ind.get("macd_histogram") or 0
    if ind["macd_bullish_cross"]:
        score += 25; signals.append(("MACD", "Bullish crossover detected — momentum shift!", "bullish"))
    elif macd > sig and hist_v > 0:
        score += 20; signals.append(("MACD", "Bullish — MACD above signal with positive histogram", "bullish"))
    elif macd > sig:
        score += 15; signals.append(("MACD", "MACD above signal line", "bullish"))
    elif hist_v > 0:
        score += 10; signals.append(("MACD", "Histogram improving — potential crossover forming", "neutral"))
    else:
        score +=  4; signals.append(("MACD", "MACD bearish — below signal", "bearish"))

    # Moving Averages (0-25)
    cur, s20, s50, s200 = (ind["current_price"], ind.get("sma_20"),
                            ind.get("sma_50"), ind.get("sma_200"))
    if s20 and s50 and s200 and cur > s20 > s50 > s200:
        score += 25; signals.append(("Moving Averages", f"Price > SMA20 > SMA50 > SMA200 — perfect alignment", "bullish"))
    elif s20 and s50 and cur > s20 and s20 > s50:
        score += 22; signals.append(("Moving Averages", f"Price > SMA20 (${s20}) > SMA50 (${s50})", "bullish"))
    elif s20 and cur > s20:
        score += 16; signals.append(("Moving Averages", f"Price above SMA20 (${s20})", "bullish"))
    elif s50 and cur > s50:
        score += 10; signals.append(("Moving Averages", f"Above SMA50 (${s50}) but below SMA20", "neutral"))
    else:
        score +=  4; signals.append(("Moving Averages", "Price below key moving averages", "bearish"))

    # Volume + Momentum (0-25)
    vr = ind.get("volume_ratio", 1.0)
    p5 = ind.get("price_change_5d", 0)
    sk = ind.get("stoch_k") or 50
    bb_u, bb_l = ind.get("bb_upper") or cur, ind.get("bb_lower") or cur
    bb_pos = (cur - bb_l) / (bb_u - bb_l) if (bb_u - bb_l) > 0 else 0.5

    if vr > 2.0 and p5 > 0:
        vp = 15; signals.append(("Volume", f"Very high volume ({vr:.1f}x avg) on upward move — institutional buying", "bullish"))
    elif vr > 1.5 and p5 > 0:
        vp = 11; signals.append(("Volume", f"High volume ({vr:.1f}x avg) confirming uptrend", "bullish"))
    elif vr > 1.5 and p5 <= 0:
        vp =  4; signals.append(("Volume", f"High volume ({vr:.1f}x avg) on downside — distribution signal", "bearish"))
    elif vr > 1.2:
        vp =  8; signals.append(("Volume", f"Above-average volume ({vr:.1f}x avg)", "neutral"))
    else:
        vp =  5; signals.append(("Volume", f"Normal/low volume ({vr:.1f}x avg)", "neutral"))

    if sk < 20:
        mp = 10; signals.append(("Momentum", f"Stochastic oversold ({sk:.0f}) — high bounce probability", "bullish"))
    elif sk < 40 and bb_pos < 0.35:
        mp =  8; signals.append(("Momentum", f"Stochastic ({sk:.0f}) near lower Bollinger Band — accumulation zone", "bullish"))
    elif sk > 80:
        mp =  2; signals.append(("Momentum", f"Stochastic overbought ({sk:.0f}) — short-term caution", "bearish"))
    else:
        mp =  5; signals.append(("Momentum", f"Stochastic neutral ({sk:.0f})", "neutral"))

    score += vp + mp
    return min(score, 100), signals


# ─────────────────────────────────────────────────────────────────────────────
# Fundamental Scoring (SEC EDGAR + FINRA data)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_fundamental_score(sec: dict, short: dict, quote: dict, ind: dict) -> tuple:
    score, signals = 0, []
    current_price  = ind["current_price"]

    # ── Revenue Growth (0-25) ──────────────────────────────────────────────
    rg = sec.get("revenue_growth")
    if rg is not None:
        if rg > 0.40:
            score += 25; signals.append(("Revenue Growth", f"Exceptional +{rg*100:.1f}% YoY (SEC 10-K)", "bullish"))
        elif rg > 0.20:
            score += 20; signals.append(("Revenue Growth", f"Strong +{rg*100:.1f}% YoY (SEC 10-K)", "bullish"))
        elif rg > 0.05:
            score += 13; signals.append(("Revenue Growth", f"Moderate +{rg*100:.1f}% YoY (SEC 10-K)", "neutral"))
        elif rg > 0:
            score +=  8; signals.append(("Revenue Growth", f"Slow +{rg*100:.1f}% YoY (SEC 10-K)", "neutral"))
        else:
            score +=  2; signals.append(("Revenue Growth", f"Revenue declining {rg*100:.1f}% YoY (SEC 10-K)", "bearish"))
    else:
        score += 10; signals.append(("Revenue Growth", "Revenue history unavailable in EDGAR", "neutral"))

    # ── Balance Sheet Health (0-20) ────────────────────────────────────────
    cash = sec.get("cash")
    debt = sec.get("long_term_debt")
    assets = sec.get("total_assets")

    if cash and assets:
        cash_to_assets = cash / assets
        if cash_to_assets > 0.15:
            score += 20; signals.append(("Balance Sheet", f"Strong cash position (${cash/1e6:.1f}M cash, {cash_to_assets*100:.0f}% of assets)", "bullish"))
        elif cash_to_assets > 0.07:
            score += 14; signals.append(("Balance Sheet", f"Adequate cash (${cash/1e6:.1f}M, {cash_to_assets*100:.0f}% of assets)", "neutral"))
        else:
            score +=  6; signals.append(("Balance Sheet", f"Low cash relative to assets (${cash/1e6:.1f}M)", "bearish"))
    elif cash:
        score += 12; signals.append(("Balance Sheet", f"Cash: ${cash/1e6:.1f}M (EDGAR)", "neutral"))
    else:
        score += 8;  signals.append(("Balance Sheet", "Balance sheet data not yet in EDGAR", "neutral"))

    # ── Profitability (0-20) ───────────────────────────────────────────────
    pm = sec.get("profit_margin")
    gm = sec.get("gross_margin")
    ocf = sec.get("op_cash_flow")

    if pm is not None:
        if pm > 0.15:
            score += 20; signals.append(("Profitability", f"Profitable — net margin {pm*100:.1f}% (EDGAR)", "bullish"))
        elif pm > 0:
            score += 13; signals.append(("Profitability", f"Marginally profitable — net margin {pm*100:.1f}% (EDGAR)", "neutral"))
        else:
            # Burning cash — check if improving
            score +=  4
            loss_str = f"Net loss margin {pm*100:.1f}%"
            if ocf and ocf > 0:
                score += 5; signals.append(("Profitability", f"{loss_str} but positive operating cash flow", "neutral"))
            else:
                signals.append(("Profitability", f"{loss_str} (EDGAR) — monitor cash burn", "bearish"))
    elif gm is not None:
        if gm > 0.4:
            score += 14; signals.append(("Profitability", f"High gross margin {gm*100:.1f}% (EDGAR) — strong unit economics", "bullish"))
        elif gm > 0.2:
            score += 10; signals.append(("Profitability", f"Gross margin {gm*100:.1f}% (EDGAR)", "neutral"))
        else:
            score +=  5; signals.append(("Profitability", f"Thin gross margin {gm*100:.1f}% (EDGAR)", "bearish"))
    else:
        score += 8;  signals.append(("Profitability", "Income data not yet available in EDGAR", "neutral"))

    # ── Short Interest — squeeze indicator (0-20) ─────────────────────────
    sp = short.get("short_pct")
    if sp:
        if sp > 0.25:
            score += 20; signals.append(("Short Interest", f"Extreme short interest {sp*100:.1f}% of volume — major squeeze potential! (FINRA)", "bullish"))
        elif sp > 0.15:
            score += 16; signals.append(("Short Interest", f"Very high short interest {sp*100:.1f}% — squeeze candidate (FINRA)", "bullish"))
        elif sp > 0.08:
            score += 10; signals.append(("Short Interest", f"Elevated short interest {sp*100:.1f}% (FINRA)", "neutral"))
        else:
            score +=  7; signals.append(("Short Interest", f"Normal short interest {sp*100:.1f}% (FINRA)", "neutral"))
    else:
        score += 8;  signals.append(("Short Interest", "Short interest data unavailable from FINRA", "neutral"))

    # ── Valuation vs. Assets (0-15) ────────────────────────────────────────
    shares = sec.get("shares_outstanding")
    if shares and shares > 0 and current_price > 0:
        market_cap = shares * current_price
        if assets:
            pb = market_cap / assets
            if pb < 1.0:
                score += 15; signals.append(("Valuation", f"Trading below book value (P/B: {pb:.2f}) — potential undervaluation", "bullish"))
            elif pb < 3.0:
                score += 10; signals.append(("Valuation", f"Reasonable P/B ratio: {pb:.2f}", "neutral"))
            else:
                score +=  5; signals.append(("Valuation", f"High P/B ratio: {pb:.2f} (premium valuation)", "bearish"))
        else:
            score += 8; signals.append(("Valuation", f"Market Cap: {_fmt_large(market_cap)}", "neutral"))
    else:
        score += 8; signals.append(("Valuation", "Valuation data unavailable", "neutral"))

    return min(score, 100), signals


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment & Trend Scoring
# ─────────────────────────────────────────────────────────────────────────────

def calculate_sentiment_score(ind: dict, short: dict) -> tuple:
    """
    Market-sentiment proxy from short interest (crowd positioning), volume
    (attention), and recent momentum (mood) — all data we already have.
    """
    score, signals = 0, []

    # Short interest — contrarian positioning (0-35)
    sp = short.get("short_pct")
    if sp is not None:
        if sp > 0.20:
            score += 35
            signals.append(("Short Interest", f"Very high short interest {sp*100:.1f}% — heavily bearish positioning, squeeze risk (FINRA)", "bullish"))
        elif sp > 0.10:
            score += 22
            signals.append(("Short Interest", f"Elevated short interest {sp*100:.1f}% — crowd is split (FINRA)", "neutral"))
        else:
            score += 14
            signals.append(("Short Interest", f"Low short interest {sp*100:.1f}% — limited contrarian setup (FINRA)", "neutral"))
    else:
        score += 14
        signals.append(("Short Interest", "Short interest data unavailable (FINRA)", "neutral"))

    # Volume — crowd attention (0-30)
    vr = ind.get("volume_ratio", 1.0)
    p5 = ind.get("price_change_5d", 0)
    if vr > 1.5 and p5 > 0:
        score += 30
        signals.append(("Crowd Attention", f"Volume surging ({vr:.1f}x avg) alongside price gains — buying interest building", "bullish"))
    elif vr > 1.5 and p5 <= 0:
        score += 8
        signals.append(("Crowd Attention", f"Volume surging ({vr:.1f}x avg) on a falling price — selling pressure", "bearish"))
    elif vr < 0.7:
        score += 12
        signals.append(("Crowd Attention", f"Below-average volume ({vr:.1f}x avg) — low conviction either way", "neutral"))
    else:
        score += 18
        signals.append(("Crowd Attention", f"Normal trading activity ({vr:.1f}x avg)", "neutral"))

    # Momentum mood (0-35)
    p20 = ind.get("price_change_20d", 0)
    if p5 > 3 and p20 > 5:
        score += 35
        signals.append(("Momentum Mood", f"Strong recent gains (+{p5:.1f}% 5d, +{p20:.1f}% 20d) — bullish mood", "bullish"))
    elif p5 > 0 and p20 > 0:
        score += 25
        signals.append(("Momentum Mood", f"Modest recent gains (+{p5:.1f}% 5d, +{p20:.1f}% 20d)", "bullish"))
    elif p5 < -3 and p20 < -5:
        score += 5
        signals.append(("Momentum Mood", f"Sharp recent losses ({p5:.1f}% 5d, {p20:.1f}% 20d) — bearish mood", "bearish"))
    else:
        score += 15
        signals.append(("Momentum Mood", f"Mixed recent performance ({p5:.1f}% 5d, {p20:.1f}% 20d)", "neutral"))

    return min(score, 100), signals


def calculate_trend_score(ind: dict) -> tuple:
    """Longer-horizon trend: 52-week positioning, moving-average stack, and momentum consistency."""
    score, signals = 0, []
    cur = ind["current_price"]
    hi, lo = ind.get("high_52w"), ind.get("low_52w")
    s20, s50, s200 = ind.get("sma_20"), ind.get("sma_50"), ind.get("sma_200")

    # 52-week range position (0-40)
    if hi and lo and hi > lo:
        pos = (cur - lo) / (hi - lo)
        if pos > 0.85:
            score += 28
            signals.append(("52-Week Range", f"Near 52-week high ({pos*100:.0f}% of range) — strong long-term uptrend, watch for resistance", "bullish"))
        elif pos > 0.5:
            score += 40
            signals.append(("52-Week Range", f"Upper half of 52-week range ({pos*100:.0f}%) — healthy uptrend with room to run", "bullish"))
        elif pos > 0.2:
            score += 22
            signals.append(("52-Week Range", f"Lower half of 52-week range ({pos*100:.0f}%) — basing or still in a downtrend", "neutral"))
        else:
            score += 10
            signals.append(("52-Week Range", f"Near 52-week low ({pos*100:.0f}% of range) — long-term downtrend", "bearish"))
    else:
        score += 18
        signals.append(("52-Week Range", "Insufficient history for 52-week range", "neutral"))

    # Moving-average trend stack (0-35)
    if s20 and s50 and s200:
        if s20 > s50 > s200:
            score += 35
            signals.append(("Trend Direction", "SMA20 > SMA50 > SMA200 — established uptrend across all timeframes", "bullish"))
        elif s20 < s50 < s200:
            score += 8
            signals.append(("Trend Direction", "SMA20 < SMA50 < SMA200 — established downtrend across all timeframes", "bearish"))
        else:
            score += 20
            signals.append(("Trend Direction", "Mixed moving-average stack — short and long-term trends disagree", "neutral"))
    else:
        score += 18
        signals.append(("Trend Direction", "Not enough history for full moving-average trend", "neutral"))

    # Multi-timeframe momentum consistency (0-25)
    p1, p5, p20 = ind.get("price_change_1d", 0), ind.get("price_change_5d", 0), ind.get("price_change_20d", 0)
    pos_count = sum(1 for p in (p1, p5, p20) if p > 0)
    if pos_count == 3:
        score += 25
        signals.append(("Momentum Consistency", "Positive across 1d, 5d and 20d — trend is accelerating", "bullish"))
    elif pos_count == 0:
        score += 5
        signals.append(("Momentum Consistency", "Negative across 1d, 5d and 20d — trend is deteriorating", "bearish"))
    else:
        score += 13
        signals.append(("Momentum Consistency", "Mixed momentum across timeframes", "neutral"))

    return min(score, 100), signals


# ─────────────────────────────────────────────────────────────────────────────
# Confidence & Strategy
# ─────────────────────────────────────────────────────────────────────────────

def get_confidence(score: float) -> dict:
    if score >= 75:
        return {"level": "100%", "label": "STRONG BUY",    "color": "#00ff88", "bg": "#003322",
                "desc": "High conviction — strong technical + fundamental alignment"}
    if score >= 55:
        return {"level": "50%",  "label": "MODERATE BUY",  "color": "#ffd700", "bg": "#332800",
                "desc": "Multiple signals aligned — solid risk/reward setup"}
    if score >= 35:
        return {"level": "20%",  "label": "CAUTIOUS BUY",  "color": "#ff9800", "bg": "#331a00",
                "desc": "Some positive signals — limited/scaled entry recommended"}
    return     {"level": "10%",  "label": "SPECULATIVE",   "color": "#ff4444", "bg": "#330000",
                "desc": "Mixed/weak signals — very small speculative position only"}


def build_recommendation(ticker, company_name, tech, fund, sentiment, trend,
                          tech_signals, fund_signals, sentiment_signals, trend_signals) -> dict:
    """
    Synthesize technical, fundamental, sentiment and trend scores into a single
    written recommendation with its own confidence rating.
    """
    overall = round(tech * 0.30 + fund * 0.25 + trend * 0.25 + sentiment * 0.20, 1)
    conf = get_confidence(overall)

    all_signals = tech_signals + fund_signals + trend_signals + sentiment_signals
    bullish = [s for s in all_signals if s[2] == "bullish"]
    bearish = [s for s in all_signals if s[2] == "bearish"]

    parts = [
        f"{company_name} ({ticker}) scores {overall}/100 overall — "
        f"Technical {tech}/100, Fundamental {fund}/100, Trend {trend}/100, Sentiment {sentiment}/100."
    ]
    if bullish:
        parts.append("Supporting the case: " + "; ".join(s[1] for s in bullish[:3]) + ".")
    if bearish:
        parts.append("Risks to weigh: " + "; ".join(s[1] for s in bearish[:2]) + ".")
    parts.append(
        f"{len(bullish)} bullish vs {len(bearish)} bearish signals out of {len(all_signals)} — "
        f"overall confidence {conf['level']} ({conf['label']})."
    )

    return {
        "overall_score":  overall,
        "confidence":     conf,
        "narrative":      " ".join(parts),
        "bullish_count":  len(bullish),
        "bearish_count":  len(bearish),
        "total_signals":  len(all_signals),
    }


def calculate_targets(hist: pd.DataFrame, ind: dict, conf: dict) -> dict:
    cur    = ind["current_price"]
    recent = hist.tail(20)
    major  = hist.tail(60)

    sup1  = round(float(recent["Low"].min()), 2)
    res1  = round(float(recent["High"].max()), 2)
    sup2  = round(float(major["Low"].min()), 2)
    res2  = round(float(major["High"].max()), 2)
    bb_up = ind.get("bb_upper") or cur * 1.05
    bb_dn = ind.get("bb_lower") or cur * 0.95

    mult = {"100%": (1.18, 1.28, 0.92), "50%": (1.12, 1.20, 0.91),
            "20%":  (1.08, 1.14, 0.90), "10%": (1.05, 1.10, 0.88)}
    bm, om, slm = mult.get(conf["level"], (1.12, 1.20, 0.91))

    tgt      = round(cur * bm, 2)
    tgt_opt  = round(cur * om, 2)
    stop     = round(max(cur * slm, sup1 * 0.97), 2)
    up_pct   = round((tgt - cur) / cur * 100, 1)
    risk_pct = round((cur - stop) / cur * 100, 1)
    rr       = round(up_pct / risk_pct, 1) if risk_pct > 0 else 0

    return {"target_20d": tgt, "target_optimistic": tgt_opt, "stop_loss": stop,
            "support_1": sup1, "support_2": sup2, "resistance_1": res1, "resistance_2": res2,
            "bb_upper": round(float(bb_up), 2), "bb_lower": round(float(bb_dn), 2),
            "upside_pct": up_pct, "risk_pct": risk_pct, "risk_reward": rr}


def build_strategy(ticker, cur_price, conf, tgts, budget):
    level = conf["level"]
    alloc  = {"100%": 0.22, "50%": 0.13, "20%": 0.05, "10%": 0.02}
    entry  = {"100%": "Buy full position at market — strong conviction",
              "50%":  "Buy 60% now, add 40% on any 3-5% dip to support",
              "20%":  "Scale in: 50% now, 50% only if price holds support 2 days",
              "10%":  "Speculative — wait for volume confirmation before entering"}
    exit_  = {"100%": f"Sell 50% at ${tgts['target_20d']}, trail stop on rest to ${tgts['target_optimistic']}",
              "50%":  f"Sell 60% at ${tgts['target_20d']}, hold 40% toward ${tgts['target_optimistic']}",
              "20%":  f"Exit full position at ${tgts['target_20d']} — book the profit",
              "10%":  f"Exit quickly at ${tgts['target_20d']} — don't overstay"}

    pct    = alloc.get(level, 0.05)
    amount = round(budget * pct, 2)
    qty    = max(1, int(amount / cur_price))
    actual = round(qty * cur_price, 2)

    return {"alloc_pct": int(pct * 100), "recommended_amount": amount, "quantity": qty,
            "actual_investment": actual,
            "entry_strategy": entry.get(level, ""),
            "exit_strategy":  exit_.get(level, ""),
            "stop_loss": tgts["stop_loss"], "target": tgts["target_20d"],
            "target_opt": tgts["target_optimistic"],
            "max_loss":     round(qty * (cur_price - tgts["stop_loss"]), 2),
            "expected_gain": round(qty * (tgts["target_20d"] - cur_price), 2),
            "time_horizon": "20-30 trading days"}


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_core(ticker: str) -> dict:
    """
    Fetch market data and compute technical/fundamental analysis for a ticker.
    Everything returned here is budget-independent, so it's safe to cache.
    """
    # ── 1. Price History (NASDAQ) ──────────────────────────────────────────
    try:
        hist = fetch_price_history(ticker, days=400)
    except Exception as e:
        return {"error": str(e), "ticker": ticker}

    if len(hist) < 20:
        return {"error": f"Only {len(hist)} trading days of data for {ticker} — need at least 20.", "ticker": ticker}

    # ── 2. Current Quote (NASDAQ) ─────────────────────────────────────────
    quote = {}
    try:
        quote = fetch_current_quote(ticker)
    except Exception:
        pass

    # Use today's close from history if quote is unavailable
    current_price = quote.get("last_price") or float(hist["Close"].iloc[-1])
    company_name  = quote.get("company_name") or ticker

    # ── 3. Fundamentals (SEC EDGAR) ───────────────────────────────────────
    sec = {}
    try:
        sec = fetch_sec_fundamentals(ticker)
        if sec.get("company_name_sec"):
            company_name = sec["company_name_sec"]
    except Exception:
        pass

    # ── 4. Short Interest (FINRA) ─────────────────────────────────────────
    short = {}
    try:
        short = fetch_short_interest(ticker)
    except Exception:
        pass

    # ── 5. Technical Analysis ─────────────────────────────────────────────
    ind = calculate_technical_indicators(hist)
    ind["current_price"] = current_price      # use live price if available
    tech_score, tech_signals = calculate_technical_score(ind)

    # ── 6. Fundamental Scoring ────────────────────────────────────────────
    fund_score, fund_signals = calculate_fundamental_score(sec, short, quote, ind)

    # ── 7. Combined Score + Confidence ────────────────────────────────────
    combined = round(tech_score * 0.60 + fund_score * 0.40, 1)
    conf     = get_confidence(combined)

    # ── 7b. Sentiment & Trend Scoring ─────────────────────────────────────
    sentiment_score, sentiment_signals = calculate_sentiment_score(ind, short)
    trend_score, trend_signals = calculate_trend_score(ind)

    # ── 7c. AI Recommendation ─────────────────────────────────────────────
    recommendation = build_recommendation(
        ticker, company_name, round(tech_score, 1), round(fund_score, 1),
        round(sentiment_score, 1), round(trend_score, 1),
        tech_signals, fund_signals, sentiment_signals, trend_signals,
    )

    # ── 8. Price Targets ──────────────────────────────────────────────────
    tgts = calculate_targets(hist, ind, conf)

    # ── 9. Chart Data ─────────────────────────────────────────────────────
    tail = hist.tail(90)
    idx  = tail.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    chart_dates  = idx.strftime("%Y-%m-%d").tolist()
    chart_prices = [round(float(p), 2) for p in tail["Close"].tolist()]
    chart_volumes= [int(v) if not np.isnan(v) else 0 for v in tail["Volume"].tolist()]

    close_full = hist["Close"]
    chart_sma20 = [round(float(v), 2) if not np.isnan(v) else None
                   for v in close_full.rolling(20).mean().tail(90).tolist()]
    chart_sma50 = [round(float(v), 2) if not np.isnan(v) else None
                   for v in close_full.rolling(50).mean().tail(90).tolist()]

    market_cap_val = (sec.get("shares_outstanding") or 0) * current_price or None

    return {
        "ticker":        ticker,
        "company_name":  company_name,
        "exchange":      quote.get("exchange", ""),
        "current_price": round(current_price, 2),
        "market_cap":    _fmt_large(market_cap_val),
        "technical":     ind,
        "fundamental":   {
            "revenue_fmt":      sec.get("revenue_fmt", "N/A"),
            "revenue_growth":   sec.get("revenue_growth"),
            "net_income_fmt":   sec.get("net_income_fmt", "N/A"),
            "assets_fmt":       sec.get("assets_fmt", "N/A"),
            "cash_fmt":         sec.get("cash_fmt", "N/A"),
            "debt_fmt":         sec.get("debt_fmt", "N/A"),
            "profit_margin":    sec.get("profit_margin"),
            "gross_margin":     sec.get("gross_margin"),
            "debt_to_equity":   sec.get("debt_to_equity"),
            "short_pct_float":  short.get("short_pct"),
            "market_cap_fmt":   _fmt_large(market_cap_val),
            "sec_available":    sec.get("sec_available", False),
            "cik":              sec.get("cik"),
            "short_pct_display": short.get("short_pct_display", "N/A"),
            "short_vol_date":   short.get("last_date", ""),
        },
        "tech_signals":     tech_signals,
        "fund_signals":     fund_signals,
        "sentiment_signals": sentiment_signals,
        "trend_signals":    trend_signals,
        "tech_score":       round(tech_score, 1),
        "fund_score":       round(fund_score, 1),
        "sentiment_score":  round(sentiment_score, 1),
        "trend_score":      round(trend_score, 1),
        "combined_score":   combined,
        "confidence":       conf,
        "recommendation":   recommendation,
        "targets":       tgts,
        "chart_dates":   chart_dates,
        "chart_prices":  chart_prices,
        "chart_volumes": chart_volumes,
        "chart_sma20":   chart_sma20,
        "chart_sma50":   chart_sma50,
        "analyzed_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_sources":  "NASDAQ (prices) + SEC EDGAR (fundamentals) + FINRA (short interest)",
    }


def _analyze_core_cached(ticker: str) -> dict:
    now = time.time()
    cached = _CORE_CACHE.get(ticker)
    if cached and now - cached[0] < _CORE_CACHE_TTL:
        return cached[1]
    result = _analyze_core(ticker)
    if "error" not in result:
        _CORE_CACHE[ticker] = (now, result)
    return result


def analyze_stock(ticker: str, budget: float = 10000) -> dict:
    ticker = ticker.upper().strip()
    if not ticker:
        return {"error": "Empty ticker", "ticker": ticker}
    if not _TICKER_RE.match(ticker):
        return {"error": f"'{ticker}' doesn't look like a valid ticker symbol", "ticker": ticker}

    core = _analyze_core_cached(ticker)
    if "error" in core:
        return core

    strategy = build_strategy(ticker, core["current_price"], core["confidence"], core["targets"], budget)
    return {**core, "strategy": strategy}


# ─────────────────────────────────────────────────────────────────────────────
# Similar Ticker Suggestions — curated sector/theme peer groups
# ─────────────────────────────────────────────────────────────────────────────

SECTOR_PEER_GROUPS = {
    "space_defense":     ["RDW", "RKLB", "ASTS", "LUNR", "PL", "KTOS", "AVAV"],
    "drone_robotics":    ["ONDS", "AVAV", "KTOS", "RCAT", "AIRO", "IRBT"],
    "biotech_clinical":  ["RLAY", "ARVN", "KROS", "IOVA", "CRSP", "BEAM", "NTLA"],
    "ai_semis":          ["NVDA", "AMD", "SMCI", "ARM", "SOUN", "BBAI", "AI", "PLTR"],
    "quantum_computing": ["IONQ", "RGTI", "QBTS", "QUBT"],
    "nuclear_smr":       ["OKLO", "SMR", "LEU", "CCJ", "NNE", "BWXT"],
    "ev_battery":        ["RIVN", "LCID", "QS", "CHPT", "FREY"],
    "crypto_mining":     ["MARA", "RIOT", "CLSK", "WULF", "BITF", "CORZ"],
    "fintech_payments":  ["SOFI", "AFRM", "UPST", "PYPL", "SQ"],
    "mega_cap_tech":     ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA"],
    "semiconductors":    ["NVDA", "AMD", "INTC", "AVGO", "QCOM", "MU", "TXN", "ARM"],
    "software_saas":     ["MSFT", "CRM", "ADBE", "NOW", "ORCL", "SNOW", "PLTR", "PANW"],
    "banks_financials":  ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP"],
    "healthcare_pharma": ["JNJ", "PFE", "MRK", "ABBV", "LLY", "UNH", "BMY"],
    "energy_oil_gas":    ["XOM", "CVX", "COP", "OXY", "SLB", "EOG"],
    "retail_consumer":   ["WMT", "TGT", "COST", "HD", "LOW", "NKE", "SBUX"],
    "airlines_travel":   ["DAL", "UAL", "AAL", "LUV", "ABNB", "BKNG"],
    "streaming_media":   ["NFLX", "DIS", "WBD", "ROKU", "PARA"],
    "auto_ev":           ["TSLA", "F", "GM", "TM", "RIVN", "LCID"],
    "social_internet":   ["META", "GOOGL", "SNAP", "PINS", "RDDT"],
    "cloud_ecommerce":   ["AMZN", "SHOP", "MELI", "EBAY", "ETSY"],
}

# Broad pool of liquid large-caps used when an input ticker isn't covered by
# any curated sector group above (e.g. V, KO, PG, GOOG, ETFs).
_GENERAL_MARKET_POOL = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "JPM", "V", "JNJ", "WMT", "XOM", "HD", "DIS", "NFLX",
    "AMD", "CRM", "COST", "BAC", "PG",
]


def _build_reverse_map(groups: dict) -> dict:
    reverse: dict = {}
    for group, members in groups.items():
        for t in members:
            reverse.setdefault(t, []).append(group)
    return reverse


_TICKER_TO_GROUPS = _build_reverse_map(SECTOR_PEER_GROUPS)


def find_similar_tickers(input_tickers: list, budget: float = 10000, max_suggestions: int = 2, max_candidates: int = 10) -> list:
    """
    Given a list of already-analyzed tickers, suggest other tickers from the
    same sector/theme group that currently look attractive for a short-term
    trade. Ranked by a "short-term score" weighted toward technicals and
    trend/momentum (since short-term trades live and die on timing), with
    sentiment and fundamentals as secondary factors.
    """
    input_set = {t.upper().strip() for t in input_tickers}

    candidates: dict = {}  # candidate ticker -> set of related input tickers
    for t in input_set:
        for group in _TICKER_TO_GROUPS.get(t, []):
            for peer in SECTOR_PEER_GROUPS[group]:
                if peer in input_set:
                    continue
                candidates.setdefault(peer, set()).add(t)

    # Fallback: if the ticker(s) entered aren't covered by any curated sector
    # group (e.g. V, KO, PG, GOOG, ETFs), suggest from a broad pool of liquid
    # large-caps so the "Similar Opportunities" section is never empty.
    if not candidates:
        for peer in _GENERAL_MARKET_POOL:
            if peer not in input_set:
                candidates[peer] = set(input_set)

    cand_items = list(candidates.items())[:max_candidates]

    with ThreadPoolExecutor(max_workers=min(8, len(cand_items))) as ex:
        analyzed = list(ex.map(lambda kv: (kv[0], kv[1], analyze_stock(kv[0], budget)), cand_items))

    scored = []
    for cand, related, result in analyzed:
        if "error" in result:
            continue
        tech, fund = result["tech_score"], result["fund_score"]
        trend, sentiment = result["trend_score"], result["sentiment_score"]
        short_term_score = round(tech * 0.4 + trend * 0.3 + sentiment * 0.2 + fund * 0.1, 1)
        all_signals = (result["tech_signals"] + result["fund_signals"]
                       + result["trend_signals"] + result["sentiment_signals"])
        bullish = [s for s in all_signals if s[2] == "bullish"]
        reason = bullish[0][1] if bullish else (all_signals[0][1] if all_signals else "")
        scored.append({
            "ticker":           cand,
            "company_name":     result["company_name"],
            "current_price":    result["current_price"],
            "tech_score":       tech,
            "fund_score":       fund,
            "trend_score":      trend,
            "sentiment_score":  sentiment,
            "combined_score":   result["combined_score"],
            "short_term_score": short_term_score,
            "confidence":       result["confidence"],
            "price_change_5d":  result["technical"]["price_change_5d"],
            "reason":           reason,
            "related_to":       sorted(related),
            "strategy":         result["strategy"],
        })

    scored.sort(key=lambda x: x["short_term_score"], reverse=True)
    return scored[:max_suggestions]


def test_connectivity() -> dict:
    out = {}
    # NASDAQ
    try:
        h = fetch_price_history("AAPL", days=10)
        out["nasdaq_prices"] = f"OK — {len(h)} rows"
    except Exception as e:
        out["nasdaq_prices"] = f"FAIL: {e}"
    # SEC
    try:
        cik_map = _load_cik_map()
        out["sec_edgar"] = f"OK — {len(cik_map)} tickers in CIK map"
    except Exception as e:
        out["sec_edgar"] = f"FAIL: {e}"
    # FINRA
    try:
        s = fetch_short_interest("AAPL")
        out["finra"] = f"OK — short pct: {s.get('short_pct_display','N/A')}"
    except Exception as e:
        out["finra"] = f"FAIL: {e}"
    return out
