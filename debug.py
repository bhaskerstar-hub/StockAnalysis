#!/usr/bin/env python3
"""
Run this from Terminal to diagnose connectivity:
  python3 /Users/bhaskerc/Documents/TickerAnalysis/debug.py
"""
import sys, time, json
import requests

TICKER = "RDW"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
CHROME_HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"macOS"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
}

print("=" * 65)
print("  Stock Analysis App — Network Diagnostic")
print("=" * 65)

# ── 1. Basic internet ─────────────────────────────────────────────
print("\n[1] Basic internet connectivity...")
try:
    r = requests.get("https://www.google.com", timeout=10, headers={"User-Agent": UA})
    print(f"    Google.com: OK (HTTP {r.status_code})")
except Exception as e:
    print(f"    FAIL — no internet? {e}")
    sys.exit(1)

# ── 2. fc.yahoo.com ───────────────────────────────────────────────
print("\n[2] fc.yahoo.com (required by yfinance >= 0.2.37)...")
try:
    r = requests.get("https://fc.yahoo.com", timeout=10, headers={"User-Agent": UA})
    print(f"    fc.yahoo.com: OK (HTTP {r.status_code})")
    FC_OK = True
except Exception as e:
    print(f"    FAIL: {e}")
    print("    → yfinance will fail. The fallback (direct requests) will be used.")
    FC_OK = False

# ── 3. Yahoo Finance homepage → cookies ───────────────────────────
print("\n[3] Yahoo Finance homepage (to receive session cookies)...")
sess = requests.Session()
sess.headers.update(CHROME_HEADERS)
COOKIES_OK = False
try:
    r = sess.get("https://finance.yahoo.com", timeout=15)
    cookies = list(sess.cookies.keys())
    print(f"    finance.yahoo.com: HTTP {r.status_code}")
    print(f"    Cookies received: {cookies}")
    COOKIES_OK = bool(cookies)
except Exception as e:
    print(f"    FAIL: {e}")

# ── 4. Crumb token ────────────────────────────────────────────────
print("\n[4] Yahoo Finance crumb token...")
sess.headers.update({"Accept": "application/json,*/*", "Sec-Fetch-Dest": "empty",
                     "Sec-Fetch-Mode": "cors", "Sec-Fetch-Site": "same-site",
                     "Referer": "https://finance.yahoo.com/"})
crumb = None
for base in ("query2", "query1"):
    try:
        r = sess.get(f"https://{base}.finance.yahoo.com/v1/test/getcrumb", timeout=12)
        print(f"    {base}: HTTP {r.status_code}  →  {r.text[:60]}")
        if r.status_code == 200 and "Too Many" not in r.text and r.text.strip():
            crumb = r.text.strip()
            break
    except Exception as e:
        print(f"    {base}: FAIL — {e}")

if crumb:
    print(f"    ✓ Crumb obtained: {crumb[:30]}...")
else:
    print("    ✗ No crumb obtained (will try without it)")

# ── 5. Chart API — the main data endpoint ────────────────────────
print(f"\n[5] Chart data API for {TICKER} (price history)...")
end = int(time.time()); start = end - 185 * 86400
params = {"period1": start, "period2": end, "interval": "1d", "includePrePost": "false"}
if crumb:
    params["crumb"] = crumb

DATA_OK = False
for base in ("query2", "query1"):
    try:
        r = sess.get(
            f"https://{base}.finance.yahoo.com/v8/finance/chart/{TICKER}",
            params=params, timeout=25
        )
        print(f"    {base}: HTTP {r.status_code}", end="")
        if r.status_code == 200:
            d = r.json()
            rows = len((d.get("chart", {}).get("result") or [{}])[0].get("timestamp", []))
            print(f"  →  {rows} rows of OHLCV data")
            if rows > 0:
                DATA_OK = True
                break
        else:
            print(f"  →  {r.text[:80]}")
    except Exception as e:
        print(f"\n    {base}: FAIL — {e}")

# ── 6. Fundamentals / quoteSummary ───────────────────────────────
print(f"\n[6] Fundamentals API for {TICKER}...")
qs_params = {"modules": "price,assetProfile", "formatted": "false"}
if crumb:
    qs_params["crumb"] = crumb
FUND_OK = False
for base in ("query2", "query1"):
    try:
        r = sess.get(
            f"https://{base}.finance.yahoo.com/v10/finance/quoteSummary/{TICKER}",
            params=qs_params, timeout=20
        )
        print(f"    {base}: HTTP {r.status_code}", end="")
        if r.status_code == 200:
            d = r.json()
            res = (d.get("quoteSummary", {}).get("result") or [{}])[0]
            name = (res.get("price") or {}).get("longName", "?")
            print(f"  →  company name: {name}")
            FUND_OK = True
            break
        else:
            print(f"  →  {r.text[:80]}")
    except Exception as e:
        print(f"\n    {base}: FAIL — {e}")

# ── 7. yfinance ──────────────────────────────────────────────────
print("\n[7] yfinance library (direct call)...")
try:
    import yfinance as yf
    import importlib.metadata as im
    ver = im.version("yfinance")
    t = yf.Ticker(TICKER)
    h = t.history(period="5d")
    status = f"OK — {len(h)} rows" if not h.empty else "empty result"
    print(f"    yfinance {ver}: {status}")
except Exception as e:
    print(f"    yfinance: FAIL — {str(e)[:120]}")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  SUMMARY")
print("=" * 65)
print(f"  fc.yahoo.com reachable : {'YES' if FC_OK else 'NO  ← yfinance will fail'}")
print(f"  Session cookies        : {'YES' if COOKIES_OK else 'NO'}")
print(f"  Crumb token            : {'YES — ' + (crumb[:20] if crumb else '') if crumb else 'NO'}")
print(f"  Price data (chart API) : {'YES ✓' if DATA_OK else 'NO  ← main problem'}")
print(f"  Fundamentals API       : {'YES ✓' if FUND_OK else 'NO (analysis will work without it)'}")

if DATA_OK:
    print("\n  ✓ Data fetching works! The app should function correctly.")
    print("  Run: bash /Users/bhaskerc/Documents/TickerAnalysis/run.sh")
else:
    print("\n  ✗ Cannot fetch price data from Yahoo Finance.")
    print("  Please share this output for further diagnosis.")
print("=" * 65)
