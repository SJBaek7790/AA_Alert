#!/usr/bin/env python3
"""
Monthly Portfolio Allocation Calculator
- HAA 한국형 (50%) + Defense First 한국형 (50%)
- Sends allocation via Telegram with diffs from last month
- Stores memory in allocation_history.json
- Runs on the day before the last business day of each month via GitHub Actions
"""

import calendar
import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from workalendar.asia import SouthKorea

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

KST = ZoneInfo("Asia/Seoul")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
HISTORY_FILE = "allocation_history.json"

# Maximum Telegram message length
TELEGRAM_MAX_LENGTH = 4096

# Friendly names for all tickers
TICKER_NAMES = {
    "418660.KS": "TIGER 미국나스닥100레버리지(합성)",
    "308620.KS": "KODEX 미국10년국채선물",
    "360200.KS": "ACE 미국S&P500",
    "489250.KS": "KODEX 미국배당다우존스",
    "280930.KS": "KODEX 미국러셀2000(H)",
    "195980.KS": "ARIRANG 신흥국MSCI(합성H)",
    "251350.KS": "KODEX 선진국MSCI World",
    "476760.KS": "ACE 미국30년국채액티브",
    "276000.KS": "TIGER 글로벌자원생산기업(합성H)",
    "182480.KS": "TIGER 미국MSCI리츠(합성H)",
    "411060.KS": "ACE KRX금현물",
    "261220.KS": "KODEX WTI원유선물(H)",
    "137610.KS": "TIGER 농산물선물Enhanced(H)",
    "0048J0.KS": "KODEX 미국머니마켓액티브",
    "0060H0.KS": "TIGER 토탈월드스탁액티브",
    "PDBC": "Invesco Optimum Yield Diversified Commodity",
    "TIP": "TIP",
    "BIL": "BIL",
    "UUP": "Invesco DB US Dollar Index",
}


# ──────────────────────────────────────────────
# Retry wrapper
# ──────────────────────────────────────────────

def retry(fn, *, max_retries: int = 3, delay: float = 2.0, backoff: float = 2.0,
          exceptions: tuple = (Exception,)):
    """
    Simple retry wrapper with exponential backoff.
    Returns the result of fn() on success.
    Raises the last exception if all retries are exhausted.
    """
    last_exc = None
    current_delay = delay
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except exceptions as e:
            last_exc = e
            if attempt < max_retries:
                print(f"  ⟳ Attempt {attempt}/{max_retries} failed: {e}. "
                      f"Retrying in {current_delay:.1f}s...")
                time.sleep(current_delay)
                current_delay *= backoff
            else:
                print(f"  ✗ All {max_retries} attempts failed: {e}")
    raise last_exc


# ──────────────────────────────────────────────
# Date helpers
# ──────────────────────────────────────────────

def get_last_business_day(year: int, month: int) -> datetime:
    """Return the last business day (Mon-Fri, non-holiday) of the given month."""
    cal = SouthKorea()
    last_day = calendar.monthrange(year, month)[1]
    dt = datetime(year, month, last_day)
    while not cal.is_working_day(dt):
        dt -= timedelta(days=1)
    return dt


def is_day_before_last_business_day(dt: datetime) -> bool:
    """Check if `dt` is 1 business day before the last business day of its month."""
    cal = SouthKorea()
    last_bd = get_last_business_day(dt.year, dt.month)
    # Walk backwards from last_bd to find the previous business day
    prev_bd = last_bd - timedelta(days=1)
    while not cal.is_working_day(prev_bd):
        prev_bd -= timedelta(days=1)
    return dt.date() == prev_bd.date()


# ──────────────────────────────────────────────
# Data fetching
# ──────────────────────────────────────────────

def fetch_prices(tickers: list[str], period: str = "15mo") -> pd.DataFrame:
    """
    Download adjusted close prices for all tickers.
    Uses yfinance with retry. Falls back to pykrx for .KS tickers that fail.
    """
    # We need at least 13 months of data for 12M return calculation
    end_date = datetime.today()
    start_date = end_date - timedelta(days=400)  # ~13 months buffer

    all_prices = {}

    # Separate KS tickers and US tickers
    ks_tickers = [t for t in tickers if t.endswith(".KS")]
    us_tickers = [t for t in tickers if not t.endswith(".KS")]

    # Download US tickers via yfinance (batch) with retry
    if us_tickers:
        print(f"Downloading US tickers: {us_tickers}")

        def _download_us():
            return yf.download(us_tickers, start=start_date, end=end_date,
                               auto_adjust=True, progress=False)

        data = retry(_download_us, max_retries=3, delay=3.0)

        if len(us_tickers) == 1:
            all_prices[us_tickers[0]] = data["Close"]
        else:
            close_df = data["Close"]
            for t in us_tickers:
                if t in close_df.columns:
                    series = close_df[t].dropna()
                    if len(series) > 0:
                        all_prices[t] = series

    # Download KS tickers one by one (some newer tickers need individual handling)
    for t in ks_tickers:
        print(f"Downloading {t}...")

        # Try yfinance with retry
        try:
            def _download_ks(ticker=t):
                return yf.download(ticker, start=start_date, end=end_date,
                                   auto_adjust=True, progress=False)

            df = retry(_download_ks, max_retries=3, delay=2.0)
            if df is not None and len(df) > 20:
                all_prices[t] = df["Close"].squeeze()
                print(f"  ✓ yfinance OK ({len(df)} rows)")
                continue
        except Exception as e:
            print(f"  ✗ yfinance failed: {e}")

        # Fallback: try pykrx for KRX-listed ETFs with retry
        try:
            from pykrx import stock as krx_stock
            code = t.replace(".KS", "")
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")

            def _download_pykrx(c=code, s=start_str, e=end_str):
                return krx_stock.get_market_ohlcv(s, e, c)

            df_krx = retry(_download_pykrx, max_retries=3, delay=2.0)
            if df_krx is not None and len(df_krx) > 20:
                all_prices[t] = df_krx["종가"]
                all_prices[t].index = pd.to_datetime(all_prices[t].index)
                print(f"  ✓ pykrx OK ({len(df_krx)} rows)")
                continue
        except Exception as e2:
            print(f"  ✗ pykrx also failed: {e2}")

        print(f"  ⚠ Could not fetch data for {t}")

    prices_df = pd.DataFrame(all_prices)
    # Cap ffill to prevent silently dragging stale data for weeks
    prices_df = prices_df.sort_index().ffill(limit=5)

    if not prices_df.empty:
        # Freshness check: verify the latest index is recent
        last_date = prices_df.index[-1]
        
        # Determine days difference (handle timezone safely)
        now_naive = pd.Timestamp.today().tz_localize(None)
        last_date_naive = last_date.tz_localize(None)
        
        days_diff = (now_naive - last_date_naive).days
        if days_diff > 5:
            raise RuntimeError(
                f"Data freshness check failed! Latest prices are from {last_date.date()} "
                f"({days_diff} days old). Check for delistings, suspensions, or data provider failures."
            )

    return prices_df


# ──────────────────────────────────────────────
# Momentum calculation
# ──────────────────────────────────────────────

def calc_return(prices: pd.Series, months: int) -> float:
    """Calculate return over N months from the last available price."""
    if prices.empty:
        return np.nan
    end_price = prices.iloc[-1]
    if np.isnan(end_price) or end_price <= 0:
        return np.nan
    target_date = prices.index[-1] - pd.DateOffset(months=months)
    # Find closest date on or before target
    mask = prices.index <= target_date
    if mask.sum() == 0:
        return np.nan
    start_price = prices.loc[mask].iloc[-1]
    if np.isnan(start_price) or start_price <= 0:
        return np.nan
    return (end_price / start_price) - 1.0


def calc_momentum(prices: pd.Series) -> float:
    """Momentum = average of 1M, 3M, 6M, 12M returns."""
    r1 = calc_return(prices, 1)
    r3 = calc_return(prices, 3)
    r6 = calc_return(prices, 6)
    r12 = calc_return(prices, 12)
    vals = [v for v in [r1, r3, r6, r12] if not np.isnan(v)]
    if not vals:
        return np.nan
    return np.mean(vals)


def calc_12m_return(prices: pd.Series) -> float:
    """12-month total return."""
    return calc_return(prices, 12)


# ──────────────────────────────────────────────
# Allocation logic
# ──────────────────────────────────────────────

def calc_haa_allocation(prices_df: pd.DataFrame) -> dict[str, float]:
    """
    HAA 한국형 (50% of total portfolio)
    Returns weights summing to 0.50
    """
    weights: dict[str, float] = {}

    # Momentum of TIPS vs BIL
    mom_tips = calc_momentum(prices_df["TIP"])
    mom_bil = calc_momentum(prices_df["BIL"])

    print(f"\n[HAA] TIPS momentum: {mom_tips:.4f}, BIL momentum: {mom_bil:.4f}")

    if mom_tips > mom_bil:
        # ── Risk On ──
        print("[HAA] → Risk ON")

        # Fixed allocations (within the 50% sleeve)
        weights["418660.KS"] = 0.15  # TIGER 나스닥100 레버리지
        weights["308620.KS"] = 0.15  # KODEX 미국10년국채선물

        # Offensive pool: pick top 4 by momentum, each 5%
        offensive_pool = [
            "360200.KS", "489250.KS", "280930.KS", "195980.KS",
            "251350.KS", "476760.KS", "308620.KS", "PDBC", "182480.KS",
        ]
        mom_scores = {}
        for t in offensive_pool:
            if t in prices_df.columns:
                mom_scores[t] = calc_momentum(prices_df[t])
                print(f"  Offensive {t} ({TICKER_NAMES.get(t,t)}): mom={mom_scores[t]:.4f}")

        # Sort descending, pick top 4
        ranked = sorted(mom_scores.items(), key=lambda x: x[1], reverse=True)
        top4 = [t for t, _ in ranked[:4]]
        print(f"  Top 4: {top4}")

        for t in top4:
            weights[t] = weights.get(t, 0) + 0.05

    else:
        # ── Risk Off ──
        print("[HAA] → Risk OFF")

        # Higher momentum between 308620.KS and 0048J0.KS gets 50%
        bond_tickers = ["308620.KS", "0048J0.KS"]
        mom_bond = {}
        for t in bond_tickers:
            if t in prices_df.columns:
                mom_bond[t] = calc_momentum(prices_df[t])
                print(f"  {t} ({TICKER_NAMES.get(t,t)}): mom={mom_bond[t]:.4f}")

        if mom_bond:
            best = max(mom_bond, key=mom_bond.get)
            weights[best] = 0.50
            print(f"  → Allocate 50% to {best}")

    return weights


def calc_defense_first_allocation(prices_df: pd.DataFrame) -> dict[str, float]:
    """
    Defense First 한국형 (50% of total portfolio)
    Returns weights summing to 0.50
    """
    weights: dict[str, float] = {}

    mom_bil = calc_momentum(prices_df["BIL"])

    # Four defensive assets
    defensive_assets = ["476760.KS", "411060.KS", "PDBC", "UUP"]
    # Absolute portfolio weights (README: 40%/30%/20%/10% of the 50% Defense sleeve)
    tier_weights = [0.20, 0.15, 0.10, 0.05]

    # Calculate momentum for each defensive asset
    mom_def = {}
    for t in defensive_assets:
        if t in prices_df.columns:
            mom_def[t] = calc_momentum(prices_df[t])
            print(f"  Defensive {t} ({TICKER_NAMES.get(t,t)}): mom={mom_def[t]:.4f}")

    # Rank from highest to lowest momentum
    ranked = sorted(mom_def.items(), key=lambda x: x[1], reverse=True)
    print(f"\n[Defense First] Ranked: {[(t, f'{m:.4f}') for t,m in ranked]}")
    print(f"[Defense First] BIL momentum: {mom_bil:.4f}")

    # Replacement asset: higher momentum between 0060H0.KS and 0048J0.KS
    replacement_candidates = ["0060H0.KS", "0048J0.KS"]
    mom_repl = {}
    for t in replacement_candidates:
        if t in prices_df.columns:
            mom_repl[t] = calc_momentum(prices_df[t])
    if mom_repl:
        replacement_ticker = max(mom_repl, key=mom_repl.get)
    else:
        replacement_ticker = "0048J0.KS"  # fallback
    print(f"  Replacement asset: {replacement_ticker} ({TICKER_NAMES.get(replacement_ticker,'')})")

    for i, (ticker, mom) in enumerate(ranked):
        if i >= 4:
            break
        w = tier_weights[i]  # weight for this rank

        if mom < mom_bil:
            # Momentum < BIL → replace with higher-momentum safe asset
            print(f"  {ticker} mom ({mom:.4f}) < BIL ({mom_bil:.4f}) → replaced by {replacement_ticker}")
            weights[replacement_ticker] = weights.get(replacement_ticker, 0) + w
        else:
            weights[ticker] = weights.get(ticker, 0) + w

    return weights


def apply_further_rules(
    weights: dict[str, float],
    prices_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Further rules (Rule #3):
    1) UUP portion → 0048J0.KS (KODEX 미국머니마켓액티브)
    2) 411060.KS (gold) portion:
       - If 12M return of BOTH gold and 10Y treasury > 0 → keep gold
       - Otherwise → 0048J0.KS
    """
    final = dict(weights)

    # Rule 3a: Move UUP → 0048J0.KS
    if "UUP" in final and final["UUP"] > 0:
        uup_w = final.pop("UUP")
        final["0048J0.KS"] = final.get("0048J0.KS", 0) + uup_w
        print(f"\n[Rule 3a] UUP ({uup_w:.2%}) → 0048J0.KS")

    # Rule 3b: Gold check
    gold_ticker = "411060.KS"
    treasury_ticker = "308620.KS"

    if gold_ticker in final and final[gold_ticker] > 0:
        r12_gold = calc_12m_return(prices_df[gold_ticker]) if gold_ticker in prices_df.columns else np.nan
        r12_treas = calc_12m_return(prices_df[treasury_ticker]) if treasury_ticker in prices_df.columns else np.nan

        print(f"\n[Rule 3b] Gold 12M return: {r12_gold:.4f}, Treasury 12M return: {r12_treas:.4f}")

        if r12_gold > 0 and r12_treas > 0:
            print("  Both positive → keep gold allocation")
        else:
            gold_w = final.pop(gold_ticker)
            final["0048J0.KS"] = final.get("0048J0.KS", 0) + gold_w
            print(f"  Not both positive → Gold ({gold_w:.2%}) → 0048J0.KS")

    # Rule 3c: PDBC split
    if "PDBC" in final and final["PDBC"] > 0:
        pdbc_w = final.pop("PDBC")
        pdbc_split = pdbc_w / 3.0
        final["261220.KS"] = final.get("261220.KS", 0) + pdbc_split
        final["276000.KS"] = final.get("276000.KS", 0) + pdbc_split
        final["137610.KS"] = final.get("137610.KS", 0) + pdbc_split
        print(f"\n[Rule 3c] PDBC ({pdbc_w:.2%}) → 261220.KS, 276000.KS, 137610.KS equally ({pdbc_split:.2%} each)")

    # Remove zero-weight entries
    final = {k: v for k, v in final.items() if v > 1e-9}

    return final


# ──────────────────────────────────────────────
# Main allocation
# ──────────────────────────────────────────────

def calculate_allocation() -> dict[str, float]:
    """Full allocation pipeline."""

    # Collect all tickers we need
    all_tickers = list(set([
        # HAA fixed
        "418660.KS", "308620.KS",
        # HAA offensive pool
        "360200.KS", "489250.KS", "280930.KS", "195980.KS",
        "251350.KS", "476760.KS", "PDBC", "182480.KS",
        # HAA risk-off
        "0048J0.KS",
        # Defense First defensive
        "411060.KS", "PDBC", "UUP",
        # Defense First replacement
        "0060H0.KS",
        # PDBC split targets
        "261220.KS", "137610.KS", "276000.KS",
        # Momentum benchmarks
        "TIP", "BIL",
    ]))

    prices_df = fetch_prices(all_tickers)

    if prices_df.empty:
        raise RuntimeError("No price data fetched. All downloads failed.")

    print(f"\nPrice data shape: {prices_df.shape}")
    print(f"Date range: {prices_df.index[0].date()} → {prices_df.index[-1].date()}")
    print(f"Tickers fetched: {list(prices_df.columns)}")

    # Check for missing tickers
    missing = [t for t in all_tickers if t not in prices_df.columns]
    if missing:
        print(f"\n⚠ WARNING: Missing data for: {missing}")
        # If critical tickers are missing, abort
        critical = {"TIP", "BIL"}
        missing_critical = critical & set(missing)
        if missing_critical:
            raise RuntimeError(f"Missing critical benchmark tickers: {missing_critical}")

    # Step 1: HAA 한국형 (50%)
    print("\n" + "=" * 60)
    print("STEP 1: HAA 한국형 (50%)")
    print("=" * 60)
    haa_weights = calc_haa_allocation(prices_df)
    print(f"HAA weights: { {k: f'{v:.2%}' for k,v in haa_weights.items()} }")

    # Step 2: Defense First 한국형 (50%)
    print("\n" + "=" * 60)
    print("STEP 2: Defense First 한국형 (50%)")
    print("=" * 60)
    def_weights = calc_defense_first_allocation(prices_df)
    print(f"Defense weights: { {k: f'{v:.2%}' for k,v in def_weights.items()} }")

    # Combine
    combined = {}
    for w_dict in [haa_weights, def_weights]:
        for k, v in w_dict.items():
            combined[k] = combined.get(k, 0) + v

    print(f"\nCombined (before Rule 3): { {k: f'{v:.2%}' for k,v in combined.items()} }")

    # Step 3: Apply further rules
    print("\n" + "=" * 60)
    print("STEP 3: Further Rules")
    print("=" * 60)
    final = apply_further_rules(combined, prices_df)

    # Verify total
    total = sum(final.values())
    print(f"\nFinal total weight: {total:.2%}")
    if abs(total - 1.0) > 0.01:
        raise RuntimeError(f"Total weight is {total:.4f}, expected 1.0")

    return final


# ──────────────────────────────────────────────
# History / JSON memory
# ──────────────────────────────────────────────

def load_history() -> dict:
    path = Path(HISTORY_FILE)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"records": []}


def save_history(history: dict):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def get_last_month_weights(history: dict) -> dict[str, float] | None:
    records = history.get("records", [])
    if not records:
        return None
    return records[-1].get("weights", {})


def add_record(history: dict, weights: dict[str, float]) -> dict:
    now = datetime.now(KST)
    today_str = now.strftime("%Y-%m-%d")
    record = {
        "date": today_str,
        "weights": {k: round(v, 4) for k, v in weights.items()},
    }
    history.setdefault("records", []).append(record)
    return history


# ──────────────────────────────────────────────
# Telegram
# ──────────────────────────────────────────────

def _send_telegram_raw(text: str, parse_mode: str = "HTML"):
    """Low-level Telegram send with retry and chunking."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("\n⚠ Telegram credentials not set. Message:")
        print(text)
        return

    # Split into chunks if message exceeds Telegram limit
    chunks = []
    if len(text) <= TELEGRAM_MAX_LENGTH:
        chunks = [text]
    else:
        lines = text.split("\n")
        current_chunk = ""
        for line in lines:
            if len(current_chunk) + len(line) + 1 > TELEGRAM_MAX_LENGTH:
                chunks.append(current_chunk)
                current_chunk = line
            else:
                current_chunk += ("\n" if current_chunk else "") + line
        if current_chunk:
            chunks.append(current_chunk)

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    for i, chunk in enumerate(chunks, 1):
        def _send(c=chunk):
            resp = requests.post(url, json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": c,
                "parse_mode": parse_mode,
            }, timeout=30)
            resp.raise_for_status()
            return resp

        try:
            retry(_send, max_retries=3, delay=2.0,
                  exceptions=(requests.RequestException,))
            print(f"✓ Telegram message {i}/{len(chunks)} sent successfully.")
        except Exception as e:
            print(f"✗ Telegram error for chunk {i}/{len(chunks)}: {e}")


def send_telegram_message(text: str):
    """Send an HTML-formatted Telegram message."""
    _send_telegram_raw(text, parse_mode="HTML")


def send_error_telegram(error: Exception, tb_str: str):
    """
    Send error details to Telegram as a JSON code block.
    Truncates traceback if needed to stay within character limits.
    """
    now = datetime.now(KST)
    error_data = {
        "status": "FAILED",
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S KST"),
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": tb_str,
    }

    # Build the message
    header = "🚨 <b>Portfolio Allocation FAILED</b>\n\n"
    error_json = json.dumps(error_data, indent=2, ensure_ascii=False)
    code_block = f"<pre>{error_json}</pre>"
    full_msg = header + code_block

    # If too long, truncate traceback to fit
    if len(full_msg) > TELEGRAM_MAX_LENGTH:
        max_tb_len = TELEGRAM_MAX_LENGTH - len(header) - 200  # reserve space for JSON structure
        error_data["traceback"] = tb_str[:max_tb_len] + "\n... (truncated)"
        error_json = json.dumps(error_data, indent=2, ensure_ascii=False)
        code_block = f"<pre>{error_json}</pre>"
        full_msg = header + code_block

    _send_telegram_raw(full_msg, parse_mode="HTML")


def format_message(
    weights: dict[str, float],
    prev_weights: dict[str, float] | None,
) -> str:
    now = datetime.now(KST)
    today_str = now.strftime("%Y-%m-%d")
    lines = []
    lines.append(f"<b>📊 Monthly Allocation Report</b>")
    lines.append(f"<b>Date: {today_str}</b>")
    lines.append("")

    # Sort by weight descending
    sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)

    lines.append("<b>── Current Allocation ──</b>")
    for ticker, w in sorted_w:
        name = TICKER_NAMES.get(ticker, ticker)
        pct = f"{w:.1%}"

        if prev_weights:
            prev_w = prev_weights.get(ticker, 0.0)
            diff = w - prev_w
            if abs(diff) < 0.0001:
                diff_str = "  (→)"
            elif diff > 0:
                diff_str = f"  (▲ +{diff:.1%})"
            else:
                diff_str = f"  (▼ {diff:.1%})"
        else:
            diff_str = "  (NEW)"

        lines.append(f"• <b>{pct}</b>  {name} [{ticker}]{diff_str}")

    # Show removed positions
    if prev_weights:
        removed = set(prev_weights.keys()) - set(weights.keys())
        for ticker in removed:
            if prev_weights[ticker] > 0.001:
                name = TICKER_NAMES.get(ticker, ticker)
                lines.append(
                    f"• <b>0.0%</b>  {name} [{ticker}]"
                    f"  (▼ -{prev_weights[ticker]:.1%})"
                )

    lines.append("")
    total = sum(weights.values())
    lines.append(f"<b>Total: {total:.1%}</b>")

    if prev_weights:
        lines.append("")
        lines.append("<b>── Changes Summary ──</b>")
        all_tickers_union = set(list(weights.keys()) + list(prev_weights.keys()))
        changes = []
        for t in all_tickers_union:
            cur = weights.get(t, 0)
            prev = prev_weights.get(t, 0)
            diff = cur - prev
            if abs(diff) > 0.0001:
                name = TICKER_NAMES.get(t, t)
                changes.append((t, name, prev, cur, diff))
        changes.sort(key=lambda x: abs(x[4]), reverse=True)
        if changes:
            for t, name, prev, cur, diff in changes:
                arrow = "▲" if diff > 0 else "▼"
                lines.append(f"  {arrow} {name}: {prev:.1%} → {cur:.1%} ({diff:+.1%})")
        else:
            lines.append("  No changes from last month.")
    else:
        lines.append("\n<i>First run — no previous data to compare.</i>")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    now = datetime.now(KST)

    print("=" * 60)
    print("Monthly Portfolio Allocation Calculator")
    print(f"Run date (KST): {now.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # ── Idempotency: skip if already ran this month ──
    history = load_history()
    current_month = now.strftime("%Y-%m")
    records = history.get("records", [])
    if records and records[-1]["date"].startswith(current_month):
        print(f"Already ran for {current_month} (last record: {records[-1]['date']}). Skipping.")
        return

    # ── Date guard: only run on the day before the last business day ──
    if not is_day_before_last_business_day(now):
        print(f"Today ({now.strftime('%Y-%m-%d')}) is not the day before "
              f"the last business day of the month. Skipping.")
        return

    # Calculate allocation
    weights = calculate_allocation()

    # Get previous weights (history already loaded above)
    prev_weights = get_last_month_weights(history)

    # Format & send message
    msg = format_message(weights, prev_weights)
    print("\n" + "=" * 60)
    print("TELEGRAM MESSAGE:")
    print("=" * 60)
    print(msg)
    send_telegram_message(msg)

    # Save to history
    history = add_record(history, weights)
    save_history(history)
    print(f"\n✓ History saved to {HISTORY_FILE}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        print(f"\nFATAL ERROR: {e}", file=sys.stderr)
        print(tb_str, file=sys.stderr)
        # Send error notification via Telegram
        try:
            send_error_telegram(e, tb_str)
        except Exception as tg_err:
            print(f"Could not send error to Telegram: {tg_err}", file=sys.stderr)
        sys.exit(1)
