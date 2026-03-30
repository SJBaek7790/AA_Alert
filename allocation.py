#!/usr/bin/env python3
"""
Monthly Portfolio Allocation Calculator
- HAA 한국형 (50%) + Defense First 한국형 (50%)
- Sends allocation via Telegram with diffs from last month
- Stores memory in allocation_history.json
- Runs monthly via GitHub Actions
"""

import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
HISTORY_FILE = "allocation_history.json"

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
    "0048J0.KS": "KODEX 미국머니마켓액티브",
    "0060H0.KS": "TIGER 토탈월드스탁액티브",
    "TIP": "iShares TIPS Bond ETF",
    "BIL": "SPDR Blmbg 1-3M T-Bill ETF",
    "UUP": "Invesco DB US Dollar Index",
}

# ──────────────────────────────────────────────
# Data fetching
# ──────────────────────────────────────────────

def fetch_prices(tickers: list[str], period: str = "15mo") -> pd.DataFrame:
    """
    Download adjusted close prices for all tickers.
    Uses yfinance. Falls back to pykrx for .KS tickers that fail.
    """
    # We need at least 13 months of data for 12M return calculation
    end_date = datetime.today()
    start_date = end_date - timedelta(days=400)  # ~13 months buffer

    all_prices = {}

    # Separate KS tickers and US tickers
    ks_tickers = [t for t in tickers if t.endswith(".KS")]
    us_tickers = [t for t in tickers if not t.endswith(".KS")]

    # Download US tickers via yfinance (batch)
    if us_tickers:
        print(f"Downloading US tickers: {us_tickers}")
        data = yf.download(us_tickers, start=start_date, end=end_date,
                           auto_adjust=True, progress=False)
        if len(us_tickers) == 1:
            all_prices[us_tickers[0]] = data["Close"]
        else:
            for t in us_tickers:
                if t in data["Close"].columns:
                    series = data["Close"][t].dropna()
                    if len(series) > 0:
                        all_prices[t] = series

    # Download KS tickers one by one (some newer tickers need individual handling)
    for t in ks_tickers:
        print(f"Downloading {t}...")
        try:
            df = yf.download(t, start=start_date, end=end_date,
                             auto_adjust=True, progress=False)
            if df is not None and len(df) > 20:
                all_prices[t] = df["Close"].squeeze()
                print(f"  ✓ yfinance OK ({len(df)} rows)")
                continue
        except Exception as e:
            print(f"  ✗ yfinance failed: {e}")

        # Fallback: try pykrx for KRX-listed ETFs
        try:
            from pykrx import stock as krx_stock
            code = t.replace(".KS", "")
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            df_krx = krx_stock.get_market_ohlcv(start_str, end_str, code)
            if df_krx is not None and len(df_krx) > 20:
                all_prices[t] = df_krx["종가"]
                all_prices[t].index = pd.to_datetime(all_prices[t].index)
                print(f"  ✓ pykrx OK ({len(df_krx)} rows)")
                continue
        except Exception as e2:
            print(f"  ✗ pykrx also failed: {e2}")

        print(f"  ⚠ Could not fetch data for {t}")

    prices_df = pd.DataFrame(all_prices)
    prices_df = prices_df.sort_index().ffill()
    return prices_df


# ──────────────────────────────────────────────
# Momentum calculation
# ──────────────────────────────────────────────

def calc_return(prices: pd.Series, months: int) -> float:
    """Calculate return over N months from the last available price."""
    if prices.empty:
        return -999.0
    end_price = prices.iloc[-1]
    target_date = prices.index[-1] - pd.DateOffset(months=months)
    # Find closest date on or before target
    mask = prices.index <= target_date
    if mask.sum() == 0:
        return -999.0
    start_price = prices.loc[mask].iloc[-1]
    if start_price == 0:
        return -999.0
    return (end_price / start_price) - 1.0


def calc_momentum(prices: pd.Series) -> float:
    """Momentum = average of 1M, 3M, 6M, 12M returns."""
    r1 = calc_return(prices, 1)
    r3 = calc_return(prices, 3)
    r6 = calc_return(prices, 6)
    r12 = calc_return(prices, 12)
    vals = [v for v in [r1, r3, r6, r12] if v > -900]
    if not vals:
        return -999.0
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
            "251350.KS", "476760.KS", "308620.KS", "276000.KS", "182480.KS",
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
    defensive_assets = ["476760.KS", "411060.KS", "261220.KS", "UUP"]
    # Allocation tiers for ranks 1-4
    tier_weights = [0.20, 0.15, 0.10, 0.05]  # × 50% of portfolio = 40%,30%,20%,10% of this sleeve

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
        r12_gold = calc_12m_return(prices_df[gold_ticker]) if gold_ticker in prices_df.columns else -999
        r12_treas = calc_12m_return(prices_df[treasury_ticker]) if treasury_ticker in prices_df.columns else -999

        print(f"\n[Rule 3b] Gold 12M return: {r12_gold:.4f}, Treasury 12M return: {r12_treas:.4f}")

        if r12_gold > 0 and r12_treas > 0:
            print(f"  Both positive → keep gold allocation")
        else:
            gold_w = final.pop(gold_ticker)
            final["0048J0.KS"] = final.get("0048J0.KS", 0) + gold_w
            print(f"  Not both positive → Gold ({gold_w:.2%}) → 0048J0.KS")

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
        "251350.KS", "476760.KS", "276000.KS", "182480.KS",
        # HAA risk-off
        "0048J0.KS",
        # Defense First defensive
        "411060.KS", "261220.KS", "UUP",
        # Defense First replacement
        "0060H0.KS",
        # Momentum benchmarks
        "TIP", "BIL",
    ]))

    prices_df = fetch_prices(all_tickers)
    print(f"\nPrice data shape: {prices_df.shape}")
    print(f"Date range: {prices_df.index[0].date()} → {prices_df.index[-1].date()}")
    print(f"Tickers fetched: {list(prices_df.columns)}")

    # Check for missing tickers
    missing = [t for t in all_tickers if t not in prices_df.columns]
    if missing:
        print(f"\n⚠ WARNING: Missing data for: {missing}")

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
        print(f"⚠ Total weight is {total:.4f}, expected 1.0")

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
    today_str = datetime.today().strftime("%Y-%m-%d")
    record = {
        "date": today_str,
        "weights": {k: round(v, 4) for k, v in weights.items()},
    }
    history.setdefault("records", []).append(record)
    return history


# ──────────────────────────────────────────────
# Telegram
# ──────────────────────────────────────────────

def send_telegram_message(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("\n⚠ Telegram credentials not set. Message:")
        print(text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    }
    resp = requests.post(url, json=payload, timeout=30)
    if resp.status_code == 200:
        print("✓ Telegram message sent successfully.")
    else:
        print(f"✗ Telegram error {resp.status_code}: {resp.text}")


def format_message(
    weights: dict[str, float],
    prev_weights: dict[str, float] | None,
) -> str:
    today_str = datetime.today().strftime("%Y-%m-%d")
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
    print("=" * 60)
    print("Monthly Portfolio Allocation Calculator")
    print(f"Run date: {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Calculate allocation
    weights = calculate_allocation()

    # Load history & get previous weights
    history = load_history()
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
    main()
