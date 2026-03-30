# Monthly Portfolio Allocation System

Automated monthly portfolio allocation calculator combining **HAA 한국형 (50%)** + **Defense First 한국형 (50%)**.

- Runs on the **day before the last business day** of each month via GitHub Actions
- Sends allocation report with diffs via Telegram
- Stores allocation history in `allocation_history.json`

---

## 1. HAA 한국형 (50%)

- Calculate Momentum = (1M + 3M + 6M + 12M) / 4
- Compare Momentum of TIP (TIPS) vs BIL (T-Bills)

### 1-1. Risk On (TIP > BIL)

- 418660.KS (TIGER 미국나스닥100레버리지(합성)) — **15%**
- 308620.KS (KODEX 미국10년국채선물) — **15%**
- Top 4 of Offensive Asset Pool, each **5%**:
  - 360200.KS (ACE 미국S&P500)
  - 489250.KS (KODEX 미국배당다우존스)
  - 280930.KS (KODEX 미국러셀2000(H))
  - 195980.KS (ARIRANG 신흥국MSCI(합성H))
  - 251350.KS (KODEX 선진국 MSCI World)
  - 476760.KS (ACE 미국30년국채액티브)
  - 308620.KS (KODEX 미국10년국채선물)
  - PDBC (Invesco Optimum Yield Diversified Commodity)
  - 182480.KS (TIGER 미국MSCI리츠(합성H))

### 1-2. Risk Off (TIP < BIL)

- Allocate 50% of the portfolio to the asset with higher momentum between:
  - 308620.KS (KODEX 미국10년국채선물) or
  - 0048J0.KS (KODEX 미국머니마켓액티브)

---

## 2. Defense First 한국형 (50%)

- Calculate Momentum = (1M + 3M + 6M + 12M) / 4
- Measure the momentum of 4 defensive assets:
  - 476760.KS (ACE 미국30년국채액티브)
  - 411060.KS (ACE KRX금현물)
  - PDBC (Invesco Optimum Yield Diversified Commodity)
  - UUP (Invesco DB US Dollar Index)
- Rank the four defensive assets from highest to lowest momentum
- Allocate within the 50% sleeve: **40%** (rank 1), **30%** (rank 2), **20%** (rank 3), **10%** (rank 4)
  - Absolute portfolio weights: 20%, 15%, 10%, 5%
- If a defensive asset's momentum < BIL momentum, replace that portion with the higher-momentum asset between:
  - 0060H0.KS (TIGER 토탈월드스탁액티브) or
  - 0048J0.KS (KODEX 미국머니마켓액티브)

---

## 3. Further Rules

After calculating weights from #1 and #2:

1. **UUP → 0048J0.KS**: Allocate any UUP portion to KODEX 미국머니마켓액티브
2. **Gold rule**: Measure 12-month total return of:
   - 411060.KS (ACE KRX금현물) and 308620.KS (KODEX 미국10년국채선물)
   - If **both** 12M returns are positive → keep gold allocation
   - Otherwise → move gold portion to 0048J0.KS (KODEX 미국머니마켓액티브)
3. **PDBC allocation**: Allocate any PDBC portion equally into 261220.KS (KODEX WTI원유선물(H)), 276000.KS (TIGER 글로벌자원생산기업(합성 H)), and 137610.KS (TIGER 농산물선물Enhanced(H)).

---

## Scheduling

- **GitHub Actions cron**: `0 9 25-31 * *` (09:00 UTC = 18:00 KST, days 25–31)
- **Script guard**: Only executes on the **day before the last business day** of the month (KST)
- **Idempotency**: Skips if an allocation record already exists for the current month

---

## Setup

### Secrets (GitHub → Settings → Secrets)

| Secret | Description |
|--------|-------------|
| `TELEGRAM_BOT_TOKEN` | Telegram Bot API token |
| `TELEGRAM_CHAT_ID` | Target chat/channel ID |

### Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python allocation.py
```
