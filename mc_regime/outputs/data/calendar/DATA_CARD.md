# Economic Calendar Data Card

**Source:** Forex Factory CSV dumps (manually downloaded 2018-2026)
**Loader:** `mc_regime.data.calendar_loader`
**Generated:** 2026-06-03

## Files

| File | Content |
|---|---|
| `calendar_all.parquet` | All 40k+ events, 10 currencies, 2018-01 → 2026-04 |
| `calendar_filtered.parquet` | USD, EUR, GBP, JPY, CHF with impact ≥ Medium (8,630 events) |
| `surprise_index_{USD,EUR,GBP,JPY,CHF}.parquet` | Daily surprise index per currency |
| `summary.json` | Per-currency statistics |

## Schema

```
timestamp       pd.Timestamp (UTC)
currency        str (e.g. "USD")
event           str (e.g. "CPI y/y")
impact          str in {High, Medium, Low, Non-economic}
actual          str (original Forex Factory value)
forecast        str (original)
previous        str (original)
actual_num      float | None (parsed: 8.79M → 8_790_000, 0.1% → 0.1)
forecast_num    float | None
previous_num    float | None
surprise        float | None (actual_num - forecast_num)
```

## Coverage

| Currency | Events | Days (with idx) | Range |
|---|---|---|---|
| USD | 7,304 | 3,009 | 2018-01-02 → 2026-04-07 |
| EUR | 6,297 | 3,010 | 2018-01-02 → 2026-04-08 |
| GBP | 3,473 | 3,010 | 2018-01-02 → 2026-04-08 |
| JPY | 2,736 | 3,008 | 2018-01-04 → 2026-04-08 |
| CHF | 817 | 3,009 | 2018-01-03 → 2026-04-08 |

## Surprise Index Construction

1. Per event type, z-score the surprise (actual - forecast) within that event.
2. Sum the z-scored surprises within each calendar day per currency.
3. Apply a 60-day rolling standardisation to the daily sum.
4. Result: a daily, comparable surprise index with mean ≈ 0 and std ≈ 1.

## Known Issues

- **Coverage gap (2018-01-01):** Forex Factory only retains ~3 months of history on its website, so our dump starts 2018. Our FX data starts 2003 (EURUSD). The surprise index cannot be constructed for 2003-2017. This affects the HMM's H0 specification — see "Implications" below.
- **Mixed units:** Some events report in percent (0.1%), others in thousands (164K), millions (8.79M), billions, or plain units. Per-event z-scoring addresses this, but cross-event comparisons are not meaningful.
- **Non-economic events:** "Bank Holiday" rows are tagged "Non-economic" and excluded by the impact filter.

## Implications for the Project

The original Spec (Section 9.1) listed "Economic Surprise Index" as one of 5 HMM features. Without surprise data for 2003-2017, the HMM can either:
1. Use only the 4 other features (CPI diff, real rate diff, GDP diff, FX vol, carry — note: Spec had 5 already, this would drop to 4).
2. Use surprise for 2018-2026 only and exclude earlier years from the HMM training.
3. Use FRED as an alternative source for US surprise data back to ~1999.

**Recommendation:** Use option 3 — pull US data (CPI, NFP, retail sales) from FRED's `acts` vs `forecasts` releases for 2003-2017. This gives a US-only surprise index. Non-US surprises stay post-2018.

## Numeric Parser Rules

| Input | Output |
|---|---|
| `47.4` | 47.4 |
| `-0.4` | -0.4 |
| `164K` | 164,000 |
| `8.79M` | 8,790,000 |
| `1.2B` | 1,200,000,000 |
| `0.5T` | 500,000,000,000 |
| `0.1%` | 0.1 (kept as percentage points) |
| `1.63\|3.0` | 2.315 (range midpoint) |
| `""` or `nan` or `"-"` or `"abc"` | None |
