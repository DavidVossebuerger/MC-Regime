# FRED Data Card

**Source:** Federal Reserve Economic Data (FRED) via `fredapi` library
**Loader:** `mc_regime.data.fred_loader`
**API Keys:** Two keys loaded from `.env` (`FRED_API_KEY`, `FRED_API_KEY_BACKUP`) with automatic rotation on rate-limit
**Generated:** 2026-06-03

## Cache Architecture

Data is fetched **once from FRED** and cached locally as CSV. Subsequent reads load from cache (no FRED call, no rate limit).

- **First-time fetch:** `python scripts/load_fred_data.py --refresh`
- **Subsequent reads:** `python scripts/load_fred_data.py` (default: read from cache)
- **In code:** `panel = load_fred_panel(csv_path)` (no API call)

## Series Successfully Fetched (24)

| Currency | Indicator | FRED ID | n_obs | Range |
|---|---|---|---|---|
| USD | CPI | CPIAUCSL | 279 | 2003-01 → 2026-04 |
| USD | Core CPI | CPILFESL | 279 | 2003-01 → 2026-04 |
| USD | PCE | PCEPI | 280 | 2003-01 → 2026-04 |
| USD | Core PCE | PCEPILFE | 280 | 2003-01 → 2026-04 |
| USD | Fed Funds | FEDFUNDS | 281 | 2003-01 → 2026-05 |
| USD | Unemployment | UNRATE | 279 | 2003-01 → 2026-04 |
| USD | NFP | PAYEMS | 280 | 2003-01 → 2026-04 |
| USD | GDP | GDPC1 | 93 | 2003-01 → 2026-01 |
| USD | 10Y Treasury | DGS10 | 5857 | 2003-01 → 2026-06 (daily) |
| USD | Industrial Production | INDPRO | 280 | 2003-01 → 2026-04 |
| EUR | CPI | CP0000EZ19M086NEST | 280 | 2003-01 → 2026-04 |
| EUR | Core CPI | 00XTOBEZ19M086NEST | 280 | 2003-01 → 2026-04 |
| EUR | Policy Rate | ECBDFR | 8555 | 2003-01 → 2026-06 (daily) |
| EUR | Unemployment | LRHUTTTTEZM156S | 241 | 2003-01 → 2023-01 |
| EUR | GDP | CLVMNACSCAB1GQEA19 | 93 | 2003-01 → 2026-01 |
| GBP | CPI | GBRCPIALLMINMEI | 267 | 2003-01 → 2025-03 |
| GBP | Core CPI | GBRCPICORMINMEI | 267 | 2003-01 → 2025-03 |
| GBP | Policy Rate | BOERUKM | 169 | 2003-01 → 2017-01 |
| GBP | Unemployment | LRHUTTTTGBM156S | 276 | 2003-01 → 2025-12 |
| GBP | GDP | NGDPRSAXDCGBQ | 93 | 2003-01 → 2026-01 |
| JPY | CPI | CPALTT01JPM657N | 222 | 2003-01 → 2021-06 |
| JPY | Policy Rate | IRSTCB01JPM156N | 252 | 2003-01 → 2023-12 |
| JPY | GDP | JPNRGDPEXP | 93 | 2003-01 → 2026-01 |
| CHF | CPI | CHECPIALLMINMEI | 268 | 2003-01 → 2025-04 |
| CHF | GDP | CLVMNACSAB1GQCH | 92 | 2003-01 → 2025-10 |

## Series Not in FRED

- **JPY Core CPI, JPY/CHF Unemployment:** OECD mirror unavailable
- **CHF Policy Rate:** exists but rate-limited on current run (retry succeeds on next `--refresh`)

## Files

| File | Format | Content |
|---|---|---|
| `fred_panel_levels.csv` | Wide CSV, date as first column | 8555 daily rows × 24 series (mixed frequencies) |
| `fred_panel_levels.parquet` | Parquet | Same data, faster reload |
| `fred_panel_yoy.csv` / `.parquet` | YoY-transformed for 13 index series | CPI, PCE, GDP |
| `fred_series_long.csv` | Long format | 21,799 rows of (date, series, value) — easy to filter / pivot |
| `coverage.json` | Per-series statistics |
| `comparison_calendar.json` | FRED vs Calendar match per series |

## FRED vs Calendar Coverage

| Use Case | Best Source |
|---|---|
| Long-history macro trends (2003-2017) | **FRED** |
| Surprise / event-based signals | **Calendar** |
| Event-window construction (rate decisions, CPI) | **Calendar** |
| Daily/weekly real-time features | **FRED** (more granular) |
| Event-type breadth (ISM, JOLTS, etc.) | **Calendar** |
| HMM input 2003-2017 | **FRED** |
| HMM input 2018+ | **FRED for length, Calendar for event-specific** |

## FRED dominates for length and breadth:

- USD CPI: FRED 279 obs vs Calendar 98 (3x)
- EUR CPI: FRED 280 vs Calendar 64 (4x)
- GBP CPI: FRED 267 vs Calendar 95 (3x)
- JPY CPI: FRED 222 vs Calendar 4 (55x)
- CHF CPI: FRED 268 vs Calendar 48 (6x)

## Construction

1. `FREDClient` reads `FRED_API_KEY` and `FRED_API_KEY_BACKUP` from `.env`.
2. `fetch_and_cache(force_refresh=True)` fetches all 24 series with 2-key rotation, then writes CSVs (wide + long) + parquets.
3. Subsequent reads via `load_fred_panel(csv_path)` or `fetch_and_cache(force_refresh=False)`.
4. Index series (CPI, PCE, GDP) get YoY transformation as separate parquet.
5. Calendar comparison: matching by currency prefix and indicator substring.

## API Limits & Cache Strategy

- FRED allows ~120 requests/minute per key.
- With 25 series and 2 retry passes, we approach the limit on big runs.
- **Two-key rotation** doubles our budget.
- **CSV cache** eliminates the problem for downstream work — the only FRED call is the one-time fetch.

## Known Issues

- **Frequency mixing:** Panel mixes daily (ECBDFR, 10Y), monthly (CPI, NFP), quarterly (GDP). Downstream code must resample to common frequency (e.g. month-end) before HMM.
- **FRED lags real-time:** Series like CPI are reported with ~1 month delay.
- **GDP quarterly:** Only 93 obs instead of 280+ for monthly. Resample or use monthly proxy (Industrial Production).
- **Daily series dominate row count:** 8555 rows = driven by daily ECB Policy Rate + 10Y Treasury. Most other series have 200-300 obs.
