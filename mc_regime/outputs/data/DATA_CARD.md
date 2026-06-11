# FX Data Card

**Source:** Dukascopy historical tick data (https://www.dukascopy.com/)
**Loader:** `mc_regime.data.fx_loader`
**Generated:** 2026-06-03

## Pairs Available

| Pair | Files | Time Range | Notes |
|---|---|---|---|
| EURUSD | 1 | 2003-05-04 → 2025-09-06 | Single file in project root |
| GBPUSD | 93 | 2007-01-01 → 2026-06-01 | New + old naming convention deduped |
| USDJPY | 78 | 2007-01-01 → 2026-06-01 | New naming convention only |

**Missing:** CHFUSD — no data available. Project will run on 3 pairs unless data is added.

## Pairs Excluded (Indices, not FX)

- `usa500idxusd` (S&P 500, 2011-09 → 2026-06)
- `usatechidxusd` (NASDAQ Tech, 2011-09 → 2026-06)

## File Format (raw)

Dukascopy CSV with columns:
```
timestamp, open, high, low, close
<epoch_ms>, <float>, <float>, <float>, <float>
```

`timestamp` is in milliseconds since Unix epoch (UTC).

## Unified Format (output)

Parquet files with columns:
```
timestamp, open, high, low, close, pair
```

`timestamp` is `pd.Timestamp` with `tz=UTC`.

## Output Files

For each pair, two parquet files:

| File | Frequency | Use |
|---|---|---|
| `{PAIR}_m30_clean.parquet` | 30 min, OHLC | High-frequency analysis |
| `{PAIR}_1h.parquet` | 1 hour, OHLC | Default analysis frequency |
| `{PAIR}_1D.parquet` | daily, OHLC | Long-horizon / sanity check |
| `{PAIR}_1ME.parquet` | month-end, OHLC | Coarse aggregates |

## Cleaning Pipeline

1. **NaN drop:** rows with NaN in OHLC are removed.
2. **Zero/negative drop:** rows with non-positive prices are removed.
3. **Weekend drop:** Saturday and Sunday bars are removed (FX is 24/5).
4. **Outlier drop:** rows where log-return exceeds 8 × rolling std (window = 500 bars).
5. **Deduplication:** multiple files per pair are concatenated and deduped on `timestamp` (last-write-wins).
6. **Sort:** ascending by `timestamp`.

## Drop Statistics

| Pair | Raw | Cleaned | Dropped | % |
|---|---|---|---|---|
| EURUSD | 390,486 | 278,651 | 111,835 | 28.6% |
| GBPUSD | 323,882 | 231,168 | 92,714 | 28.6% |
| USDJPY | 334,396 | 238,645 | 95,751 | 28.6% |

The 28.6% drop is almost entirely weekend bars (2/7 ≈ 28.6%). Outlier removal affects <0.1% of bars.

## Regeneration

```bash
cd /root/research/MC-Regime/mc_regime
python3 scripts/load_fx_data.py --freq 1h
```

## Known Issues

- **End-of-data flat bars:** The last few 30-min bars before the data ends (e.g. EURUSD 2025-09-05 22:00-23:30) have identical OHLC values. These are likely market-closed periods that escaped the weekend filter. May need additional cleaning in downstream analysis.
- **JPY quote convention:** USDJPY in Dukascopy is "1 USD = X JPY" (e.g. 119.03). Spec's "JPY/USD" notation is a misnomer — actual pair is USDJPY.
- **CHFUSD missing:** 4th pair from the spec is unavailable in the raw data. Project scope needs adjustment.
