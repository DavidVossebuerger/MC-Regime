# Macro Time-Series Data Card

**Source:** Derived from Forex Factory calendar CSVs (Actual values)
**Loader:** `mc_regime.data.macro_loader`
**Generated:** 2026-06-03

## Output

`macro_wide.parquet` — single DataFrame with one row per month-end, columns named `{CCY}_{EVENT}`.

## Coverage

| Stat | Value |
|---|---|
| Months | 100 (2018-01-31 → 2026-04-30) |
| Series | 23 (across 5 currencies) |
| Median coverage | 80% |
| Min coverage | 4% (sparse JPY/CHF events) |
| Max coverage | 100% (USD ISM PMI) |

## Per-Currency Series

### USD (10 series, all with 90+ obs)

| Event | n | Range | Mean | Notes |
|---|---|---|---|---|
| Unemployment Rate | 102 | 2018-01 → 2026-04 | 4.54% | Standard |
| Non-Farm Employment Change | 103 | 2018-01 → 2026-04 | 131,893 | In persons |
| ISM Manufacturing PMI | 104 | 2018-01 → 2026-04 | 52.5 | Index > 50 = expansion |
| CPI y/y | 98 | 2018-01 → 2026-03 | 3.47% | Headline inflation |
| Core Retail Sales m/m | 99 | 2018-01 → 2026-04 | 0.39% | |
| Retail Sales m/m | 99 | 2018-01 → 2026-04 | 0.39% | |
| Core CPI m/m | 97 | 2018-01 → 2026-03 | 0.28% | |
| PPI m/m | 93 | 2018-01 → 2026-03 | 0.23% | |
| Core PCE Price Index m/m | 90 | 2018-01 → 2026-03 | 0.24% | Fed's preferred metric |

### EUR (7 series, 66-90 obs)

| Event | n | Range | Mean |
|---|---|---|---|
| German ifo Business Climate | 90 | 2018-01 → 2026-01 | 92.97 |
| CPI Flash Estimate y/y | 80 | 2018-01 → 2026-03 | 3.36% |
| Flash Manufacturing PMI | 80 | 2018-01 → 2025-10 | 50.14 |
| Flash Services PMI | 80 | 2018-01 → 2025-10 | 50.87 |
| Core CPI Flash Estimate y/y | 70 | 2018-01 → 2026-03 | 2.76% |
| Main Refinancing Rate | 66 | 2018-01 → 2026-03 | 1.42% (ECB rate) |
| German Prelim CPI m/m | 66 | 2018-01 → 2026-03 | 0.16% |

### GBP (4 series, 17-95 obs)

| Event | n | Range |
|---|---|---|
| CPI y/y | 95 | 2018-01 → 2026-03 |
| GDP m/m | 71 | 2018-07 → 2026-03 |
| Claimant Count Change | 60 | 2018-01 → 2026-03 |
| Unemployment Rate | 17 | 2018-01 → 2019-05 (sparse) |

### JPY (1 series)

| Event | n | Range |
|---|---|---|
| National Core CPI y/y | 4 | 2023-03 → 2025-03 (sparse) |

### CHF (2 series, 28-48 obs)

| Event | n | Range |
|---|---|---|
| CPI m/m | 48 | 2020-01 → 2026-04 |
| SNB Policy Rate | 28 | 2019-06 → 2026-03 |

## Schema

`macro_wide.parquet`:
```
timestamp              pd.Timestamp (month-end, tz-naive)
USD_CPI y/y            float | NaN
USD_Core CPI m/m       float | NaN
USD_Unemployment Rate  float | NaN
... (23 columns total)
```

`coverage.json`:
```json
{
  "USD_CPI y/y": {
    "n_obs": 98,
    "first": "2018-01-12",
    "last": "2026-03-11",
    "mean": 3.471
  },
  ...
}
```

## Construction Method

For each `(currency, event)` pair:
1. Filter calendar to that pair with `impact >= Medium`.
2. Take `actual_num` (parsed numeric: 8.79M → 8,790,000; 0.1% → 0.1; "1.63|3.0" → midpoint).
3. Drop events without a numeric actual.
4. Floor timestamp to month-end.
5. Per month, take the last release (or mean if multiple in same month).
6. Outer-merge all `(currency, event)` series on month.

## Implications for the Project

- **USD macro features for HMM:** strong coverage. CPI y/y, Unemployment Rate, ISM PMI, NFP all suitable.
- **EUR macro features:** good coverage. CPI Flash, Main Refinancing Rate, German ifo all suitable.
- **GBP macro features:** medium. CPI y/y works; Unemployment Rate is sparse.
- **JPY / CHF:** event-name matching is poor. Either the user provides additional calendar data with more events, or these pairs get fewer macro features in the HMM.
- **Real Interest Rate:** can be derived as `Main Refinancing Rate (EUR) - CPI Flash Estimate y/y (EUR)`. USD: `Fed Funds Rate - CPI y/y`. The Fed Funds Rate series is not in the calendar — could be added from a Fed minutes/event calendar, or approximated.
- **Carry Differential:** not in calendar; must be derived from FX forwards data (not currently in dataset).

## Known Issues

- **JPY/CHF Event-Namen-Mismatch:** Viele JPY/CHF-Events haben im Calendar andere Namen als in `DEFAULT_EVENTS` angenommen. Beispiel: "BoJ Policy Rate" gibt es nicht, sondern "BoJ Policy Statement" oder ähnliches. Falls vollständige JPY/CHF-Coverage gewünscht: erweitere `DEFAULT_EVENTS` mit korrekten Event-Namen aus dem Calendar.
- **Datenlücke vor 2018:** Wie beim Surprise-Index: Calendar-Daten reichen nur 2018+. FX-Daten reichen 2003+. Macro-Features für HMM sind also nur 2018+ verfügbar.
- **Mixed Units:** Manche Events in Personen (164K), andere in % (0.1%), andere in Index (52.5). Z-Score-Normalisierung pro Event-Type ist nötig, bevor sie ins HMM gehen.
