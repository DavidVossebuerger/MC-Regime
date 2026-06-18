# Regime-Preserving Bootstrap for FX Fair-Value Models
[![SSRN Paper](https://img.shields.io/badge/SSRN-Paper-blue?style=for-the-badge&logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA1MTIgMjA1Ij4KICA8cmVjdCB3aWR0aD0iNTEyIiBoZWlnaHQ9IjIwNSIgZmlsbD0iIzFCNUUyMCIvPgogIDx0ZXh0IHg9IjI1NiIgeT0iMTE1IiBmb250LWZhbWlseT0iQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIwIiBmb250LXdlaWdodD0iYm9sZCIgZmlsbD0id2hpdGUiIHRleHQtYW5jaG9yPSJtaWRkbGUiPlNTUk48L3RleHQ+Cjwvc3ZnPg==&logoColor=white)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6890880)
Methodology, metrics, and empirical validation of a regime-preserving bootstrap procedure for FX fair-value strategies (BEER, FEER, PPP).

**Paper:** [`docs/paper.pdf`](docs/paper.pdf) (14 pages, English)
**Author:** David Vossebürger (<david.vossebuerger@proton.me>)

## Overview

Standard block-bootstrap methods (IID, Moving Block, Stationary Block) destroy the economic regime structure of macro-FX data and thereby bias Sharpe ratio inference. This project proposes two regime-aware resamplers:

- **Regime-Uniform**: resampling of entire regime blocks uniformly at random.
- **Regime-Markov**: resampling of entire regime blocks via a calibrated Markov transition matrix.

The two methods are validated against three naive baselines (IID, MBB, SB) on three currency pairs (EURUSD, GBPUSD, USDJPY) at $B = 1999$ bootstrap replicates with a BEER fair-value pricer.

**Key result:** Regime-Markov reduces the cross-pool Sharpe loss by a factor of $1.35\times$ over Regime-Uniform and by $4.59\times$ over the naive IID baseline. The regime-aware-vs-naive gap is the dominant contribution; the calibrated Markov transition structure provides the additional incremental gain over plain block resampling. All four Spec-$\S$~5 decision criteria are met; the calibrated transition matrix passes the $\S$~11.7 stationarity gate (TV drift < 0.10) without any cap loosening.

## Quick Start

```bash
git clone https://github.com/DavidVossebuerger/MC-Regime.git
cd MC-Regime
pip install -e .

python3 -u mc_regime/scripts/run_poc.py \
    --replications 1999 \
    --pairs all \
    --pricer beer \
    --calibrate-target empirical \
    --output outputs/runs/main
```

The FRED panel cache and FX parquets are included in the repository under `mc_regime/outputs/data/`; no API key, no data download required.

Wall-clock: approximately 25 minutes on a modern multi-core CPU.

## Code Structure

```
mc_regime/
├── src/mc_regime/
│   ├── data/                 # FX, FRED, calendar, macro loaders
│   ├── regimes/              # ExpertRegimeProvider, HMM, transition matrix, whitelist
│   ├── bootstrap/            # IID, MBB, SB, Regime samplers + engine
│   ├── structure/            # L1-L4 + arrangement metrics
│   ├── inference/            # Paired Wilcoxon, FDR, BCa (deprecated)
│   ├── pricers/              # BEER (FMOLS, HAC)
│   └── scripts/              # Reproduction scripts
├── scripts/
│   └── run_poc.py            # End-to-end pipeline
├── tests/unit/               # 23 unit tests
├── outputs/
│   ├── data/                 # FRED panel + 3 FX parquets (committed)
│   └── runs/                 # Per-run outputs (gitignored)
└── docs/
    └── paper.pdf            # The paper
```

## Tests

```bash
python3 -m pytest tests/ -x
```

23 unit tests covering the transition matrix calibration, the
HMMSdetector pair-aware feature builder, the Wilcoxon inference
helper, the arrangement-distance metrics, the BEER pricer, and
the sinkhorn cap-sensitivity sweep.

## Reproduction (main run)

```bash
python3 -u mc_regime/scripts/run_poc.py \
    --replications 1999 \
    --pairs all \
    --pricer beer \
    --calibrate-target empirical \
    --output outputs/runs/main
```

The output directory contains:

- `decision.json` — GO/NO-GO + L0 + arrangement summary
- `arrangement_distances.csv` — 5 methods × 199 sims × 3 metrics
- `test_results.json` — 24 Wilcoxon + paired tests (pair level)
- `fdr_adjusted.json` — BH-FDR $q$-values
- `transition_matrix_calibrated.npy` — Sinkhorn output
- `eurusd/`, `gbpusd/`, `usdjpy/` — per-pair decisions

## Methodology Highlights

- **Regime taxonomy:** 8 expert-defined EUR/USD macro-monetary
  regimes (PRE_GFC, GFC, EURO_CRISIS_ERA, TAPER_TRANSITION,
  NIRP_DOVISH, COVID, INFLATION_SHOCK, DISINFLATION)
- **BEER pricer:** Clark-MacDonald (1998) reduced form with
  FMOLS estimation and HAC standard errors
- **Bootstrap:** 5 methods compared (IID, MBB, SB, Regime-Uniform,
  Regime-Markov)
- **Metrics:** L0 (Sharpe loss), L1 (correlation), L2 (Wasserstein),
  L3 (network/spectral), L4 (temporal), arrangement (composition/
  transition)
- **Statistics:** Paired sign test with BH-FDR, smallest
  achievable $p$-value is $0.25$ for $n=3$ paired tests
- **$\S$~11.7 gate:** Sinkhorn-Knopp calibration of the transition
  matrix, TV drift must be < 0.10
- **Cap insensitivity:** TV drift is invariant for cap values
  $0.5\%$–$10\%$; the 3% cap is non-binding

## References

Clark and MacDonald (1998), Efron (1979), Politis and Romano (1994),
Politis, Romano, and Wolf (1999), Künsch (1989), DiCiccio and Efron
(1996), Lo (2002), MacDonald and Ricci (2003), Sinkhorn (1964),
Benjamini and Hochberg (1995). Full bibliography in the paper.
