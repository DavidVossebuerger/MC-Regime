"""Expert-defined economic regime provider for EUR/USD.

Defines 8 mutually exclusive EUR/USD regimes based on combined ECB + Fed
policy regimes and FX-specific carry/volatility dynamics. Regime boundaries
follow the monetary policy regime classifications documented in
'Ökonomische Regime — EUR, USD, JPY, GBP (Historie)' and academic literature
on ECB monetary regimes (Giordano & Goghie 2023).

Regimes (priority order, exhaustive coverage 2003-2025):
  R1_PRE_GFC            : 2003-01 to 2008-08  (Classic carry: EZB 2-4.25%, Fed 1-5.25%)
  R2_GFC                : 2008-09 to 2009-06  (Lehman → TARP recovery, VIX 80)
  R3_EURO_CRISIS_ERA    : 2009-07 to 2012-08  (GFC recovery + Euro-Crisis + OMT)
  R4_TAPER_TRANSITION   : 2012-09 to 2014-05  (Post-Eurocrisis + Bernanke taper, USD rally)
  R5_NIRP_DOVISH        : 2014-06 to 2019-12  (EZB NIRP, Fed liftoff then plateau, EUR 1.05-1.40)
  R6_COVID              : 2020-01 to 2021-12  (PEPP, coordinated easing, EUR 1.07-1.23)
  R7_INFLATION_SHOCK    : 2022-01 to 2023-09  (Synchronised hawkish, EUR/USD parity 0.96)
  R8_DISINFLATION       : 2023-10 to 2025-12  (Plateau → EZB cuts first, EUR recovery)

These are 8 distinct economic regimes; each is treated as a "Bauklotz" type
when resampling for the Monte Carlo bootstrap. The blocks in each regime
preserve within-regime temporal structure (autocorrelation, vol clustering)
while the MC re-arrangement varies the between-regime ordering.
"""

from dataclasses import dataclass
from typing import List
import pandas as pd

from mc_regime.regimes.base import Block


_REGIME_WINDOWS = [
    # (id, name, start, end) — exhaustive coverage 2003-01 to 2025-12
    (1, "PRE_GFC",            "2003-01-01", "2008-08-31"),
    (2, "GFC",                "2008-09-01", "2009-06-30"),
    (3, "EURO_CRISIS_ERA",    "2009-07-01", "2012-08-31"),
    (4, "TAPER_TRANSITION",   "2012-09-01", "2014-05-31"),
    (5, "NIRP_DOVISH",        "2014-06-01", "2019-12-31"),
    (6, "COVID",              "2020-01-01", "2021-12-31"),
    (7, "INFLATION_SHOCK",    "2022-01-01", "2023-09-30"),
    (8, "DISINFLATION",       "2023-10-01", "2025-12-31"),
]


@dataclass
class ExpertRegime:
    """A single expert-defined regime assignment for one month."""
    date: pd.Timestamp
    regime_id: int
    regime_name: str


class ExpertRegimeProvider:
    """Assigns each month to one of 8 expert-defined EUR/USD regimes.

    Uses date-window logic only (no FRED input required). This is intentional:
    the regimes are defined by known policy decision dates and crisis windows,
    not by data-driven clustering. The mapping is fully deterministic and
    reproducible.

    Args:
        fred_levels_path: Optional FRED path (not used in date-only logic,
            kept for API compatibility with HMMBlockProvider).
    """

    def __init__(self, fred_levels_path: str = None):
        self.fred_levels_path = fred_levels_path
        self._regimes: List[ExpertRegime] = []

    def fit(self) -> "ExpertRegimeProvider":
        """Generate regime assignments covering 2003-01 to 2026-12."""
        # Generate all month-end dates in coverage range
        dates = pd.date_range(start="2003-01-01", end="2026-12-31", freq="ME")
        regimes = []
        for ts in dates:
            rid, rname = self._classify(ts)
            regimes.append(ExpertRegime(date=ts, regime_id=rid, regime_name=rname))
        self._regimes = regimes
        return self

    def _classify(self, ts: pd.Timestamp) -> tuple:
        """Classify a single month-end into a regime (priority-ordered windows).

        Windows are exhaustive over 2003-01 to 2025-12. Gaps between
        explicit event windows are NOT present by design:
        R1 covers 2003-2008, R2 2008-2009, R3 2010-2012, R4 2013-2014,
        R5 2014-2019, R6 2020-2021, R7 2022-2023, R8 2023-2025.
        Months outside this range (e.g. 2026) get a default R8.
        """
        for rid, rname, start, end in _REGIME_WINDOWS:
            if pd.Timestamp(start) <= ts <= pd.Timestamp(end):
                return rid, rname
        # Outside 2003-2025: default to last regime (R8)
        return 8, "DISINFLATION"

    def get_blocks(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> List[Block]:
        """Return contiguous Block objects for each regime in the date range.

        Adjacent months with the same regime_id are merged into one Block.
        """
        relevant = [r for r in self._regimes
                    if start_date <= r.date <= end_date]
        if not relevant:
            return []
        blocks: List[Block] = []
        cur_id = relevant[0].regime_id
        cur_start = relevant[0].date
        cur_end = relevant[0].date
        for r in relevant[1:]:
            if r.regime_id == cur_id:
                cur_end = r.date
            else:
                blocks.append(Block(start=cur_start, end=cur_end, regime_id=cur_id))
                cur_id = r.regime_id
                cur_start = r.date
                cur_end = r.date
        blocks.append(Block(start=cur_start, end=cur_end, regime_id=cur_id))
        return blocks

    def get_regime_series(self) -> pd.Series:
        """Return a Series of regime_id indexed by month-end date."""
        return pd.Series(
            [r.regime_id for r in self._regimes],
            index=pd.DatetimeIndex([r.date for r in self._regimes]),
            name="regime_id",
        )

    @property
    def n_states(self) -> int:
        return 8

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame summarising each regime's period and count."""
        rows = []
        for r in self._regimes:
            rows.append({"regime_id": r.regime_id, "regime_name": r.regime_name,
                         "date": r.date})
        df = pd.DataFrame(rows)
        summary = df.groupby(["regime_id", "regime_name"]).agg(
            start=("date", "min"), end=("date", "max"), n_months=("date", "count")
        ).reset_index()
        return summary
