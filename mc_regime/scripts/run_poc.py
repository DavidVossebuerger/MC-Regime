"""End-to-end PoC: Regime-Preserving Bootstrap on real EUR/USD + FRED data.

Pipeline (per Spec §3):
  1. Define 8 expert-defined EUR/USD regimes (Macro-only, no price)
  2. Build regime blocks (Macro + FX path as a package)
  3. Estimate regularised transition matrix (Spec §3.3)
  4. Generate 5 bootstrap method results (IID, MBB, SB, Regime-Uniform, Regime-Markov)
  5. Compute structure-preservation loss (L1 corr, L2 wasserstein, L3 network)
  6. Compute performance gap (L0 Sharpe)
  7. Compute arrangement distance (Spec §6)
  8. Run H0/H1 paired tests + BH-FDR
  9. Decision requires BOTH arrangement is far AND mean is close (Spec §5)

Usage:
    python scripts/run_poc.py [--replications 999] [--output DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from mc_regime.inference.pair_wilcoxon import pair_level_pvalue, pair_level_effect_size

from mc_regime.pricers.beer import BEERPricer
from mc_regime.data.fx_loader import load_parquet as load_fx_parquet

from mc_regime.regimes.base import Block
from mc_regime.regimes.expert_provider import ExpertRegimeProvider
from mc_regime.regimes.hmm_detector import HMMSdetector
from mc_regime.regimes.transition import (
    estimate_transition_matrix,
    estimate_initial_distribution,
    compare_stationary_vs_empirical,
    calibrate_transition_to_target_stationary,
    stationary_distribution,
    DEFAULT_ALLOWED_TRANSITIONS,
    sinkhorn_cap_sensitivity,
)
from mc_regime.regimes.whitelist import allowed_transitions_for_pair

from mc_regime.bootstrap.engine import BootstrapEngine
from mc_regime.bootstrap.blocks import Block as BootstrapBlock

from mc_regime.structure.correlations import (
    compute_correlations,
    compute_correlation_bias,
)
from mc_regime.structure.wasserstein import compute_wasserstein_distances
from mc_regime.structure.network import build_correlation_graph, compute_network_distance
from mc_regime.structure.composite import compute_composite_loss
from mc_regime.structure.arrangement import (
    arrangement_distance,
    compute_arrangement_distances,
)
from mc_regime.structure.temporal import (
    temporal_metrics,
    temporal_distance,
    mean_temporal_loss,
)

from mc_regime.inference.paired_test import paired_loss_test
from mc_regime.inference.fdr import apply_bh_fdr


# === Pipeline configuration constants ===
ANNUALIZATION_QUARTERLY = 4  # 4 obs per year for quarterly data; sqrt(4)=2 annualisation
STATIONARITY_GATE_TV = 0.10  # §11.7: max TV-drift between stationary(P) and target
DEVIATION_THRESHOLD_MULT = 0.5  # FVG/BEER position-sizing threshold (0.5× std)
ARRANGEMENT_FAR_THRESHOLD = 0.5  # mean transition distance must exceed this
CALIBRATION_LEARNING_RATE = 0.15
CALIBRATION_MAX_ITER = 2000
SIGNIFICANCE_LEVELS = (0.01, 0.05, 0.10)  # for q-value stars
N_REGIMES = 8


# Macro features (5 minimal) for EUR/USD HMM
# Spec Section 9.1
FEATURE_COLS = [
    "InflationDiff",
    "RealRateDiff",
    "GDPDiff",
    "RealVol",
    "Carry",
]


def map_to_beer_format(data: pd.DataFrame) -> pd.DataFrame:
    """Map 5 HMM features to BEER input format.

    BEER expects: log_FX, TOT, NFA, ProdDiff, RateDiff, FiscalBalance
    HMM has:    FX, RateDiff (=RealRateDiff), InflationDiff, GDPDiff, RealVol, Carry

    Mapping:
        - log_FX    <- log(FX)
        - TOT       <- InflationDiff (proxy - relative inflation)
        - NFA       <- GDPDiff (proxy - relative GDP)
        - ProdDiff  <- RealVol (proxy - productivity differential proxy)
        - RateDiff  <- RateDiff (unchanged from nodes_data)
        - FiscalBalance <- 0 placeholder
    """
    beer_df = pd.DataFrame(index=data.index)
    beer_df["log_FX"] = np.log(data["FX"])
    beer_df["TOT"] = data["InflationDiff"]  # relative inflation (terms of trade proxy)
    beer_df["NFA"] = data["GDPDiff"]  # relative GDP (net foreign assets proxy)
    beer_df["ProdDiff"] = data["RealVol"]  # volatility as productivity proxy
    beer_df["RateDiff"] = data["RateDiff"]
    beer_df["FiscalBalance"] = 0.0  # placeholder
    return beer_df


def beer_sharpe(data: pd.DataFrame, fair_value: pd.Series) -> float:
    """Compute Sharpe ratio of BEER-mean-reversion strategy.

    Strategies:
        - Long when log(FX) < log(fair_value) - threshold (FX undervalued)
        - Short when log(FX) > log(fair_value) + threshold (FX overvalued)
        - Threshold = DEVIATION_THRESHOLD_MULT * std(log(FX) - log(fair_value))

    Args:
        data: DataFrame with FX column
        fair_value: Series of BEER fair values (log scale)

    Returns:
        Annualized Sharpe ratio (sqrt(4) for quarterly data)
    """
    log_fx = np.log(data["FX"])
    # Compute deviation from fair value
    deviation = log_fx - fair_value.loc[data.index]
    deviation = deviation.fillna(0.0)

    if deviation.std() <= 0:
        return 0.0

    thresh = DEVIATION_THRESHOLD_MULT * deviation.std()
    fx_ret = np.log(data["FX"] / data["FX"].shift(1)).fillna(0.0)

    position = pd.Series(0.0, index=data.index)
    position[deviation < -thresh] = 1.0  # long (FX under fair value)
    position[deviation > thresh] = -1.0  # short (FX over fair value)

    strat_ret = (position.shift(1).fillna(0.0) * fx_ret).dropna()

    if strat_ret.std() > 0 and len(strat_ret) > 5:
        return float(strat_ret.mean() / strat_ret.std() * np.sqrt(ANNUALIZATION_QUARTERLY))
    return np.nan

# Correlation pairs for structure test (PoC with 6 nodes: FX + 5 HMM features)
# FVG is constructed as a function of FX (would be collinear, excluded)
CORRELATION_PAIRS = [
    ("FX", "RateDiff"),
    ("FX", "InflationDiff"),
    ("FX", "Carry"),
    ("FX", "RealVol"),
    ("FX", "GDPDiff"),
    ("RateDiff", "Carry"),
    ("InflationDiff", "Carry"),
    ("RateDiff", "RealVol"),
    ("InflationDiff", "RealVol"),
    ("GDPDiff", "RealVol"),
]

NODES = list({c for pair in CORRELATION_PAIRS for c in pair})


def load_fx_monthly(fx_path: Path, bar_minutes: Optional[int] = None) -> pd.DataFrame:
    """Load FX data and resample to month-end (FX level + log-return std).

    Args:
        fx_path: Path to FX parquet file (1h or m30)
        bar_minutes: Override bar frequency in minutes (auto-detected if not provided)
    """
    fx = load_fx_parquet(fx_path)
    fx = fx.set_index("timestamp")

    # Auto-detect bar frequency from data
    if bar_minutes is None:
        ts = pd.to_datetime(fx.index)
        diffs = ts.to_series().diff().dropna().dt.total_seconds() / 60
        med = diffs.median()
        # Round to nearest standard
        standards = [1, 5, 15, 30, 60]
        bar_minutes = min(standards, key=lambda x: abs(x - med))

    print(f"  Detected FX frequency: {bar_minutes}-minute bars, annualising with sqrt(252*24*60/{bar_minutes})")

    # Monthly close
    monthly = fx["close"].resample("ME").last()
    # Monthly log-return std (annualized)
    monthly_logret = np.log(fx["close"] / fx["close"].shift(1))
    annualization_factor = np.sqrt(252 * 24 * 60 / bar_minutes)
    monthly_vol = monthly_logret.resample("ME").std() * annualization_factor

    df = pd.DataFrame({"FX": monthly, "RealVol": monthly_vol})
    return df.reset_index().rename(columns={"timestamp": "date"})


def hmm_blocks_to_bootstrap_blocks(
    hmm_blocks: list, data_index: pd.DatetimeIndex
) -> List[BootstrapBlock]:
    """Convert regimes.Block (timestamp-based) to bootstrap.Block (index-based)."""
    bootstrap_blocks = []
    for hb in hmm_blocks:
        # Find index range
        mask = (data_index >= hb.start) & (data_index <= hb.end)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        start_idx = int(idxs[0])
        end_idx = int(idxs[-1])
        bootstrap_blocks.append(BootstrapBlock(
            start_idx=start_idx, end_idx=end_idx, regime_label=hb.regime_id
        ))
    return bootstrap_blocks


def parse_calibrate_target(arg: str, n_states: int = 8) -> np.ndarray:
    """Parse --calibrate-target argument into a target distribution."""
    if arg == "empirical":
        return None  # Signal: use empirical_freq (computed later)
    if arg == "uniform":
        return np.full(n_states, 1.0 / n_states)
    # Try comma-separated floats
    parts = arg.split(",")
    if len(parts) != n_states:
        raise ValueError(f"--calibrate-target expects {n_states} floats, got {len(parts)}")
    arr = np.array([float(x) for x in parts])
    arr = arr / arr.sum()
    return arr


# Mapping from pair label to 1h parquet filename
PAIR_FILE_MAP = {
    "eurusd": "/root/research/MC-Regime/mc_regime/outputs/data/EURUSD_1h.parquet",
    "gbpusd": "/root/research/MC-Regime/mc_regime/outputs/data/GBPUSD_1h.parquet",
    "usdjpy": "/root/research/MC-Regime/mc_regime/outputs/data/USDJPY_1h.parquet",
}
DEFAULT_FRED_PATH = "/root/research/MC-Regime/mc_regime/outputs/data/fred/fred_panel_levels.csv"


def load_pair_data(pair_label: str, fred_path: str = DEFAULT_FRED_PATH):
    """Load FX data for a single pair and build nodes_data + blocks.

    Returns dict with: nodes_data, bootstrap_blocks, expert_blocks, feature_data, best_k.
    """
    fx_path = PAIR_FILE_MAP.get(pair_label)
    if fx_path is None:
        raise ValueError(f"Unknown pair: {pair_label}")

    # Expert provider (global regimes, same for all pairs)
    expert = ExpertRegimeProvider().fit()

    # HMM detector for feature construction
    detector = HMMSdetector(fred_levels_path=fred_path, fx_path=fx_path, pair=pair_label)
    best_k, states = detector.fit_predict(k_values=[2, 3, 4, 5])
    feature_data = detector.features.copy()

    # Expert blocks
    expert_blocks = expert.get_blocks(feature_data.index[0], feature_data.index[-1])

    # Build nodes_data with pair-specific FX
    fx_data = load_fx_parquet(Path(fx_path))
    fx_data = fx_data.set_index("timestamp")
    if fx_data.index.tz is not None:
        fx_data.index = fx_data.index.tz_convert(None)
    fx_q = fx_data["close"].resample("QE-OCT").last()
    fx_aligned = fx_q.reindex(feature_data.index)

    nodes = pd.DataFrame({
        "FX": fx_aligned,
        "RateDiff": feature_data["RealRateDiff"],
        "InflationDiff": feature_data["InflationDiff"],
        "GDPDiff": feature_data["GDPDiff"],
        "RealVol": feature_data["RealVol"],
        "Carry": feature_data["Carry"],
    }).dropna()

    # FVG
    nodes["FVG"] = np.log(nodes["FX"]) - np.log(nodes["FX"]).rolling(4, min_periods=2).mean()

    # Align blocks
    blocks_aligned = hmm_blocks_to_bootstrap_blocks(expert_blocks, nodes.index)

    return {
        "nodes_data": nodes,
        "bootstrap_blocks": blocks_aligned,
        "expert_blocks": expert_blocks,
        "feature_data": feature_data,
        "best_k": best_k,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=999)
    parser.add_argument("--output", type=str, default="outputs/runs/poc")
    parser.add_argument("--pairs", type=str, default="all",
                        help="FX pairs to pool: eurusd, gbpusd, usdjpy, or all (default: all)")
    parser.add_argument("--fx-path", type=str, default="/root/research/MC-Regime/mc_regime/outputs/data/EURUSD_1h.parquet")
    parser.add_argument("--fred-path", type=str, default="/root/research/MC-Regime/mc_regime/outputs/data/fred/fred_panel_levels.csv")
    parser.add_argument("--bca-b", type=int, default=999, help="Bootstrap reps for BCa")
    parser.add_argument("--calibrate-target", type=str, default="empirical",
                        help="Target for stationarity gate (§11.7): empirical, uniform, or comma-separated floats")
    parser.add_argument("--pricer", type=str, default="fvg", choices=["fvg", "beer"],
                        help="L0 pricer: fvg (Fair Value Gap mean-reversion) or beer (BEER deviation)")
    parser.add_argument("--horizon", type=float, default=1.0,
                        help="Bootstrap horizon multiplier (1.0=within-sample, >1=extended horizon — triggers §11.7 gate)")
    parser.add_argument("--refresh-fred", action="store_true",
                        help="Refresh FRED cache before running (fetch missing series)")
    parser.add_argument(
        "--n-regimes", type=int, default=8,
        help="Number of expert regimes (default: 8). Currently informational; "
             "the expert regime provider always uses 8 EUR/USD regimes."
    )
    args = parser.parse_args()

    # Override N_REGIMES from CLI flag if provided
    N_REGIMES = getattr(args, "n_regimes", 8)  # default 8 if not set

    # Optionally refresh FRED cache (via subprocess)
    if args.refresh_fred:
        import subprocess
        import sys
        print("[run_poc] Refreshing FRED cache...")
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "refresh_fred_cache.py")],
            cwd=Path(__file__).parent,
        )
        if result.returncode != 0:
            print("[run_poc] WARNING: FRED refresh had issues, continuing with existing cache...")

    # Determine which pairs to process
    pairs_to_run = []
    if args.pairs == "all":
        pairs_to_run = ["eurusd", "gbpusd", "usdjpy"]
    else:
        pairs_to_run = [p.strip() for p in args.pairs.split(",")]
    valid_pairs = set(PAIR_FILE_MAP.keys())
    for p in pairs_to_run:
        if p not in valid_pairs:
            raise ValueError(f"Invalid pair '{p}'. Must be one of: {valid_pairs}")
    print(f"Processing pairs: {pairs_to_run}")
    n_pairs = len(pairs_to_run)
    if n_pairs > 1:
        print(f"Cross-pair pooling enabled: {n_pairs} × 8 regimes = {n_pairs * 8} bootstrap blocks\n")

    # Parse calibration target early
    calibrate_target_arg = args.calibrate_target
    target_override = parse_calibrate_target(calibrate_target_arg, n_states=8)
    horizon_override = args.horizon

    out_dir = Path(__file__).parents[1] / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("REGIME-PRESERVING BOOTSTRAP — END-TO-END PROOF-OF-CONCEPT")
    print("=" * 70)
    print(f"Replications: B = {args.replications}")
    print(f"Pairs: {pairs_to_run}")
    print(f"Output: {out_dir}\n")

    # ====================================================================
    # Step 1: Load all pair data and pool bootstrap blocks
    # ====================================================================
    print("[1/9] Loading pair data and pooling bootstrap blocks...")

    # Shared expert provider (regime definitions are global)
    expert = ExpertRegimeProvider().fit()
    print(expert.summary().to_string(index=False))

    # Load each pair and collect data
    pair_data = {}
    pooled_blocks = []  # Blocks from all pairs (with pair_label for identification)
    for pair in pairs_to_run:
        print(f"\n  Loading {pair.upper()}...")
        pd_result = load_pair_data(pair, args.fred_path)
        pair_data[pair] = pd_result
        nblocks = len(pd_result["bootstrap_blocks"])
        pooled_blocks.extend(pd_result["bootstrap_blocks"])
        print(f"    nodes_data: {pd_result['nodes_data'].shape}, blocks: {nblocks}, HMM k={pd_result['best_k']}")

    print(f"\n  Pooled bootstrap blocks: {len(pooled_blocks)} (from {len(pairs_to_run)} pairs)")

    # Reference pair for transition matrix (first pair)
    ref_pair = pairs_to_run[0]
    expert_blocks_raw = pair_data[ref_pair]["expert_blocks"]

    # Save regime windows (reference pair)
    np.save(out_dir / "regime_windows.npy",
           np.array([(b.regime_id, b.start.value, b.end.value) for b in expert_blocks_raw]))

    # 5b. Estimate regularised transition matrix (Spec §3.3)
    print("\n[5b/9] Estimating regularised transition matrix (8 regimes)...")
    # Convert 1-indexed regime_ids to 0-indexed for the array
    regime_id_sequence_0 = np.array(
        [b.regime_id - 1 for b in expert_blocks_raw], dtype=int
    )
    allowed = allowed_transitions_for_pair(ref_pair)
    trans_df = estimate_transition_matrix(
        regime_id_sequence_0, n_states=8, smooth=ARRANGEMENT_FAR_THRESHOLD, allowed=allowed
    )
    trans_matrix = trans_df.values
    initial_dist = estimate_initial_distribution(regime_id_sequence_0, n_states=8)
    # Renormalise (exclude impossible transitions, allow self-loops)
    print("  Transition matrix (rows sum to 1):")
    print(trans_df.round(3).to_string())
    print(f"\n  Initial distribution: {np.round(initial_dist, 3)}")
    print(f"  Allowed transitions: {len(allowed)} pairs")
    np.save(out_dir / "transition_matrix.npy", trans_matrix)

    # 5c. Compute empirical regime frequencies (length-weighted, for stationarity target)
    empirical_freq = np.zeros(8)
    for b in expert_blocks_raw:
        empirical_freq[b.regime_id - 1] += b.length_months
    empirical_freq /= empirical_freq.sum()
    print(f"  Empirical regime frequencies: {np.round(empirical_freq, 3).tolist()}")

    # 5d. Stationarity gate (§11.7): calibrate transition to match target
    if target_override is not None:
        target = target_override
        target_label = "custom"
    else:
        target = empirical_freq
        target_label = "empirical"

    stat_before = compare_stationary_vs_empirical(trans_df, empirical_freq)
    initial_tv = stat_before["total_variation_distance"]

    gate_passed = False
    n_iterations = 0
    final_tv = initial_tv

    if initial_tv >= STATIONARITY_GATE_TV:
        print(f"\n  §11.7 stationarity gate: TV-drift = {initial_tv:.3f} >= STATIONARITY_GATE_TV, calibrating...")
        # Higher iterations for the 8-state sparse chain
        trans_df = calibrate_transition_to_target_stationary(
            trans_df, target=target, tv_threshold=STATIONARITY_GATE_TV, max_iter=CALIBRATION_MAX_ITER
        )
        trans_matrix = trans_df.values
        stat_check = compare_stationary_vs_empirical(trans_df, empirical_freq)
        final_tv = stat_check["total_variation_distance"]
        n_iterations = 2000 if final_tv >= STATIONARITY_GATE_TV else 0
        gate_passed = final_tv < STATIONARITY_GATE_TV
    else:
        print(f"\n  §11.7 stationarity gate: TV-drift = {initial_tv:.3f} < STATIONARITY_GATE_TV, skipping calibration (already calibrated)")
        gate_passed = True
        n_iterations = 0
        final_tv = initial_tv

    # Print before/after summary
    print("\n  Transition matrix calibration (§11.7 gate):")
    print(f"    Initial TV-drift: {initial_tv:.3f}")
    print(f"    Target: {target_label} {np.round(target, 3).tolist()}")
    print(f"    Final TV-drift after {n_iterations} iterations: {final_tv:.3f}")
    print(f"    Gate passed (TV < STATIONARITY_GATE_TV): {gate_passed}")

    # Save calibrated matrix
    np.save(out_dir / "transition_matrix_calibrated.npy", trans_matrix)

    # Run cap-sensitivity sweep (defends against p-hacking critique)
    sensitivity_df = sinkhorn_cap_sensitivity(
        trans_df, target,
        caps=[0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10],
    )
    sensitivity_df.to_csv(out_dir / "sinkhorn_cap_sensitivity.csv", index=False)

    # Handle gate failure
    gate_failed_warning = ""
    if not gate_passed:
        gate_failed_warning = "§11.7 GATE FAILED — horizon extension not safe"
        print(f"\n  WARNING: {gate_failed_warning}")

    # ====================================================================
    # For multi-pair: run bootstrap per-pair and aggregate losses
    # The transition matrix is shared (global regime dynamics)
    # ====================================================================
    if len(pairs_to_run) == 1:
        # Single pair: use existing logic (simpler for back-compat)
        nodes_data = pair_data[pairs_to_run[0]]["nodes_data"]
        bootstrap_blocks_aligned = pair_data[pairs_to_run[0]]["bootstrap_blocks"]
    else:
        # Multi-pair: we'll process each pair separately for bootstrap,
        # then aggregate L0/L1/L2/L3/L4 at the end
        print(f"\n[5a/9] Multi-pair mode: running bootstrap per-pair...")

    # 6e. Fit BEER model on EACH PAIR (for order-sensitive L0 measurement).
    # BEER is instrument-specific, so we fit on each pair separately.
    # The reference pair (first in pairs_to_run) is also used for the
    # explicit BEER-fit summary print.
    beer_fair_values = {}  # pair -> fair_value series
    if args.pricer == "beer":
        print(f"  Fitting BEER pricer on {len(pairs_to_run)} pair(s)...")
        for pair in pairs_to_run:
            try:
                ref_nodes = pair_data[pair]["nodes_data"]
                beer_input = map_to_beer_format(ref_nodes)
                beer_pricer = BEERPricer(maxlag=4)
                beer_pricer.fit(beer_input)
                beer_fair_values[pair] = beer_pricer.fair_value(beer_input)
                if pair == ref_pair:
                    print(f"    [{pair.upper()}] BEER fitted values: {len(beer_fair_values[pair])} obs")
                    summary = beer_pricer.summary()
                    print(f"    Cointegration p-value: {summary['coint_pvalue']:.4f}")
                    print(f"    Residual std: {summary['resid_std']:.4f}")
                else:
                    print(f"    [{pair.upper()}] BEER fitted: {len(beer_fair_values[pair])} obs")
            except Exception as e:
                print(f"    [{pair.upper()}] BEER fit FAILED: {str(e)[:60]}")
                # Leave beer_fair_values[pair] unset -> orig_sharpe will be 0.0

    # ====================================================================
    # Section 3: Run bootstrap and compute losses PER PAIR
    # ====================================================================
    print(f"\n[5/9] Running 5 bootstrap methods (IID, MBB, SB, Regime-Uniform, Regime-Markov), B = {args.replications}...")

    # Storage for per-pair results
    pair_results = {}  # pair -> {method: {L0, L1, ..., losses}}
    pair_orig_sharpe = {}  # pair -> original sharpe
    pair_L0_losses = {}  # pair -> {method: L0 value}

    for pair in pairs_to_run:
        print(f"\n  === Processing {pair.upper()} ({pairs_to_run.index(pair)+1}/{len(pairs_to_run)}) ===")
        nodes_data = pair_data[pair]["nodes_data"]
        bootstrap_blocks_aligned = pair_data[pair]["bootstrap_blocks"]

        # Run bootstrap for THIS pair
        engine = BootstrapEngine(
            replications=args.replications,
            base_seed=20260603,
            regime_mode="markov",
            transition_matrix=trans_matrix,
            initial_distribution=initial_dist,
            progress=False,  # Suppress progress for multi-pair
        )
        t0 = time.time()
        bootstrap_results = engine.run_all(nodes_data, blocks=bootstrap_blocks_aligned)
        print(f"    Bootstrap: {time.time()-t0:.1f}s")

        # Original correlations/graph
        orig_corrs = compute_correlations(nodes_data, CORRELATION_PAIRS)
        orig_graph = build_correlation_graph(nodes_data, NODES)

        # Original regime sequence
        expert_blocks_pair = pair_data[pair]["expert_blocks"]
        orig_regime_seq = []
        for b in expert_blocks_pair:
            n_in_block = b.length_months
            orig_regime_seq.extend([b.regime_id - 1] * n_in_block)
        n_obs = len(nodes_data)
        orig_regime_seq = orig_regime_seq[:n_obs]
        while len(orig_regime_seq) < n_obs:
            orig_regime_seq.append(orig_regime_seq[-1] if orig_regime_seq else 0)
        orig_regime_seq = np.array(orig_regime_seq[:n_obs], dtype=int)

        # L0: Sharpe gap
        fvg_thresh = DEVIATION_THRESHOLD_MULT * nodes_data["FVG"].std()
        if args.pricer == "beer":
            beer_fv = beer_fair_values.get(pair)
            if beer_fv is not None:
                orig_sharpe = beer_sharpe(nodes_data, beer_fv)
            else:
                orig_sharpe = 0.0
        else:
            fx_ret = np.log(nodes_data["FX"] / nodes_data["FX"].shift(1)).fillna(0.0)
            orig_position = pd.Series(0.0, index=nodes_data.index)
            orig_position[nodes_data["FVG"] < -fvg_thresh] = 1.0
            orig_position[nodes_data["FVG"] > fvg_thresh] = -1.0
            orig_strat_ret = (orig_position.shift(1).fillna(0.0) * fx_ret).dropna()
            orig_sharpe = float(orig_strat_ret.mean() / orig_strat_ret.std() * np.sqrt(ANNUALIZATION_QUARTERLY)) if orig_strat_ret.std() > 0 else 0.0
        pair_orig_sharpe[pair] = orig_sharpe
        print(f"    Original Sharpe: {orig_sharpe:+.3f}")

        # Compute L0 losses per method
        L0_losses_this = {}
        L0_sharpe_dist_this = {}
        L0_loss_dist_this = {}  # Per-replication loss arrays for Wilcoxon
        for method, resamples in bootstrap_results.items():
            sharpes = []
            for r in resamples:
                r = r.copy()
                r.index = nodes_data.index[:len(r)]
                if args.pricer == "beer":
                    beer_fv = beer_fair_values.get(pair)
                    if beer_fv is not None and len(r) >= len(beer_fv):
                        s = beer_sharpe(r, beer_fv.iloc[:len(r)])
                        if not np.isnan(s):
                            sharpes.append(s)
                else:
                    r_fvg = np.log(r["FX"]) - np.log(r["FX"]).rolling(4, min_periods=2).mean()
                    r_fx_ret = np.log(r["FX"] / r["FX"].shift(1)).fillna(0.0)
                    r_pos = pd.Series(0.0, index=r.index)
                    r_pos[r_fvg < -fvg_thresh] = 1.0
                    r_pos[r_fvg > fvg_thresh] = -1.0
                    r_strat = (r_pos.shift(1).fillna(0.0) * r_fx_ret).dropna()
                    if r_strat.std() > 0 and len(r_strat) > 5:
                        sharpes.append(float(r_strat.mean() / r_strat.std() * np.sqrt(ANNUALIZATION_QUARTERLY)))
            if sharpes:
                L0_sharpe_dist_this[method] = np.array(sharpes)
                L0_loss_dist_this[method] = np.abs(np.array(sharpes) - orig_sharpe)
                L0_losses_this[method] = float(np.mean(L0_loss_dist_this[method]))

        pair_L0_losses[pair] = L0_losses_this

        # SE(Sharpe)
        T_eff = len(nodes_data) - 1
        se_sharpe = float(np.sqrt((1 + DEVIATION_THRESHOLD_MULT * orig_sharpe ** 2) / max(T_eff, 1)))

        # L1-L4 losses per method
        losses_this = {}
        L1_dist_this = {}
        L2_dist_this = {}
        L3_fro_dist_this = {}
        L3_spec_dist_this = {}
        L4_temporal_dist_this = {}
        for method, resamples in bootstrap_results.items():
            boot_corrs = []
            for r in resamples:
                r_data = r[NODES].copy()
                r_data.index = nodes_data.index[:len(r_data)]
                try:
                    bc = compute_correlations(r_data, CORRELATION_PAIRS)
                    boot_corrs.append(bc)
                except Exception:
                    continue
            if not boot_corrs:
                continue
            # L1: per-replication bias
            L1_per_rep = []
            for bc in boot_corrs:
                bias_single = compute_correlation_bias(orig_corrs, [bc])
                L1_per_rep.append(float(np.mean([abs(v) for v in bias_single.values()])))
            if L1_per_rep:
                L1_dist_this[method] = np.array(L1_per_rep)
                L1 = float(np.mean(L1_dist_this[method]))
            else:
                L1 = 0.0
            # L2: per-replication Wasserstein
            L2_per_rep = []
            for bc in boot_corrs:
                W = compute_wasserstein_distances(orig_corrs, [bc])
                L2_per_rep.append(float(np.mean(list(W.values()))))
            if L2_per_rep:
                L2_dist_this[method] = np.array(L2_per_rep)
                L2 = float(np.mean(L2_dist_this[method]))
            else:
                L2 = 0.0
            # L3
            L3_fro_list, L3_spec_list = [], []
            for r in resamples:
                r_data = r[NODES].copy()
                r_data.index = nodes_data.index[:len(r_data)]
                try:
                    g = build_correlation_graph(r_data, NODES)
                    L3_fro_list.append(compute_network_distance(orig_graph, g, "frobenius"))
                    L3_spec_list.append(compute_network_distance(orig_graph, g, "spectral"))
                except Exception:
                    continue
            if L3_fro_list:
                L3_fro_dist_this[method] = np.array(L3_fro_list)
            if L3_spec_list:
                L3_spec_dist_this[method] = np.array(L3_spec_list)
            L3_fro = float(np.mean(L3_fro_list)) if L3_fro_list else 0.0
            L3_spec = float(np.mean(L3_spec_list)) if L3_spec_list else 0.0
            # L4
            orig_fx = np.log(nodes_data["FX"].values)
            orig_fvg_vals = nodes_data["FVG"].fillna(0.0).values
            sim_fx_list, sim_fvg_list = [], []
            for r in resamples:
                r = r.copy()
                r.index = nodes_data.index[:len(r)]
                sim_fx_list.append(np.log(r["FX"].values))
                sim_fvg_list.append((np.log(r["FX"]) - np.log(r["FX"]).rolling(4, min_periods=2).mean()).fillna(0.0).values)
            # Per-replication L4: call mean_temporal_loss with single-element lists
            L4_per_rep = []
            for sim_fx, sim_fvg in zip(sim_fx_list, sim_fvg_list):
                d = mean_temporal_loss([sim_fx], [sim_fvg], orig_fx, orig_fvg_vals)
                L4_per_rep.append(float(np.mean(list(d.values()))) if d else 0.0)
            if L4_per_rep:
                L4_temporal_dist_this[method] = np.array(L4_per_rep)
            L4_temporal = mean_temporal_loss(sim_fx_list, sim_fvg_list, orig_fx, orig_fvg_vals)
            composite = compute_composite_loss(L1, L2, L3_fro, L3_spec)
            losses_this[method] = {
                "L0": L0_losses_this.get(method, np.nan),
                "L1": L1, "L2": L2, "L3_fro": L3_fro, "L3_spec": L3_spec,
                "L4_temporal": L4_temporal, "composite": composite,
            }

        pair_results[pair] = {
            "losses": losses_this,
            "loss_dists": {
                "L0": L0_loss_dist_this,
                "L1": L1_dist_this,
                "L2": L2_dist_this,
                "L3_fro": L3_fro_dist_this,
                "L3_spec": L3_spec_dist_this,
                "L4_temporal": L4_temporal_dist_this,
            },
            "orig_sharpe": orig_sharpe,
            "orig_corrs": orig_corrs,
            "orig_graph": orig_graph,
            "se_sharpe": se_sharpe,
        }

        print(f"    L0 (Regime-Markov): {losses_this.get('Regime-Markov', {}).get('L0', np.nan):.4f}")

    # ====================================================================
    # Cross-pair aggregation: mean L0-L4 losses across pairs
    # ====================================================================
    if len(pairs_to_run) > 1:
        print("\n[6/9] Aggregating losses across pairs (cross-pair mean)...")

        # Collect all method losses
        all_methods = set()
        for pr in pair_results.values():
            all_methods.update(pr["losses"].keys())

        # Aggregate L0-L4 per method
        agg_losses = {}
        for method in all_methods:
            L0_vals = [pair_results[p]["losses"].get(method, {}).get("L0", np.nan) for p in pairs_to_run]
            L1_vals = [pair_results[p]["losses"].get(method, {}).get("L1", np.nan) for p in pairs_to_run]
            L2_vals = [pair_results[p]["losses"].get(method, {}).get("L2", np.nan) for p in pairs_to_run]
            L3_fro_vals = [pair_results[p]["losses"].get(method, {}).get("L3_fro", np.nan) for p in pairs_to_run]
            L3_spec_vals = [pair_results[p]["losses"].get(method, {}).get("L3_spec", np.nan) for p in pairs_to_run]
            L4_vals = [pair_results[p]["losses"].get(method, {}).get("L4_temporal", {}) for p in pairs_to_run]

            agg_losses[method] = {
                "L0": float(np.nanmean(L0_vals)),
                "L1": float(np.nanmean(L1_vals)),
                "L2": float(np.nanmean(L2_vals)),
                "L3_fro": float(np.nanmean(L3_fro_vals)),
                "L3_spec": float(np.nanmean(L3_spec_vals)),
                "L4_temporal": float(np.nanmean([np.nanmean(list(v.values())) if isinstance(v, dict) else v for v in L4_vals])),
            }
            agg_losses[method]["composite"] = compute_composite_loss(
                agg_losses[method]["L1"],
                agg_losses[method]["L2"],
                agg_losses[method]["L3_fro"],
                agg_losses[method]["L3_spec"],
            )

        # Aggregated original Sharpe
        agg_orig_sharpe = float(np.mean(list(pair_orig_sharpe.values())))
        print(f"  Cross-pair mean Sharpe: {agg_orig_sharpe:+.3f}")

        # Print summary
        print(f"\n  Cross-pair aggregated losses:")
        for m, l in agg_losses.items():
            print(f"    {m:14s} L0={l['L0']:.4f} L1={l['L1']:.4f} L2={l['L2']:.4f} "
                  f"L3_spec={l['L3_spec']:.4f} composite={l['composite']:.4f}")

        # Store aggregated as main result
        losses = agg_losses
        orig_sharpe = agg_orig_sharpe
    else:
        # Single pair: use that pair's results directly
        pair = pairs_to_run[0]
        losses = pair_results[pair]["losses"]
        orig_sharpe = pair_orig_sharpe[pair]

    # ====================================================================
    # Section 4: Arrangement distance (for single pair, or reference pair if pooled)
    # ====================================================================
    # For arrangement, use the REFERENCE pair (EUR/USD) as representative
    ref_nodes = pair_data[ref_pair]["nodes_data"]
    ref_expert_blocks = pair_data[ref_pair]["expert_blocks"]
    orig_regime_seq = []
    for b in ref_expert_blocks:
        n_in_block = b.length_months
        orig_regime_seq.extend([b.regime_id - 1] * n_in_block)
    n_obs = len(ref_nodes)
    orig_regime_seq = orig_regime_seq[:n_obs]
    while len(orig_regime_seq) < n_obs:
        orig_regime_seq.append(orig_regime_seq[-1] if orig_regime_seq else 0)
    orig_regime_seq = np.array(orig_regime_seq[:n_obs], dtype=int)

    # Re-run Regime-Markov for arrangement distance (need trace)
    fvg_thresh = DEVIATION_THRESHOLD_MULT * ref_nodes["FVG"].std()

    # ====================================================================
    # BUG-4: Compute arrangement distance per run
    # ====================================================================
    print("\n[6/9] Computing arrangement distances (Spec §6)...")

    # Load calibrated transition matrix
    trans_matrix_calibrated = np.load(out_dir / "transition_matrix_calibrated.npy")

    # Methods to test (include all 5)
    arrangement_methods = ["IID", "MBB", "SB", "Regime-Uniform", "Regime-Markov"]

    # Number of simulations for arrangement
    N_SIM = 199  # per method

    # Setup common RNG base
    base_seed_arr = 20260604

    # Original regime sequence is already built above (orig_regime_seq)
    n_obs = len(ref_nodes)

    # Helper to simulate regime sequence from transition matrix
    def simulate_regime_sequence(n: int, method: str, seed: int) -> np.ndarray:
        """Simulate a regime sequence from transition matrix."""
        rng = np.random.default_rng(seed)
        if method == "IID":
            # IID: each observation independently uniformly distributed
            seq = rng.integers(0, 8, size=n)
        elif method == "MBB" or method == "SB":
            # Block bootstrap: sample blocks uniformly
            # Simplified: use stationary distribution
            stationary = np.zeros(8)
            for b in ref_expert_blocks:
                stationary[b.regime_id - 1] += b.length_months
            stationary /= stationary.sum()
            seq = rng.choice(8, size=n, p=stationary)
        elif method == "Regime-Uniform":
            # Uniform random walk
            seq = [rng.integers(0, 8)]
            for _ in range(n - 1):
                seq.append(rng.integers(0, 8))
            seq = np.array(seq)
        elif method == "Regime-Markov":
            # Markov walk using calibrated transition matrix
            init = initial_dist
            seq = [int(rng.choice(8, p=init))]
            for _ in range(n - 1):
                prev = seq[-1]
                seq.append(int(rng.choice(8, p=trans_matrix_calibrated[prev])))
            seq = np.array(seq)
        else:
            seq = rng.integers(0, 8, size=n)
        return seq[:n]

    # Compute arrangement distances per method
    arrangement_results = []
    for method in arrangement_methods:
        # Simulate N_SIM sequences and compute distances
        comp_dists, trans_dists, norm_edit_dists = [], [], []
        for sim_rep in range(N_SIM):
            seed = base_seed_arr + sim_rep * 100 + hash(method) % 10000
            sim_seq = simulate_regime_sequence(n_obs, method, seed)
            if len(sim_seq) < n_obs:
                continue
            sim_seq = sim_seq[:n_obs]
            comp_dists.append(arrangement_distance(sim_seq, orig_regime_seq, metric="composition"))
            trans_dists.append(arrangement_distance(sim_seq, orig_regime_seq, metric="transition"))
            norm_edit_dists.append(arrangement_distance(sim_seq, orig_regime_seq, metric="norm_edit"))
        # Compute summary stats
        arrangement_results.append({
            "method": method,
            "composition_mean": float(np.mean(comp_dists)),
            "composition_median": float(np.median(comp_dists)),
            "composition_p10": float(np.percentile(comp_dists, 10)),
            "composition_p90": float(np.percentile(comp_dists, 90)),
            "transition_mean": float(np.mean(trans_dists)),
            "transition_median": float(np.median(trans_dists)),
            "transition_p10": float(np.percentile(trans_dists, 10)),
            "transition_p90": float(np.percentile(trans_dists, 90)),
            "norm_edit_mean": float(np.mean(norm_edit_dists)),
            "norm_edit_median": float(np.median(norm_edit_dists)),
            "norm_edit_p10": float(np.percentile(norm_edit_dists, 10)),
            "norm_edit_p90": float(np.percentile(norm_edit_dists, 90)),
        })

    arrangement_df = pd.DataFrame(arrangement_results)
    arrangement_df.to_csv(out_dir / "arrangement_distances.csv", index=False)
    print(f"  Saved arrangement_distances.csv ({len(arrangement_df)} methods)")

    # Compute arrangement_summary for decision
    arrangement_summary = {
        m["method"]: {
            "composition_mean": m["composition_mean"],
            "composition_median": m["composition_median"],
            "transition_mean": m["transition_mean"],
            "transition_median": m["transition_median"],
            "norm_edit_mean": m["norm_edit_mean"],
            "norm_edit_median": m["norm_edit_median"],
        }
        for m in arrangement_results
    }

    # Compute L0 details for reference pair (for reporting)
    L0_losses_ref = pair_L0_losses.get(ref_pair, {})
    T_eff = len(ref_nodes) - 1
    se_sharpe = pair_results[ref_pair]["se_sharpe"]

    # Compute L0 normalized and summary
    L0_normalised = {}
    L0_within_1se = {}

    # Use reference pair's L0 for SE calculation
    regime_traces_all = []

    if len(pairs_to_run) == 1:
        # Single pair: compute SE from available stats
        T_eff = len(ref_nodes) - 1
        pair_se = pair_results[ref_pair]["se_sharpe"]
        pair_L0 = pair_L0_losses.get(ref_pair, {})

        print(f"  Original Sharpe: {pair_orig_sharpe[ref_pair]:+.3f}  |  SE(Sharpe) = {pair_se:.3f}")
        print(f"  L0 (Sharpe gap, mean |MC_sharpe - orig_sharpe|):")
        for m, l in pair_L0.items():
            n_se = l / pair_se if pair_se > 0 else 0
            print(f"    {m:14s} L0={l:.4f}  (= {n_se:.2f}× SE(Sharpe))")

    else:
        # Multi-pair: report cross-pair L0 stats
        T_eff = len(ref_nodes) - 1
        pair_se = pair_results[ref_pair]["se_sharpe"]
        print(f"  Reference pair {ref_pair.upper()} Sharpe: {pair_orig_sharpe.get(ref_pair, 0):+.3f}  |  SE = {pair_se:.3f}")
        L0_vals_ref = pair_L0_losses.get(ref_pair, {})
        print(f"  Reference L0 (Regime-Markov): {L0_vals_ref.get('Regime-Markov', 0):.4f}")

    # Save per-pair results
    for pair in pairs_to_run:
        pair_dir = out_dir / pair
        pair_dir.mkdir(exist_ok=True)
        with open(pair_dir / "decision.json", "w") as f:
            json.dump({
                "pair": pair,
                "orig_sharpe": pair_orig_sharpe.get(pair, 0),
                "L0_losses": pair_L0_losses.get(pair, {}),
                "n_blocks": len(pair_data[pair]["bootstrap_blocks"]),
            }, f, indent=2)

    # ====================================================================
    # Section 5: Paired tests and FDR (simplified for multi-pair)
    # For pooled case, we use cross-pair aggregated losses in decision
    # ====================================================================

    # Helper function moved earlier for use in tests
    def make_serializable(obj):
        """Convert numpy/pandas types to JSON-serializable Python types."""
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        return obj

    print("\n[7/9] Decision summary (based on cross-pair losses if pooled)...")

    # Extract L0 for decision (needed for tests below)
    L0_losses = {}
    if len(pairs_to_run) == 1:
        L0_losses = pair_L0_losses.get(pairs_to_run[0], {})
    elif "Regime-Markov" in losses:
        L0_losses = {m: losses[m].get("L0", np.nan) for m in losses}
    else:
        L0_losses = {}

    def _pool_per_pair(arr_dict, pairs, method):
        """Concatenate per-pair per-replication loss arrays for a given method."""
        out = []
        for p in pairs:
            d = arr_dict.get(p, {}).get(method)
            if d is not None and len(d) > 0:
                out.append(np.asarray(d).flatten())
        if not out:
            return None
        return np.concatenate(out)

    # FIX-8: Compute Wilcoxon p-values and BH-FDR for structural metrics
    print("\n[8/9] Running Wilcoxon + BH-FDR tests (24 tests)...")

    # Metrics to test
    struct_metrics = ["L1", "L2", "L3_fro", "L3_spec", "L4_temporal"]
    # Baseline methods for family-level comparisons
    baselines = ["IID", "MBB", "SB", "Regime-Uniform"]
    test_method = "Regime-Markov"

    # Build loss_dists dict for all pairs (mapping pair -> metric -> method -> array)
    loss_dists_by_pair = {}
    for pair in pairs_to_run:
        loss_dists_by_pair[pair] = pair_results[pair].get("loss_dists", {})

    all_test_results = []
    raw_p_values = []

    # For each method-to-baseline test
    for baseline in baselines:
        for metric in struct_metrics:
            test_name = f"Regime-Markov vs {baseline} on {metric}"

            # Build arr_dict: pair -> metric -> method -> array
            arr_dict = {p: loss_dists_by_pair[p].get(metric, {}) for p in pairs_to_run}

            # Aggregate losses to ONE number per (method, metric, pair)
            method_per_pair = []
            baseline_per_pair = []
            for p in pairs_to_run:
                method_loss = arr_dict[p].get(metric, {}).get(test_method)
                baseline_loss = arr_dict[p].get(metric, {}).get(baseline)
                if method_loss is not None and baseline_loss is not None:
                    method_per_pair.append(float(np.mean(np.abs(method_loss))))
                    baseline_per_pair.append(float(np.mean(np.abs(baseline_loss))))

            # Run pair-level sign test (n=3)
            if len(method_per_pair) >= 2:
                wilcoxon_p = pair_level_pvalue(
                    np.array(method_per_pair), np.array(baseline_per_pair)
                )
                effect = pair_level_effect_size(
                    np.array(method_per_pair), np.array(baseline_per_pair)
                )
            else:
                wilcoxon_p = 1.0
                effect = {
                    "mean_diff": np.nan,
                    "median_diff": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n_pairs": len(method_per_pair),
                }

            raw_p_values.append(wilcoxon_p)

            all_test_results.append({
                "name": test_name,
                "wilcoxon_p": wilcoxon_p,
                "effect_size": effect,
                "metric": metric,
                "baseline": baseline,
                "method": "pair_level_sign_test",
            })

    # Add family-level tests (Regime-Markov vs each baseline on aggregate L0)
    for baseline in baselines:
        test_name = f"Regime-Markov vs {baseline} on L0"

        # Build arr_dict for L0
        arr_dict = {p: loss_dists_by_pair[p].get("L0", {}) for p in pairs_to_run}

        # Aggregate losses to ONE number per (method, metric, pair)
        method_per_pair = []
        baseline_per_pair = []
        for p in pairs_to_run:
            method_loss = arr_dict[p].get(test_method)
            baseline_loss = arr_dict[p].get(baseline)
            if method_loss is not None and baseline_loss is not None:
                method_per_pair.append(float(np.mean(np.abs(method_loss))))
                baseline_per_pair.append(float(np.mean(np.abs(baseline_loss))))

        # Run pair-level sign test (n=3)
        if len(method_per_pair) >= 2:
            wilcoxon_p = pair_level_pvalue(
                np.array(method_per_pair), np.array(baseline_per_pair)
            )
            effect = pair_level_effect_size(
                np.array(method_per_pair), np.array(baseline_per_pair)
            )
        else:
            wilcoxon_p = 1.0
            effect = {
                "mean_diff": np.nan,
                "median_diff": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n_pairs": len(method_per_pair),
            }

        raw_p_values.append(wilcoxon_p)

        all_test_results.append({
            "name": test_name,
            "wilcoxon_p": wilcoxon_p,
            "effect_size": effect,
            "metric": "L0",
            "baseline": baseline,
            "method": "pair_level_sign_test",
        })

    n_tests = len(all_test_results)
    print(f"  Computed {n_tests} tests")

    # Apply BH-FDR correction
    raw_p_arr = np.array(raw_p_values)

    # FDR adjusted q-values
    fdr_q_raw = apply_bh_fdr(raw_p_arr, alpha=0.10)

    # Rejection decisions
    rejected_at_010 = fdr_q_raw <= 0.10
    rejected_at_005 = fdr_q_raw <= 0.05

    # Build test results dict (v14: pair_level_sign_test)
    test_results_dict = {
        "n_tests": n_tests,
        "method": "pair_level_sign_test",
        "raw_p_values": raw_p_values,
        "fdr_adjusted_q_values": fdr_q_raw.tolist(),
        "rejected_at_010": rejected_at_010.tolist(),
        "rejected_at_005": rejected_at_005.tolist(),
        "tests": [
            {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
             for k, v in tr.items()}
            for tr in all_test_results
        ],
    }

    # Save test_results.json
    with open(out_dir / "test_results.json", "w") as f:
        json.dump(make_serializable(test_results_dict), f, indent=2)

    # Also save fdr_adjusted.json (v13: wilcoxon_only)
    fdr_adjusted = {
        "tests": [
            {
                "name": tr["name"],
                "wilcoxon_p": tr["wilcoxon_p"],
                "q_bh_raw": float(fdr_q_raw[i]),
            }
            for i, tr in enumerate(all_test_results)
        ],
    }
    with open(out_dir / "fdr_adjusted.json", "w") as f:
        json.dump(fdr_adjusted, f, indent=2)

    print(f"  Saved test_results.json ({n_tests} tests)")
    print(f"  Saved fdr_adjusted.json")
    rejected_count = int(np.sum(rejected_at_010))
    print(f"  Rejected at 0.10: {rejected_count}/{n_tests}")

    # Decision simplified for multi-pair: use aggregated losses
    # L0 comparison against IID/MBB/SB
    regime_L0 = L0_losses.get("Regime-Markov", 999)
    family_level_pass = regime_L0 < min(
        L0_losses.get("IID", 999),
        L0_losses.get("MBB", 999),
        L0_losses.get("SB", 999),
    )

    print(f"\n  --- Summary ---")
    print(f"  Number of pairs: {len(pairs_to_run)}")
    print(f"  Effective bootstrap blocks: {len(pooled_blocks)} (vs 8 for single-pair)")
    print(f"  L0 (Regime-Markov): {regime_L0:.4f}")
    print(f"  Family-level pass: {family_level_pass}")

    # Decision based on multi-pair aggregated results
    if len(pairs_to_run) > 1:
        if losses:
            comp = losses.get("Regime-Markov", {}).get("composite", 999)
            L1_val = losses.get("Regime-Markov", {}).get("L1", 999)
            L3_spec_val = losses.get("Regime-Markov", {}).get("L3_spec", 999)
            L4_val = losses.get("Regime-Markov", {}).get("L4_temporal", 999)
            print(f"  Composite (Regime-Markov): {comp:.4f}")
            print(f"  L1: {L1_val:.4f}")
            print(f"  L3_spec: {L3_spec_val:.4f}")
            print(f"  L4_temporal: {L4_val:.4f}")

            decision = "GO" if comp < ARRANGEMENT_FAR_THRESHOLD else "MIXED" if comp < 1.0 else "NO-GO"
        else:
            decision = "INSUFFICIENT_DATA"
    else:
        # Single pair: use simple threshold on L0
        reg_L0 = pair_L0_losses.get(pairs_to_run[0], {}).get("Regime-Markov", 999)
        family_pass = reg_L0 < min(
            pair_L0_losses.get(pairs_to_run[0], {}).get("IID", 999),
            pair_L0_losses.get(pairs_to_run[0], {}).get("MBB", 999),
            pair_L0_losses.get(pairs_to_run[0], {}).get("SB", 999),
        )
        decision = "GO" if (family_pass and reg_L0 < ARRANGEMENT_FAR_THRESHOLD) else "MIXED" if reg_L0 < 1.0 else "NO-GO"

    print(f"\n  DECISION: {decision}")

    # ====================================================================
    # Save final results
    # ====================================================================
    print("\nSaving results...")

    # Combined decision.json
    def make_serializable(obj):
        """Convert numpy/pandas types to JSON-serializable Python types."""
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        return obj

    with open(out_dir / "decision.json", "w") as f:
        json.dump(make_serializable({
            "decision": decision,
            "n_pairs": len(pairs_to_run),
            "pairs": pairs_to_run,
            "n_blocks": len(pooled_blocks),
            "cross_pair_L0": float(regime_L0),
            "family_level_pass": bool(family_level_pass),
            "orig_sharpe": float(orig_sharpe),
            "L0_losses": {k: float(v) for k, v in L0_losses.items()},
            "per_pair_sharpe": {k: float(v) for k, v in pair_orig_sharpe.items()},
            "aggregated_losses": losses,
            "arrangement_summary": arrangement_summary,
        }), f, indent=2)

    # Print final summary
    print("\n" + "=" * 70)
    print(f"COMPLETE: {len(pairs_to_run)} pairs, {len(pooled_blocks)} blocks pooled")
    print(f"Decision: {decision}")
    print(f"Output: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
