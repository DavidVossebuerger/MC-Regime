"""Bootstrap engine that orchestrates multiple sampling methods.

Per Spec §3.4: paths are generated **from scratch** via Markov chain. The
Regime sampler is registered in TWO modes:
- "Regime-Uniform" : regime picked uniformly per step (control baseline)
- "Regime-Markov"  : regime walked via transition matrix (proposed method)
"""

import hashlib
import sys
import time
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from mc_regime.bootstrap.blocks import Block
from mc_regime.bootstrap.samplers import (
    IIDSampler,
    MovingBlockSampler,
    RegimeSampler,
    StationarySampler,
)


class BootMethod(str, Enum):
    """Bootstrap method names."""
    IID = "IID"
    MBB = "MBB"
    SB = "SB"
    REGIME_UNIFORM = "Regime-Uniform"
    REGIME_MARKOV = "Regime-Markov"


def _derive_seed(base_seed: int, method: str, rep: int) -> int:
    """Derive a deterministic seed from base_seed, method, and rep."""
    combined = f"{base_seed}:{method}:{rep}"
    hash_val = hashlib.sha256(combined.encode()).hexdigest()
    return int(hash_val[:8], 16)


class BootstrapEngine:
    """Orchestrates bootstrap replications across multiple sampling methods.

    Args:
        replications: Number of bootstrap replications per method.
        base_seed: Base seed for deterministic RNG.
        regime_mode: "uniform" or "markov" for the Regime sampler.
        transition_matrix: 2D array of Markov transition probabilities
            (required when regime_mode="markov").
        initial_distribution: 1D array of initial state probs
            (defaults to uniform if None).
        progress: If True, print a progress bar with ETA (default False).
    """

    METHOD_NAMES = [m.value for m in BootMethod]

    def __init__(
        self,
        replications: int = 999,
        base_seed: int = 42,
        regime_mode: str = "markov",
        transition_matrix: Optional[np.ndarray] = None,
        initial_distribution: Optional[np.ndarray] = None,
        progress: bool = True,
    ):
        self.replications = replications
        self.base_seed = base_seed
        self.regime_mode = regime_mode
        self.transition_matrix = transition_matrix
        self.initial_distribution = initial_distribution
        self.progress = progress

    def _make_sampler(self, method: str):
        if method == BootMethod.IID.value:
            return IIDSampler()
        if method == BootMethod.MBB.value:
            return MovingBlockSampler()
        if method == BootMethod.SB.value:
            return StationarySampler()
        if method.startswith("Regime"):
            mode = "markov" if "Markov" in method else "uniform"
            return RegimeSampler(
                mode=mode,
                transition_matrix=self.transition_matrix,
                initial_distribution=self.initial_distribution,
            )
        raise ValueError(f"Unknown method: {method}")

    def _print_progress(self, method: str, rep: int, t0: float, n_reps: int):
        if not self.progress:
            return
        elapsed = time.time() - t0
        done = rep + 1

        # Always show live bar on stderr (terminal)
        rate = done / max(elapsed, 1e-6)
        eta = (n_reps - done) / max(rate, 1e-6)
        bar_len = 30
        filled = int(bar_len * done / n_reps)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(
            f"\r  [{method:14s}] [{bar}] {done:4d}/{n_reps} "
            f"({100*done/n_reps:5.1f}%) ETA {eta:5.0f}s",
            end="", flush=True, file=sys.stderr,
        )

        # Write to stdout every 5% or every 100 reps (whichever is more frequent)
        pct_interval = max(1, n_reps // 20)  # 5%
        report_interval = min(100, pct_interval)  # at least every 100 if less than 5%
        if done == n_reps or done % report_interval == 0:
            print(
                f"[{method:14s}] progress {done}/{n_reps} "
                f"({100*done/n_reps:5.1f}%) ETA {eta:5.0f}s",
                flush=True,
            )

        if done == n_reps:
            print(f"  done in {elapsed:.1f}s", flush=True)

    def run_all(
        self,
        data: pd.DataFrame,
        blocks: Optional[List[Block]] = None,
    ) -> Dict[str, List[pd.DataFrame]]:
        """Run all 5 bootstrap methods, returning dict[method] -> list of resamples.

        Also populates ``self.regime_traces`` (dict[method] -> list of
        regime-label sequences, one per replication) for arrangement-distance.
        """
        n = len(data)
        results: Dict[str, List[pd.DataFrame]] = {}
        self.regime_traces: Dict[str, List[List[int]]] = {}

        for method in self.METHOD_NAMES:
            sampler = self._make_sampler(method)
            replicates: List[pd.DataFrame] = []
            traces: List[List[int]] = []
            t0 = time.time()

            for rep in range(self.replications):
                seed = _derive_seed(self.base_seed, method, rep)
                rng = np.random.default_rng(seed)

                if method.startswith("Regime") and blocks is not None:
                    resampled = sampler.sample(data, n, rng, blocks=blocks)
                    # Capture the regime trace for arrangement distance
                    trace = getattr(sampler, "last_regime_trace", None)
                    if trace is not None:
                        traces.append(list(trace))
                else:
                    resampled = sampler.sample(data, n, rng)

                replicates.append(resampled)
                self._print_progress(method, rep, t0, self.replications)

            results[method] = replicates
            self.regime_traces[method] = traces

        return results

    def run_method(
        self,
        method: str,
        data: pd.DataFrame,
        blocks: Optional[List[Block]] = None,
    ) -> List[pd.DataFrame]:
        """Run a single bootstrap method."""
        if method not in self.METHOD_NAMES:
            raise ValueError(f"Unknown method: {method}. Valid: {self.METHOD_NAMES}")

        n = len(data)
        sampler = self._make_sampler(method)
        replicates: List[pd.DataFrame] = []
        t0 = time.time()

        for rep in range(self.replications):
            seed = _derive_seed(self.base_seed, method, rep)
            rng = np.random.default_rng(seed)

            if method.startswith("Regime") and blocks is not None:
                resampled = sampler.sample(data, n, rng, blocks=blocks)
            else:
                resampled = sampler.sample(data, n, rng)

            replicates.append(resampled)
            self._print_progress(method, rep, t0, self.replications)

        return replicates