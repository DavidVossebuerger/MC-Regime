"""Regime-Preserving Bootstrap sampler.

Per Spec §3.4: paths are **generated from scratch** by a Markov chain over
regime labels, not by uniform-with-replacement. At each step, we
- sample the next regime_id from the transition matrix (Markov step)
- sample a real block of that regime_id (with replacement)
- concatenate until we have ≥ n observations, then truncate.

Two modes:
- "uniform" (baseline control): pick regime uniformly at random per step.
- "markov" (proposed method): walk the chain via transition matrix.
"""

from typing import List, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from mc_regime.bootstrap.blocks import Block
from mc_regime.bootstrap.samplers.base import Sampler


class RegimeSampler:
    """Regime-Preserving Bootstrap.

    Resamples entire regime/event blocks to preserve regime structure
    in the bootstrapped samples. In mode="markov", block selection follows
    a Markov chain over regime labels (Spec §3.4). In mode="uniform",
    regime selection is uniform at each step (control baseline).

    Args:
        mode: Sampling mode - "uniform" or "markov".
        transition_matrix: Markov transition probability matrix (k x k).
            Required if mode="markov".
        initial_distribution: Initial state distribution for Markov chain.
            Defaults to uniform if None.
    """

    def __init__(
        self,
        mode: Literal["uniform", "markov"] = "uniform",
        transition_matrix: Optional[np.ndarray] = None,
        initial_distribution: Optional[np.ndarray] = None,
    ):
        if mode not in ("uniform", "markov"):
            raise ValueError("mode must be 'uniform' or 'markov'")
        if mode == "markov" and transition_matrix is None:
            raise ValueError("transition_matrix required for mode='markov'")

        self.mode = mode
        self.transition_matrix = transition_matrix
        self.initial_distribution = initial_distribution

    def _walk_markov(
        self,
        n_steps: int,
        rng: np.random.Generator,
    ) -> List[int]:
        """Generate a sequence of regime indices via Markov chain.

        Starting state drawn from initial_distribution. Each subsequent
        state drawn from transition_matrix[prev_state].
        """
        k = self.transition_matrix.shape[0]
        init = self.initial_distribution if self.initial_distribution is not None \
            else np.ones(k) / k
        states = [int(rng.choice(k, p=init))]
        for _ in range(n_steps - 1):
            prev = states[-1]
            states.append(int(rng.choice(k, p=self.transition_matrix[prev])))
        return states

    def _walk_uniform(self, n_steps: int, n_regimes: int,
                      rng: np.random.Generator) -> List[int]:
        """Pick regime uniformly at random per step (control baseline)."""
        return [int(rng.integers(0, n_regimes)) for _ in range(n_steps)]

    def sample(
        self,
        data: pd.DataFrame,
        n: int,
        rng: np.random.Generator,
        blocks: Optional[List[Block]] = None,
    ) -> pd.DataFrame:
        """Resample `n` rows from data preserving regime-block structure.

        Algorithm:
        1. Build a mapping regime_id -> list of blocks (with replacement)
        2. Walk regime sequence (Markov or uniform) for enough steps to
           reach ≥ n observations
        3. For each regime_id in the sequence, randomly pick one of the
           real blocks of that regime and concatenate
        4. Truncate to exactly n rows
        """
        T = len(data)

        if blocks is None or len(blocks) == 0:
            blocks = [Block(start_idx=0, end_idx=T - 1, regime_label=0)]

        # Group blocks by regime_label
        regime_to_blocks: dict = {}
        for b in blocks:
            if b.regime_label is None:
                continue
            regime_to_blocks.setdefault(int(b.regime_label), []).append(b)

        if not regime_to_blocks:
            # No labelled blocks — fall back to single block
            regime_to_blocks = {0: [Block(start_idx=0, end_idx=T - 1, regime_label=0)]}

        n_regimes = len(regime_to_blocks)
        regime_ids = sorted(regime_to_blocks.keys())

        # Walk regime sequence: aim for more steps than needed to ensure ≥ n
        # We can't know exact block length in advance, so over-sample
        n_steps = max(n, 30)  # at least 30 steps to ensure enough material
        if self.mode == "markov":
            seq = self._walk_markov(n_steps, rng)
        else:
            seq = self._walk_uniform(n_steps, n_regimes, rng)

        # Build resampled series: for each regime_id, pick a random block
        resampled_parts = []
        regime_trace = []  # Track regime of each observation (for arrangement distance)
        collected = 0
        for rid in seq:
            # Map sampled idx to actual regime_id (in case of mapping)
            actual_rid = regime_ids[rid] if rid < len(regime_ids) else regime_ids[0]
            available = regime_to_blocks[actual_rid]
            chosen = available[int(rng.integers(0, len(available)))]
            block_data = data.iloc[chosen.start_idx:chosen.end_idx + 1]
            resampled_parts.append(block_data)
            regime_trace.extend([int(actual_rid)] * chosen.length)
            collected += chosen.length
            if collected >= n:
                break

        result = pd.concat(resampled_parts, ignore_index=True)
        result = result.iloc[:n].reset_index(drop=True)
        # Save the regime trace (per-observation regime labels) for arrangement distance
        self.last_regime_trace = regime_trace[:n]
        return result
