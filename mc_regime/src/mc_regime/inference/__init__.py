"""Statistical inference module for regime bootstrap evaluation.

Provides:
- Paired Wilcoxon signed-rank test for structure-preservation losses
- BCa (Bias-Corrected and Accelerated) bootstrap p-values
- Benjamini-Hochberg False Discovery Rate correction
- Cohen's d effect size
"""

from mc_regime.inference.paired_test import paired_loss_test
from mc_regime.inference.bca import bca_p_value
from mc_regime.inference.fdr import apply_bh_fdr
from mc_regime.inference.effect_size import cohens_d

__all__ = [
    "paired_loss_test",
    "bca_p_value",
    "apply_bh_fdr",
    "cohens_d",
]