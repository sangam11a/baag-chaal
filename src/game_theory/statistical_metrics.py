"""
Rigorous Statistical Testing and Effect Size Engine for Game AI Research.

Implements:
1. Non-Parametric Bootstrap Confidence Intervals (Percentile Bootstrap)
2. Cohen's d Effect Size with Pooled Standard Deviation
3. Paired Wilcoxon Signed-Rank Hypothesis Test
4. Win Rate Odds Ratios and Wilson Score Intervals
"""

from __future__ import annotations
import math
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from scipy import stats


class StatisticianMetrics:
    """
    Standardized statistical evaluation suite for Q1 Springer Nature publication.
    """

    @staticmethod
    def bootstrap_confidence_interval(
        data: List[float] | np.ndarray,
        num_bootstrap: int = 1000,
        confidence: float = 0.95,
        seed: int = 42,
    ) -> Tuple[float, float, float]:
        """
        Computes non-parametric percentile bootstrap CI for the sample mean.
        Returns:
            (mean, ci_lower, ci_upper)
        """
        arr = np.asarray(data, dtype=np.float64)
        if len(arr) == 0:
            return 0.0, 0.0, 0.0
        if len(arr) == 1:
            return float(arr[0]), float(arr[0]), float(arr[0])

        rng = np.random.default_rng(seed)
        n = len(arr)
        boot_indices = rng.integers(0, n, size=(num_bootstrap, n))
        boot_means = np.mean(arr[boot_indices], axis=1)

        alpha = (1.0 - confidence) / 2.0
        ci_lower = float(np.percentile(boot_means, alpha * 100))
        ci_upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
        sample_mean = float(np.mean(arr))

        return sample_mean, ci_lower, ci_upper

    @staticmethod
    def cohens_d(group1: List[float] | np.ndarray, group2: List[float] | np.ndarray) -> float:
        """
        Computes Cohen's d effect size between two groups.
        |d| < 0.2: negligible, 0.2-0.5: small, 0.5-0.8: medium, > 0.8: large.
        """
        a = np.asarray(group1, dtype=np.float64)
        b = np.asarray(group2, dtype=np.float64)
        n1, n2 = len(a), len(b)
        if n1 < 2 or n2 < 2:
            return 0.0

        var1 = np.var(a, ddof=1)
        var2 = np.var(b, ddof=1)
        s_pooled = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if s_pooled <= 1e-12:
            return 0.0

        d = (np.mean(a) - np.mean(b)) / s_pooled
        return float(d)

    @staticmethod
    def paired_wilcoxon_test(
        scores_a: List[float] | np.ndarray,
        scores_b: List[float] | np.ndarray,
    ) -> Tuple[float, float]:
        """
        Performs Wilcoxon signed-rank test on paired sample observations.
        Returns:
            (statistic, p_value)
        """
        a = np.asarray(scores_a, dtype=np.float64)
        b = np.asarray(scores_b, dtype=np.float64)
        if len(a) != len(b):
            raise ValueError(f"Paired samples must have identical length ({len(a)} != {len(b)})")
        diff = a - b
        if np.all(diff == 0):
            return 0.0, 1.0  # Identical outcomes

        try:
            res = stats.wilcoxon(a, b, alternative="two-sided")
            return float(res.statistic), float(res.pvalue)
        except Exception:
            # Fallback if zero differences dominate
            return 0.0, 1.0

    @staticmethod
    def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Wilson score interval for binomial proportions.
        """
        if total == 0:
            return 0.0, 0.0
        z = 1.96 if confidence == 0.95 else 1.645
        p = successes / total
        denom = 1.0 + (z**2) / total
        center = (p + (z**2) / (2.0 * total)) / denom
        margin = (z * math.sqrt((p * (1.0 - p) + (z**2) / (4.0 * total)) / total)) / denom
        return max(0.0, center - margin), min(1.0, center + margin)

