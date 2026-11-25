"""Exact small-sample tools for the study protocol, §9 (n = 11 participants).

signed_rank_test  exact Wilcoxon signed-rank test by enumerating all 2^m sign patterns of the m non-zero
                  differences (ranks of |x| with average ties; zeros ranked then given sign 0, i.e. Pratt).
                  Exact for ties and zeros. Returns p, the statistic T = sum of signed ranks, and the
                  matched-pairs rank-biserial r = T / sum of ranks.
hodges_lehmann    one-sample Hodges-Lehmann estimate (median of Walsh averages) with the exact
                  distribution-free CI from the signed-rank null distribution, at any level (two-sided or
                  one-sided bound); reports the achieved level (discrete with n = 11).
bca               participant-level BCa bootstrap CI for any statistic of the participants.
holm              Holm step-down adjusted p-values.
"""

from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np
from scipy.stats import norm, rankdata


@lru_cache(maxsize=None)
def _patterns(m: int) -> np.ndarray:
    return np.array(list(itertools.product((-1.0, 1.0), repeat=m)))


def signed_rank_test(x: np.ndarray, alternative: str = "two-sided") -> dict:
    x = np.asarray(x, dtype=float)
    ranks = rankdata(np.abs(x))
    signs = np.sign(x)
    nonzero = signs != 0
    t = float(np.sum(signs * ranks))
    r = ranks[nonzero]
    if r.size == 0:
        return {"p": 1.0, "T": 0.0, "rank_biserial": 0.0, "n_nonzero": 0}
    null = _patterns(r.size) @ r
    eps = 1e-9
    if alternative == "two-sided":
        p = np.mean(np.abs(null) >= abs(t) - eps)
    elif alternative == "greater":
        p = np.mean(null >= t - eps)
    elif alternative == "less":
        p = np.mean(null <= t + eps)
    else:
        raise ValueError(alternative)
    return {"p": float(p), "T": t, "rank_biserial": t / float(r.sum()), "n_nonzero": int(r.size)}


@lru_cache(maxsize=None)
def _wplus_cdf(n: int) -> np.ndarray:
    """P(W+ <= w) for w = 0 .. n(n+1)/2 under H0 (ranks 1..n, no ties)."""
    counts = np.zeros(n * (n + 1) // 2 + 1)
    counts[0] = 1
    for rank in range(1, n + 1):
        shifted = np.zeros_like(counts)
        shifted[rank:] = counts[:-rank]                 # rank r with a + sign adds r to W+
        counts = counts + shifted
    return np.cumsum(counts) / 2 ** n


def hodges_lehmann(x: np.ndarray, level: float = 0.95, side: str = "two-sided") -> dict:
    """side: 'two-sided' -> (lo, hi); 'lower' -> (lo, inf); 'upper' -> (-inf, hi)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    i, j = np.triu_indices(n)
    walsh = np.sort((x[i] + x[j]) / 2)
    m = len(walsh)
    cdf = _wplus_cdf(n)
    tail = (1 - level) / 2 if side == "two-sided" else (1 - level)
    feasible = np.flatnonzero(cdf <= tail)               # c with P(W+ <= c) <= tail
    estimate = float(np.median(walsh))
    if feasible.size == 0:
        return {"estimate": estimate, "lo": -np.inf, "hi": np.inf, "achieved_level": 1.0}
    c = int(feasible[-1])                                 # critical value C: P(W+ <= C) <= tail
    lo, hi = float(walsh[c]), float(walsh[m - 1 - c])     # (C+1)-th smallest and (C+1)-th largest
    achieved = 1 - (2 if side == "two-sided" else 1) * float(cdf[c])
    if side == "lower":
        hi = np.inf
    elif side == "upper":
        lo = -np.inf
    return {"estimate": estimate, "lo": lo, "hi": hi, "achieved_level": achieved}


def bca(statistic, n: int, level: float = 0.95, resamples: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """BCa interval of statistic(indices) over participants 0..n-1 (indices resampled with replacement)."""
    rng = np.random.default_rng(seed)
    full = statistic(np.arange(n))
    boot = np.array([statistic(rng.integers(0, n, n)) for _ in range(resamples)])
    boot = boot[np.isfinite(boot)]
    if boot.size == 0 or np.all(boot == boot[0]):
        return full, full, full
    z0 = norm.ppf(np.clip(np.mean(boot < full) + 0.5 * np.mean(boot == full), 1e-6, 1 - 1e-6))
    jack = np.array([statistic(np.delete(np.arange(n), k)) for k in range(n)])
    d = jack.mean() - jack
    a = np.sum(d ** 3) / (6 * np.sum(d ** 2) ** 1.5) if np.sum(d ** 2) > 0 else 0.0
    z = norm.ppf([(1 - level) / 2, (1 + level) / 2])
    adjusted = norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo, hi = np.quantile(boot, adjusted)
    return float(full), float(lo), float(hi)


def holm(p: dict[str, float]) -> dict[str, float]:
    names = sorted(p, key=p.get)
    m, running, adjusted = len(names), 0.0, {}
    for k, name in enumerate(names):
        running = max(running, min(1.0, (m - k) * p[name]))
        adjusted[name] = running
    return adjusted


if __name__ == "__main__":
    # Self-checks against scipy's exact Wilcoxon (no ties / zeros) and known HL properties.
    from scipy.stats import wilcoxon
    rng = np.random.default_rng(1)
    for _ in range(200):
        x = rng.normal(0.3, 1, 11)
        mine = signed_rank_test(x)["p"]
        ref = wilcoxon(x, method="exact").pvalue
        assert abs(mine - ref) < 1e-9, (mine, ref)
        g = signed_rank_test(x, "greater")["p"]
        assert abs(g - wilcoxon(x, alternative="greater", method="exact").pvalue) < 1e-9
    x = rng.normal(1.0, 1, 11)
    hl = hodges_lehmann(x)
    assert hl["lo"] <= hl["estimate"] <= hl["hi"] and 0.95 <= hl["achieved_level"] < 0.97, hl
    # Coverage of the exact HL interval under a symmetric null: should be close to its achieved level.
    hits = sum(h["lo"] <= 0 <= h["hi"] for h in (hodges_lehmann(rng.standard_t(3, 11)) for _ in range(4000)))
    print(f"HL 95% coverage under the null: {hits / 4000:.3f} (achieved level {hl['achieved_level']:.3f})")
    print("self-checks passed; example HL", {k: round(v, 3) for k, v in hl.items()},
          "holm", holm({"a": 0.01, "b": 0.04, "c": 0.03}))
