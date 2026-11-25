"""Paired comparison tests for the study protocol, §9b.

Pooled (treat every stand-up / window as independent):  mcnemar_exact, cochran_q, delong
Clustered by participant:  mcnemar_clustered (Durkalski et al. 2003), delong_clustered (Obuchowski 1997)
Exact over participants:   sign_flip (2^n sign patterns on the mean difference)

  python comparison/test_tools.py       self-checks, with the false-positive rate on clustered null data
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, chi2, norm, rankdata


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """a, b: boolean outcomes on the same items. only_a = a right and b wrong."""
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_a": only_a, "only_b": only_b, "p": float(p)}


def mcnemar_clustered(a: np.ndarray, b: np.ndarray, cluster: np.ndarray) -> dict:
    """Durkalski et al. (2003): sum over clusters of (b_k - c_k) / n_k, chi-square with 1 df."""
    d = np.array([np.mean(a[cluster == c].astype(float) - b[cluster == c].astype(float)) for c in np.unique(cluster)])
    denominator = float(np.sum(d ** 2))
    statistic = float(np.sum(d)) ** 2 / denominator if denominator > 0 else 0.0
    return {"chi2": statistic, "p": float(chi2.sf(statistic, 1)) if denominator > 0 else 1.0}


def sign_flip(d: np.ndarray) -> float:
    """Exact two-sided sign-flip p-value for mean(d) = 0 over participants."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    signs = 1 - 2 * ((np.arange(2 ** len(d))[:, None] >> np.arange(len(d))) & 1)
    return float(np.mean(np.abs(signs @ d) >= abs(d.sum()) - 1e-12))


def cochran_q(x: np.ndarray) -> dict:
    """x: (items, models) booleans. Cochran's Q, chi-square with k - 1 df."""
    x = x.astype(float)
    k, columns, rows = x.shape[1], x.sum(axis=0), x.sum(axis=1)
    denominator = k * rows.sum() - np.sum(rows ** 2)
    q = (k - 1) * (k * np.sum(columns ** 2) - columns.sum() ** 2) / denominator if denominator > 0 else 0.0
    return {"q": float(q), "df": k - 1, "p": float(chi2.sf(q, k - 1)) if denominator > 0 else 1.0}


def components(score: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """AUC and DeLong's structural components: V10 per positive, V01 per negative."""
    positive, negative = score[y], score[~y]
    m, n = len(positive), len(negative)
    ranks = rankdata(np.r_[positive, negative])
    v10 = (ranks[:m] - rankdata(positive)) / n
    v01 = 1 - (ranks[m:] - rankdata(negative)) / m
    return float(v10.mean()), v10, v01


def _result(theta: np.ndarray, covariance: np.ndarray) -> dict:
    difference = float(theta[0] - theta[1])
    variance = float(covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1])
    if variance <= 0:
        return {"auc_a": theta[0], "auc_b": theta[1], "difference": difference, "se": 0.0, "lo": difference, "hi": difference,
                "p": 1.0 if difference == 0 else np.nan}
    se = np.sqrt(variance)
    return {"auc_a": theta[0], "auc_b": theta[1], "difference": difference, "se": se, "lo": difference - 1.959964 * se,
            "hi": difference + 1.959964 * se, "p": float(2 * norm.sf(abs(difference) / se))}


def delong(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> dict:
    parts = [components(s, y) for s in (a, b)]
    v10, v01 = np.column_stack([p[1] for p in parts]), np.column_stack([p[2] for p in parts])
    return _result(np.array([p[0] for p in parts]), np.cov(v10, rowvar=False) / len(v10) + np.cov(v01, rowvar=False) / len(v01))


def delong_clustered(a: np.ndarray, b: np.ndarray, y: np.ndarray, cluster: np.ndarray) -> dict:
    """Obuchowski (1997): components summed within clusters, so items of one participant may be correlated."""
    parts = [components(s, y) for s in (a, b)]
    theta = np.array([p[0] for p in parts])
    v10 = np.column_stack([p[1] for p in parts]) - theta
    v01 = np.column_stack([p[2] for p in parts]) - theta
    clusters = np.unique(cluster)
    s10 = np.array([v10[cluster[y] == c].sum(axis=0) for c in clusters])
    s01 = np.array([v01[cluster[~y] == c].sum(axis=0) for c in clusters])
    k, m, n = len(clusters), len(v10), len(v01)
    i10, i01 = len(np.unique(cluster[y])), len(np.unique(cluster[~y]))
    cross = k / (k - 1) * s10.T @ s01 / (m * n)
    covariance = (i10 / (i10 - 1) * s10.T @ s10 / m ** 2 + i01 / (i01 - 1) * s01.T @ s01 / n ** 2 + cross + cross.T)
    return _result(theta, covariance)


def self_check() -> None:
    from sklearn.metrics import roc_auc_score
    from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
    rng = np.random.default_rng(0)
    y = rng.random(800) < 0.2
    a, b = np.round(rng.normal(size=800) + y, 1), np.round(rng.normal(size=800) + 0.8 * y, 1)      # with ties
    r = delong(a, b, y)
    assert np.isclose(r["auc_a"], roc_auc_score(y, a)) and np.isclose(r["auc_b"], roc_auc_score(y, b))
    # one item per cluster: the clustered version must agree with DeLong
    c = delong_clustered(a, b, y, np.arange(800))
    assert np.isclose(r["se"], c["se"], rtol=0.01), (r["se"], c["se"])
    boot = []
    for _ in range(1000):
        i = rng.integers(0, 800, 800)
        boot.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    assert np.isclose(r["se"], np.std(boot), rtol=0.1), (r["se"], np.std(boot))
    u, v = rng.random(300) < 0.9, rng.random(300) < 0.85
    table = [[np.sum(u & v), np.sum(u & ~v)], [np.sum(~u & v), np.sum(~u & ~v)]]
    assert np.isclose(mcnemar_exact(u, v)["p"], mcnemar(table, exact=True).pvalue)
    x = rng.random((300, 4)) < 0.9
    assert np.isclose(cochran_q(x)["q"], cochrans_q(x).statistic)
    print("self-checks passed (AUC vs scikit-learn, DeLong vs bootstrap and one-item clusters, McNemar and Cochran vs statsmodels)")

    # False-positive rate at 0.05 when two detectors are equally good on average but differ by participant.
    ids, rejected = np.repeat(np.arange(11), 47), np.zeros(6)
    trials = 1000
    for _ in range(trials):
        u = rng.random(len(ids)) < np.clip(0.94 + rng.normal(0, 0.05, 11), 0, 1)[ids]
        v = rng.random(len(ids)) < np.clip(0.94 + rng.normal(0, 0.05, 11), 0, 1)[ids]
        shares = [u[ids == k].mean() - v[ids == k].mean() for k in range(11)]
        y = rng.random(len(ids)) < 0.3
        a = rng.normal(size=len(ids)) + y * (1.5 + rng.normal(0, 0.4, 11)[ids])
        b = rng.normal(size=len(ids)) + y * (1.5 + rng.normal(0, 0.4, 11)[ids])
        rejected += [mcnemar_exact(u, v)["p"] < .05, mcnemar_clustered(u, v, ids)["p"] < .05, sign_flip(shares) < .05,
                     delong(a, b, y)["p"] < .05, delong_clustered(a, b, y, ids)["p"] < .05, 0]
    for name, value in zip(("McNemar pooled", "McNemar clustered", "sign-flip", "DeLong pooled", "DeLong clustered"), rejected):
        print(f"  false-positive rate, no true difference, 11 participants: {name} {value / trials:.3f} (nominal 0.05)")


if __name__ == "__main__":
    self_check()
