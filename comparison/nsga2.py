"""NSGA-II over binary panel chromosomes.

Copied from the author's ADNI plasma-protein study (adni-plasma-protein-panel/src/nulisa/nsga2.py, 2026-11-24) for
the channel search of ga_channels.py (RQ7). Only change: the hypervolume trace uses the run's own maximum panel size
as reference instead of the fixed 21 of the protein study. "Section" numbers below refer to that study's protocol.

Protocol sections 16 (representation), 17 (objectives), 20 (settings),
21 (representative solution per run) and 64.4 (operators, repair, seeding).

Objectives, both minimised internally:

    f1 = -CV metric        (maximise cross-validated AUC / Uno's C)
    f2 = |S|               (minimise panel size)

Fitness is memoised per (split, chromosome) as section 12.2 requires.  The cache
key is the packed bit string, so two chromosomes with the same selected set hit
the same entry regardless of how they were produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

__all__ = ["GASettings", "GAResult", "run_nsga2", "nondominated_front",
           "hypervolume_2d"]


@dataclass(frozen=True)
class GASettings:
    population_size: int = 200
    generations: int = 150
    crossover_probability: float = 0.80
    min_panel_size: int = 1
    max_panel_size: int = 20
    mutation_rate: float | None = None   # None -> 1/p (protocol section 20)
    trace_every: int = 5
    near_optimal_tolerance: float = 0.01


@dataclass
class GAResult:
    """Everything section 55 asks to be saved for one feature-selection run."""

    selected_panel: np.ndarray                 # frozen candidate indices
    selected_metric: float
    selected_size: int
    front_panels: list[np.ndarray] = field(default_factory=list)
    front_metric: np.ndarray = field(default_factory=lambda: np.zeros(0))
    front_size: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    archive_best_by_size: dict[int, tuple[float, np.ndarray]] = field(default_factory=dict)
    trace: list[dict[str, float]] = field(default_factory=list)
    n_evaluations: int = 0
    n_cache_hits: int = 0
    seed: int = 0


# ------------------------------------------------------- Pareto machinery
def nondominated_front(objectives: np.ndarray) -> np.ndarray:
    """Boolean mask of the non-dominated rows of a minimisation objective array."""
    n = objectives.shape[0]
    le = (objectives[:, None, :] <= objectives[None, :, :]).all(axis=2)
    lt = (objectives[:, None, :] < objectives[None, :, :]).any(axis=2)
    dominates = le & lt
    return dominates.sum(axis=0) == 0


def _fast_nondominated_sort(objectives: np.ndarray) -> np.ndarray:
    """Pareto rank (0 = best front) for every row of a minimisation array."""
    n = objectives.shape[0]
    le = (objectives[:, None, :] <= objectives[None, :, :]).all(axis=2)
    lt = (objectives[:, None, :] < objectives[None, :, :]).any(axis=2)
    dominates = le & lt                       # dominates[i, j]: i dominates j
    dominated_by = dominates.sum(axis=0)
    ranks = np.full(n, -1, dtype=np.int64)
    current = dominated_by == 0
    rank = 0
    remaining = dominated_by.copy()
    while current.any():
        ranks[current] = rank
        remaining[current] = -1
        # Removing this front decrements the counts of everyone it dominated.
        remaining = remaining - dominates[current].sum(axis=0)
        current = remaining == 0
        rank += 1
    ranks[ranks < 0] = rank                   # numerical safety net
    return ranks


def _crowding_distance(objectives: np.ndarray) -> np.ndarray:
    """Standard NSGA-II crowding distance within a single front."""
    n, m = objectives.shape
    distance = np.zeros(n, dtype=np.float64)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        order = np.argsort(objectives[:, j], kind="stable")
        values = objectives[order, j]
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        span = values[-1] - values[0]
        if span <= 0:
            continue
        distance[order[1:-1]] += (values[2:] - values[:-2]) / span
    return distance


def hypervolume_2d(metric: np.ndarray, size: np.ndarray,
                   ref_metric: float = 0.5, ref_size: int = 21) -> float:
    """Hypervolume of a (metric, size) front against a fixed reference point.

    Reported per generation as the section 20 convergence diagnostic.  The
    reference point is fixed across all runs so hypervolume traces are
    comparable between splits and seeds.

    With size minimised and metric maximised, the point (s_i, m_i) dominates the
    rectangle [s_i, ref_size] x [ref_metric, m_i].  Slicing the union of those
    rectangles by metric level gives, over points sorted by ascending size (and
    therefore ascending metric along the staircase),

        HV = sum_i (m_i - m_{i-1}) * (ref_size - s_i),   m_0 = ref_metric

    which is what this computes.  Accumulating only the widest rectangle would
    make hypervolume *fall* as the front fills in with small high-quality
    panels, i.e. exactly backwards as a convergence diagnostic.
    """
    keep = (metric > ref_metric) & (size < ref_size)
    if not keep.any():
        return 0.0
    m, s = np.asarray(metric)[keep], np.asarray(size)[keep].astype(float)
    order = np.lexsort((-m, s))          # ascending size, best metric first
    m, s = m[order], s[order]

    volume = 0.0
    previous_metric = ref_metric
    for i in range(s.shape[0]):
        if m[i] <= previous_metric:      # dominated by a smaller panel
            continue
        volume += (m[i] - previous_metric) * (ref_size - s[i])
        previous_metric = m[i]
    return float(volume)


# ------------------------------------------------------------- operators
def _repair(chromosomes: np.ndarray, lo: int, hi: int, rng: np.random.Generator) -> None:
    """Random addition/removal until lo <= |S| <= hi (protocol section 64.4)."""
    sizes = chromosomes.sum(axis=1)
    for i in np.flatnonzero(sizes < lo):
        off = np.flatnonzero(~chromosomes[i])
        pick = rng.choice(off, size=int(lo - sizes[i]), replace=False)
        chromosomes[i, pick] = True
    for i in np.flatnonzero(sizes > hi):
        on = np.flatnonzero(chromosomes[i])
        pick = rng.choice(on, size=int(sizes[i] - hi), replace=False)
        chromosomes[i, pick] = False


def _initialise(n: int, p: int, lo: int, hi: int,
                rng: np.random.Generator) -> np.ndarray:
    """Panel sizes uniform on [lo, hi], then that many genes at random.

    Section 64.4 rejects independent per-bit Bernoulli initialisation, which
    would concentrate starting panels near p/2 and burn early generations on
    repair.
    """
    pop = np.zeros((n, p), dtype=bool)
    sizes = rng.integers(lo, hi + 1, size=n)
    for i, k in enumerate(sizes):
        pop[i, rng.choice(p, size=int(k), replace=False)] = True
    return pop


def _tournament(rank: np.ndarray, crowding: np.ndarray, n: int,
                rng: np.random.Generator) -> np.ndarray:
    """Binary tournament on Pareto rank, then crowding distance (section 64.4)."""
    a = rng.integers(0, rank.shape[0], size=n)
    b = rng.integers(0, rank.shape[0], size=n)
    a_wins = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowding[a] > crowding[b]))
    return np.where(a_wins, a, b)


# ----------------------------------------------------------------- driver
def run_nsga2(
    evaluate: Callable[[np.ndarray], float],
    n_candidates: int,
    settings: GASettings,
    seed: int,
    cache: dict[bytes, float] | None = None,
) -> GAResult:
    """Run one NSGA-II search.

    ``evaluate`` maps an array of frozen candidate indices to a cross-validated
    metric to be maximised.  It is called only on cache misses.

    ``cache`` is the section 12.2 fitness memo.  Passing one dictionary across
    the 10 GA seeds of a patient split is exactly the memoisation the protocol
    specifies -- keyed on (patient split, chromosome) -- and is where most of
    the saving comes from, because independent seeds rediscover many of the
    same panels.  It changes runtime, not results.
    """
    rng = np.random.default_rng(seed)
    p = int(n_candidates)
    lo, hi = settings.min_panel_size, min(settings.max_panel_size, p)
    mutation_rate = settings.mutation_rate if settings.mutation_rate else 1.0 / p

    cache = {} if cache is None else cache
    n_eval = 0
    n_hits = 0
    archive: dict[int, tuple[float, np.ndarray]] = {}

    def score(chromosomes: np.ndarray) -> np.ndarray:
        nonlocal n_eval, n_hits
        out = np.empty(chromosomes.shape[0], dtype=np.float64)
        for i, chrom in enumerate(chromosomes):
            key = np.packbits(chrom).tobytes()
            hit = cache.get(key)
            if hit is None:
                columns = np.flatnonzero(chrom).astype(np.int64)
                hit = float(evaluate(columns))
                cache[key] = hit
                n_eval += 1
                size = int(columns.shape[0])
                best = archive.get(size)
                # Tie-break inside the archive exactly as section 64.5 specifies,
                # so the archive is reproducible rather than arrival-ordered.
                if best is None or hit > best[0] or (
                        hit == best[0] and _lex_smaller(columns, best[1])):
                    archive[size] = (hit, columns)
            else:
                n_hits += 1
            out[i] = hit
        return out

    population = _initialise(settings.population_size, p, lo, hi, rng)
    metric = score(population)
    objectives = np.column_stack([-metric, population.sum(axis=1).astype(float)])
    trace: list[dict[str, float]] = []

    for generation in range(settings.generations):
        rank = _fast_nondominated_sort(objectives)
        crowding = np.zeros(rank.shape[0])
        for r in np.unique(rank):
            mask = rank == r
            crowding[mask] = _crowding_distance(objectives[mask])

        parents_idx = _tournament(rank, crowding, settings.population_size, rng)
        parents = population[parents_idx]
        mates = population[_tournament(rank, crowding, settings.population_size, rng)]

        # Uniform crossover (section 64.4).
        offspring = parents.copy()
        do_cross = rng.uniform(size=settings.population_size) < settings.crossover_probability
        swap = rng.uniform(size=(settings.population_size, p)) < 0.5
        swap &= do_cross[:, None]
        offspring[swap] = mates[swap]

        # Per-bit mutation at rate 1/p (section 20).
        flip = rng.uniform(size=(settings.population_size, p)) < mutation_rate
        offspring ^= flip
        _repair(offspring, lo, hi, rng)

        child_metric = score(offspring)
        child_obj = np.column_stack([-child_metric,
                                     offspring.sum(axis=1).astype(float)])

        # Elitist (mu + lambda) survival.
        merged = np.vstack([population, offspring])
        merged_metric = np.concatenate([metric, child_metric])
        merged_obj = np.vstack([objectives, child_obj])

        merged_rank = _fast_nondominated_sort(merged_obj)
        merged_crowd = np.zeros(merged_rank.shape[0])
        for r in np.unique(merged_rank):
            mask = merged_rank == r
            merged_crowd[mask] = _crowding_distance(merged_obj[mask])
        order = np.lexsort((-merged_crowd, merged_rank))[:settings.population_size]

        population = merged[order]
        metric = merged_metric[order]
        objectives = merged_obj[order]

        if settings.trace_every and (generation % settings.trace_every == 0
                                     or generation == settings.generations - 1):
            front = nondominated_front(objectives)
            sizes = objectives[:, 1]
            trace.append({
                "generation": int(generation),
                "hypervolume": hypervolume_2d(metric[front], sizes[front].astype(int), ref_size=hi + 1),
                "best_metric": float(metric.max()),
                "median_panel_size": float(np.median(sizes)),
                "front_size": int(front.sum()),
                "n_evaluations": int(n_eval),
                "n_cache_hits": int(n_hits),
            })

    front_mask = nondominated_front(objectives)
    front_panels = [np.flatnonzero(c).astype(np.int64)
                    for c in population[front_mask]]
    front_metric = metric[front_mask]
    front_size = objectives[front_mask, 1].astype(int)

    panel, best_metric, best_size = _select_representative(
        front_panels, front_metric, front_size, settings.near_optimal_tolerance)

    return GAResult(
        selected_panel=panel,
        selected_metric=float(best_metric),
        selected_size=int(best_size),
        front_panels=front_panels,
        front_metric=front_metric,
        front_size=front_size,
        archive_best_by_size=archive,
        trace=trace,
        n_evaluations=n_eval,
        n_cache_hits=n_hits,
        seed=int(seed),
    )


def _lex_smaller(a: np.ndarray, b: np.ndarray) -> bool:
    """Lexicographic comparison of two sorted index sets (section 64.5)."""
    return tuple(int(v) for v in a) < tuple(int(v) for v in b)


def _select_representative(
    panels: list[np.ndarray],
    metric: np.ndarray,
    size: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, float, int]:
    """One representative panel per run (protocol sections 21 and 64.5).

    Acceptable = metric >= max(metric) - tolerance, with a FIXED tolerance
    rather than a fold-derived one-standard-error rule: five cross-validation
    folds are correlated, so the apparent standard error of the adaptively best
    panel is not a reliable sampling-uncertainty estimate (section 21).

    Among acceptable panels take the smallest; ties broken by the highest
    cross-validated metric, then by the lexicographically smallest index set.
    """
    if not panels:
        return np.empty(0, dtype=np.int64), 0.5, 0
    acceptable = np.flatnonzero(metric >= metric.max() - tolerance)
    smallest = size[acceptable].min()
    candidates = acceptable[size[acceptable] == smallest]
    best_metric = metric[candidates].max()
    candidates = candidates[metric[candidates] == best_metric]
    best = min(candidates, key=lambda i: tuple(int(v) for v in panels[i]))
    return panels[best], float(metric[best]), int(size[best])
