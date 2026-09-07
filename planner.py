"""Probe-set design for small candidate-hypothesis problems.

The original Gate-2 planner counted any nonzero template difference as a
"separated" hypothesis pair. That is useful as an algebraic identifiability
receipt, but it is not a sound experiment-design score: a difference many
orders of magnitude below the recording noise should not outrank one robust
separation merely because it is nonzero.

This module therefore keeps the original scorer as an explicit legacy/audit
function and adds a noise-aware Gaussian pair-confusion score.
"""

from __future__ import annotations

import math
import numpy as np


def pairwise_separation_score(
    templates: np.ndarray,
    eps: float = 1e-12,
) -> tuple[int, float, float]:
    """LEGACY algebraic score.

    Returns `(nonzero_pairs, weakest_positive_distance, total_distance)`.
    This score intentionally ignores measurement noise and is retained only so
    old Gate-2 results and the scoring-flaw audit remain reproducible.
    """
    templates = np.asarray(templates, dtype=float)
    distances = []
    separated = 0
    for i in range(templates.shape[0]):
        for j in range(i):
            d = float(np.linalg.norm(templates[i] - templates[j]))
            distances.append(d)
            separated += int(d > eps)
    positive = [d for d in distances if d > eps]
    min_positive = min(positive) if positive else 0.0
    return separated, float(min_positive), float(sum(distances))


def _pair_error_from_distance(distance: float, noise_std: float) -> float:
    """Equal-prior Bayes error for two isotropic-Gaussian templates.

    If both hypotheses have covariance sigma^2 I and mean separation `d`, the
    optimal two-class error is Phi(-d/(2 sigma)).
    """
    sigma = float(noise_std)
    if sigma <= 0.0:
        raise ValueError("noise_std must be positive")
    D = float(distance) / sigma
    return 0.5 * math.erfc(D / (2.0 * math.sqrt(2.0)))


def noise_aware_pairwise_score(
    templates: np.ndarray,
    *,
    noise_std: float,
    max_pair_error: float = 0.05,
) -> dict:
    """Noise-aware pairwise design surrogate.

    A pair is counted as *resolved* only if its predicted equal-prior Gaussian
    two-hypothesis error is <= `max_pair_error`. Among designs resolving the
    same number of pairs, lower total/worst pair-confusion is preferred.

    This is still a finite-hypothesis design heuristic, not full Bayesian
    expected information gain. It is deliberately simple and auditable.
    """
    templates = np.asarray(templates, dtype=float)
    if templates.ndim != 2:
        raise ValueError("templates must have shape [hypothesis, features]")
    if not (0.0 < float(max_pair_error) < 0.5):
        raise ValueError("max_pair_error must lie in (0, 0.5)")

    distances: list[float] = []
    d2_values: list[float] = []
    pair_errors: list[float] = []

    sigma = float(noise_std)
    if sigma <= 0.0:
        raise ValueError("noise_std must be positive")

    for i in range(templates.shape[0]):
        for j in range(i):
            d = float(np.linalg.norm(templates[i] - templates[j]))
            distances.append(d)
            D = d / sigma
            d2_values.append(float(D * D))
            pair_errors.append(_pair_error_from_distance(d, sigma))

    if not pair_errors:
        return {
            "resolved_pairs": 0,
            "total_pairs": 0,
            "sum_pair_error": 0.0,
            "mean_pair_error": 0.0,
            "worst_pair_error": 0.0,
            "min_mahalanobis2": 0.0,
            "sum_mahalanobis2": 0.0,
            "min_template_distance": 0.0,
        }

    return {
        "resolved_pairs": int(
            sum(e <= float(max_pair_error) for e in pair_errors)
        ),
        "total_pairs": int(len(pair_errors)),
        "sum_pair_error": float(sum(pair_errors)),
        "mean_pair_error": float(np.mean(pair_errors)),
        "worst_pair_error": float(max(pair_errors)),
        "min_mahalanobis2": float(min(d2_values)),
        "sum_mahalanobis2": float(sum(d2_values)),
        "min_template_distance": float(min(distances)),
    }


def greedy_probe_set_legacy(
    single_probe_bank: np.ndarray,
    *,
    budget: int,
) -> tuple[list[int], list[dict]]:
    """Original Gate-2 algebraic planner, retained for the audit."""
    bank = np.asarray(single_probe_bank, dtype=float)
    selected: list[int] = []
    trace: list[dict] = []

    for step in range(int(budget)):
        best = None
        for probe_index in range(bank.shape[0]):
            if probe_index in selected:
                continue
            trial = selected + [probe_index]
            templates = np.concatenate([bank[i] for i in trial], axis=1)
            score = pairwise_separation_score(templates)
            key = (score[0], score[1], score[2], -probe_index)
            if best is None or key > best[0]:
                best = (key, probe_index, score)
        if best is None:
            break
        _, chosen, score = best
        selected.append(int(chosen))
        trace.append(
            {
                "step": step + 1,
                "probe_index": int(chosen),
                "nonzero_pairs": int(score[0]),
                "min_positive_pair_distance": float(score[1]),
                "sum_pair_distances": float(score[2]),
            }
        )

    return selected, trace


def greedy_probe_set(
    single_probe_bank: np.ndarray,
    *,
    budget: int,
    noise_std: float,
    max_pair_error: float = 0.05,
) -> tuple[list[int], list[dict]]:
    """Greedily choose a noise-aware stimulation set.

    `single_probe_bank` has shape `[probe, hypothesis, time]`.

    The lexicographic score is:

    1. maximize the number of hypothesis pairs whose predicted pairwise error is
       below the declared tolerance;
    2. minimize total pairwise confusion;
    3. minimize the worst pairwise confusion;
    4. maximize the weakest and then total Mahalanobis separation.

    This fixes the original failure where many numerically nonzero but
    sub-noise distinctions could beat a smaller number of useful distinctions.
    """
    bank = np.asarray(single_probe_bank, dtype=float)
    selected: list[int] = []
    trace: list[dict] = []

    for step in range(int(budget)):
        best = None
        for probe_index in range(bank.shape[0]):
            if probe_index in selected:
                continue
            trial = selected + [probe_index]
            templates = np.concatenate([bank[i] for i in trial], axis=1)
            score = noise_aware_pairwise_score(
                templates,
                noise_std=float(noise_std),
                max_pair_error=float(max_pair_error),
            )
            key = (
                score["resolved_pairs"],
                -score["sum_pair_error"],
                -score["worst_pair_error"],
                score["min_mahalanobis2"],
                score["sum_mahalanobis2"],
                -probe_index,
            )
            if best is None or key > best[0]:
                best = (key, probe_index, score)
        if best is None:
            break

        _, chosen, score = best
        selected.append(int(chosen))
        trace.append(
            {
                "step": step + 1,
                "probe_index": int(chosen),
                "resolved_pairs": int(score["resolved_pairs"]),
                "total_pairs": int(score["total_pairs"]),
                "sum_pair_error": float(score["sum_pair_error"]),
                "worst_pair_error": float(score["worst_pair_error"]),
                "min_mahalanobis2": float(score["min_mahalanobis2"]),
                "sum_mahalanobis2": float(score["sum_mahalanobis2"]),
                "min_template_distance": float(
                    score["min_template_distance"]
                ),
            }
        )

    return selected, trace
