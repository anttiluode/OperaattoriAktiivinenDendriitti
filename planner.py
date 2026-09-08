"""Probe-set design utilities for the active-dendrite gates.

Legacy scores are retained for auditability. Newer gates should use the
noise-aware or multi-hypothesis objectives below.
"""

from __future__ import annotations

import math
import numpy as np


def pairwise_separation_score(
    templates: np.ndarray,
    eps: float = 1e-12,
) -> tuple[int, float, float]:
    """LEGACY algebraic score: nonzero pairs, weakest positive gap, total gap."""
    templates = np.asarray(templates, dtype=float)
    distances = []
    separated = 0
    for i in range(templates.shape[0]):
        for j in range(i):
            d = float(np.linalg.norm(templates[i] - templates[j]))
            distances.append(d)
            separated += int(d > eps)
    positive = [d for d in distances if d > eps]
    return (
        separated,
        float(min(positive) if positive else 0.0),
        float(sum(distances)),
    )


def _pair_error_from_distance(distance: float, noise_std: float) -> float:
    """Equal-prior Bayes error for two isotropic-Gaussian templates."""
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
    """Noise-aware pairwise design surrogate."""
    templates = np.asarray(templates, dtype=float)
    if templates.ndim != 2:
        raise ValueError("templates must have shape [hypothesis, features]")
    if not (0.0 < float(max_pair_error) < 0.5):
        raise ValueError("max_pair_error must lie in (0, 0.5)")
    sigma = float(noise_std)
    if sigma <= 0.0:
        raise ValueError("noise_std must be positive")

    distances = []
    d2_values = []
    pair_errors = []
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


def gaussian_localization_accuracy_estimate(
    templates: np.ndarray,
    *,
    noise_std: float,
    samples_per_hypothesis: int = 10000,
    seed: int = 0,
    tie_tol: float = 1e-12,
) -> float:
    """Estimate nearest-template accuracy under isotropic Gaussian noise.

    Exact-equivalence classes are handled with an explicit uniform tie rule.
    This matters for deliberately blind controls: H identical equally likely
    templates must score 1/H rather than zero.

    For each true hypothesis i, non-tied competitors are handled in the exact
    Gaussian discriminant subspace. If m templates are tied with i, the
    probability of a correct label is the probability that the equivalence
    class beats all non-tied templates, multiplied by 1/m.
    """
    templates = np.asarray(templates, dtype=float)
    if templates.ndim != 2:
        raise ValueError("templates must have shape [hypothesis, features]")
    sigma = float(noise_std)
    if sigma <= 0.0:
        raise ValueError("noise_std must be positive")
    samples = int(samples_per_hypothesis)
    if samples <= 0:
        raise ValueError("samples_per_hypothesis must be positive")
    if float(tie_tol) < 0.0:
        raise ValueError("tie_tol must be nonnegative")

    H = int(templates.shape[0])
    if H <= 1:
        return 1.0

    rng = np.random.default_rng(int(seed))
    per_hypothesis = []

    for i in range(H):
        diffs = templates - templates[i]
        norms = np.linalg.norm(diffs, axis=1)
        scale = max(1.0, float(np.linalg.norm(templates[i])))
        tied = norms <= float(tie_tol) * scale
        tie_size = int(np.sum(tied))

        competitors = [
            j for j in range(H)
            if j != i and not bool(tied[j])
        ]
        if not competitors:
            per_hypothesis.append(1.0 / float(tie_size))
            continue

        V = (templates[competitors] - templates[i]) / sigma
        gram = V @ V.T
        vals, vecs = np.linalg.eigh(gram)
        vals = np.clip(vals, 0.0, None)
        L = vecs * np.sqrt(vals)

        z = rng.normal(size=(samples, len(competitors)))
        projected_noise = z @ L.T
        thresholds = 0.5 * np.diag(gram)
        class_wins = np.all(
            projected_noise < thresholds[None, :],
            axis=1,
        )
        per_hypothesis.append(
            float(np.mean(class_wins)) / float(tie_size)
        )

    return float(np.mean(per_hypothesis))


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
        trace.append({
            "step": step + 1,
            "probe_index": int(chosen),
            "nonzero_pairs": int(score[0]),
            "min_positive_pair_distance": float(score[1]),
            "sum_pair_distances": float(score[2]),
        })
    return selected, trace


def greedy_probe_set(
    single_probe_bank: np.ndarray,
    *,
    budget: int,
    noise_std: float,
    max_pair_error: float = 0.05,
) -> tuple[list[int], list[dict]]:
    """Greedily choose a noise-aware stimulation set using pairwise confusion."""
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
        trace.append({
            "step": step + 1,
            "probe_index": int(chosen),
            "resolved_pairs": int(score["resolved_pairs"]),
            "total_pairs": int(score["total_pairs"]),
            "sum_pair_error": float(score["sum_pair_error"]),
            "worst_pair_error": float(score["worst_pair_error"]),
            "min_mahalanobis2": float(score["min_mahalanobis2"]),
            "sum_mahalanobis2": float(score["sum_mahalanobis2"]),
            "min_template_distance": float(score["min_template_distance"]),
        })
    return selected, trace


def greedy_probe_set_by_accuracy(
    single_probe_bank: np.ndarray,
    *,
    budget: int,
    noise_std: float,
    samples_per_hypothesis: int = 20000,
    seed: int = 0,
) -> tuple[list[int], list[dict]]:
    """Greedily maximize estimated multi-hypothesis localization accuracy."""
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
            accuracy = gaussian_localization_accuracy_estimate(
                templates,
                noise_std=float(noise_std),
                samples_per_hypothesis=int(samples_per_hypothesis),
                seed=int(seed) + step,
            )
            key = (accuracy, -probe_index)
            if best is None or key > best[0]:
                best = (key, probe_index, accuracy)
        if best is None:
            break
        _, chosen, accuracy = best
        selected.append(int(chosen))
        trace.append({
            "step": step + 1,
            "probe_index": int(chosen),
            "estimated_localization_accuracy": float(accuracy),
        })
    return selected, trace
