"""Probe-set design for small candidate-hypothesis problems."""

from __future__ import annotations
import numpy as np


def pairwise_separation_score(templates: np.ndarray, eps: float = 1e-12) -> tuple[int, float, float]:
    """Lexicographic score: separated pairs, weakest positive separation, total."""
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


def greedy_probe_set(
    single_probe_bank: np.ndarray,
    *,
    budget: int,
) -> tuple[list[int], list[dict]]:
    """Greedily choose probes that separate candidate models.

    `single_probe_bank` has shape [probe, hypothesis, time]. The planner first
    rewards splitting previously aliased pairs, then weakest positive separation,
    then total pairwise separation. This is a deterministic design benchmark,
    not yet Bayesian adaptive experiment design.
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
            templates = np.concatenate(
                [bank[i] for i in trial],
                axis=1,
            )
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
                "separated_pairs": int(score[0]),
                "min_positive_pair_distance": float(score[1]),
                "sum_pair_distances": float(score[2]),
            }
        )

    return selected, trace
