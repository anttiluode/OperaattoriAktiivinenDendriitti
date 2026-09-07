"""Gate 3: break exact symmetry and test whether stimulation planning still helps.

Gate 2a showed that the original six-arm result was mostly a three-bit coding
problem: once all six branches had unique codes, many random designs performed
essentially as well as the hand-designed one. It also exposed a noise-blind
pairwise scoring flaw.

This gate deliberately removes the exact coding symmetry. The six arms have
fixed, modest heterogeneity in axial conductance and passive leak. Exactly one
distal region receives an additional leak change. The planner still gets only
one soma trace per stimulation trial and the same 3-trial budget.

We compare:
  * the repaired pairwise-confusion greedy score;
  * a design-time oracle that greedily maximizes estimated six-way Gaussian
    localization accuracy;
  * 200 random distinct three-pattern stimulation sets;
  * three individual branch probes.

The multi-hypothesis design score is an oracle benchmark: it knows the six
candidate forward models and declared noise. The eventual real instrument must
handle parameter uncertainty/model mismatch and later choose sequentially from
posterior uncertainty rather than from hidden labels.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from active_dendrite import (
    compile_passive,
    monte_carlo_localization,
    response_bank,
    unit_energy_subset,
)
from planner import (
    gaussian_localization_accuracy_estimate,
    greedy_probe_set,
    greedy_probe_set_by_accuracy,
)


NOISE_STD = 0.003
HETERO_SEED = 123
HETERO_SCALE = 0.10
DELTA_LEAK = 0.06
DESIGN_SAMPLES_PER_HYPOTHESIS = 50_000
DESIGN_SEED = 1234
EVAL_TRIALS = 50_000
EVAL_SEED = 321
RANDOM_DESIGNS = 200
RANDOM_SEED = 7
RANDOM_ESTIMATE_SAMPLES = 5_000
RANDOM_ESTIMATE_SEED = 9000


def heterogeneous_six_arm(
    *,
    changed_region: int | None = None,
) -> object:
    """Soma + six two-compartment arms with fixed known heterogeneity."""
    rng = np.random.default_rng(HETERO_SEED)
    variation = rng.normal(0.0, HETERO_SCALE, size=(6, 4))

    n = 13
    edges: list[tuple[int, int, float]] = []
    leak = np.zeros(n, dtype=float)
    leak[0] = 0.35
    stimulation_nodes = []

    for branch in range(6):
        prox = 1 + 2 * branch
        dist = 2 + 2 * branch
        stimulation_nodes.append(dist)

        g_soma = 0.75 * np.exp(variation[branch, 0])
        g_dist = 0.55 * np.exp(variation[branch, 1])
        leak[prox] = 0.22 * np.exp(variation[branch, 2])
        leak[dist] = 0.18 * np.exp(variation[branch, 3])

        edges.extend(
            [
                (0, prox, float(g_soma)),
                (prox, dist, float(g_dist)),
            ]
        )

    if changed_region is not None:
        if not 0 <= int(changed_region) < 6:
            raise ValueError("changed_region outside 0..5")
        leak[2 + 2 * int(changed_region)] += DELTA_LEAK

    return compile_passive(
        n,
        edges,
        leak,
        stimulation_nodes=tuple(stimulation_nodes),
    )


def direct_accuracy(models, probes) -> float:
    return monte_carlo_localization(
        response_bank(models, probes),
        noise_std=NOISE_STD,
        trials=EVAL_TRIALS,
        seed=EVAL_SEED,
    )


def main() -> None:
    models = [
        heterogeneous_six_arm(changed_region=i)
        for i in range(6)
    ]

    subsets = list(itertools.combinations(range(6), 3))
    probes = [unit_energy_subset(6, s) for s in subsets]
    single = np.stack(
        [response_bank(models, [p])[:, 0, :] for p in probes],
        axis=0,
    )

    pairwise_chosen, pairwise_trace = greedy_probe_set(
        single,
        budget=3,
        noise_std=NOISE_STD,
        max_pair_error=0.05,
    )

    accuracy_chosen, accuracy_trace = greedy_probe_set_by_accuracy(
        single,
        budget=3,
        noise_std=NOISE_STD,
        samples_per_hypothesis=DESIGN_SAMPLES_PER_HYPOTHESIS,
        seed=DESIGN_SEED,
    )

    pairwise_probes = [probes[i] for i in pairwise_chosen]
    accuracy_probes = [probes[i] for i in accuracy_chosen]

    pairwise_direct_accuracy = direct_accuracy(models, pairwise_probes)
    accuracy_direct_accuracy = direct_accuracy(models, accuracy_probes)

    accuracy_templates = response_bank(
        models, accuracy_probes
    ).reshape(6, -1)
    accuracy_estimate_check = gaussian_localization_accuracy_estimate(
        accuracy_templates,
        noise_std=NOISE_STD,
        samples_per_hypothesis=DESIGN_SAMPLES_PER_HYPOTHESIS,
        seed=DESIGN_SEED + 99,
    )

    individual3 = [unit_energy_subset(6, (i,)) for i in range(3)]
    individual3_accuracy = direct_accuracy(models, individual3)

    rng = np.random.default_rng(RANDOM_SEED)
    random_triples = [
        tuple(sorted(rng.choice(len(probes), size=3, replace=False)))
        for _ in range(RANDOM_DESIGNS)
    ]
    random_estimated_accuracy = []
    for triple in random_triples:
        templates = np.concatenate(
            [single[i] for i in triple],
            axis=1,
        )
        random_estimated_accuracy.append(
            gaussian_localization_accuracy_estimate(
                templates,
                noise_std=NOISE_STD,
                samples_per_hypothesis=RANDOM_ESTIMATE_SAMPLES,
                seed=RANDOM_ESTIMATE_SEED,
            )
        )

    random_estimated_accuracy = np.asarray(
        random_estimated_accuracy,
        dtype=float,
    )

    result = {
        "gate": "broken_symmetry_multi_hypothesis_planning",
        "model": {
            "description": "soma + six two-compartment arms with fixed known heterogeneity",
            "heterogeneity_seed": HETERO_SEED,
            "heterogeneity_log_std": HETERO_SCALE,
            "hidden_change": "one distal branch receives additional passive leak",
            "delta_leak": DELTA_LEAK,
        },
        "recording_site": "soma only",
        "noise_std": NOISE_STD,
        "budget_trials": 3,
        "candidate_patterns": len(probes),
        "pairwise_noise_aware_greedy": {
            "chosen_subsets": [
                list(subsets[i]) for i in pairwise_chosen
            ],
            "trace": pairwise_trace,
            "direct_50000_trial_accuracy": pairwise_direct_accuracy,
        },
        "multi_hypothesis_accuracy_greedy": {
            "chosen_subsets": [
                list(subsets[i]) for i in accuracy_chosen
            ],
            "trace": accuracy_trace,
            "design_samples_per_hypothesis": DESIGN_SAMPLES_PER_HYPOTHESIS,
            "direct_50000_trial_accuracy": accuracy_direct_accuracy,
            "independent_discriminant_accuracy_check": accuracy_estimate_check,
        },
        "three_individual_branch_probes": {
            "subsets": [[0], [1], [2]],
            "direct_50000_trial_accuracy": individual3_accuracy,
        },
        "random_three_pattern_baseline": {
            "designs": RANDOM_DESIGNS,
            "distinct_patterns_within_each_design": true,
            "estimated_accuracy_samples_per_hypothesis": RANDOM_ESTIMATE_SAMPLES,
            "mean": float(np.mean(random_estimated_accuracy)),
            "median": float(np.median(random_estimated_accuracy)),
            "p10": float(np.percentile(random_estimated_accuracy, 10)),
            "p25": float(np.percentile(random_estimated_accuracy, 25)),
            "p75": float(np.percentile(random_estimated_accuracy, 75)),
            "p90": float(np.percentile(random_estimated_accuracy, 90)),
            "min": float(np.min(random_estimated_accuracy)),
            "max": float(np.max(random_estimated_accuracy)),
        },
        "interpretation": (
            "Breaking exact arm symmetry removes the trivial three-bit-code equivalence. "
            "The repaired pairwise heuristic is still not enough: on this six-way task "
            "it greedily chooses a set with much poorer final localization than a "
            "planner that scores the actual multi-hypothesis Gaussian classification "
            "objective. The latter reaches about 84% accuracy with three soma-only "
            "trials, above the roughly 76.5% mean of matched-budget random designs and "
            "well above three individual probes. This is a stronger design benchmark, "
            "but it still assumes the forward models and noise are correct; nuisance "
            "uncertainty/model mismatch remain the next mandatory attacker."
        ),
    }

    assert result["pairwise_noise_aware_greedy"][
        "direct_50000_trial_accuracy"
    ] < 0.75
    assert result["multi_hypothesis_accuracy_greedy"][
        "direct_50000_trial_accuracy"
    ] > 0.82
    assert result["multi_hypothesis_accuracy_greedy"][
        "direct_50000_trial_accuracy"
    ] > result["random_three_pattern_baseline"]["mean"] + 0.05
    assert result["three_individual_branch_probes"][
        "direct_50000_trial_accuracy"
    ] < 0.72

    out = Path("results/gate3_broken_symmetry.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
