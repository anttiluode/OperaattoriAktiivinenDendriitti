"""Gate 2a: attack the early stimulation-planner claim.

This audit was motivated by two observations:

1. In the exactly symmetric six-arm toy, most of the apparent designed-vs-random
   advantage may simply come from whether the three binary stimulation patterns
   assign all six branches distinct membership codes.
2. The original planner counted every nonzero template difference as a useful
   separation, even if that difference was far below the declared recording
   noise.

The audit therefore conditions the random comparison on code uniqueness,
enumerates all distinct three-pattern code-valid designs, and includes a tiny
synthetic counterexample where the old score provably chooses sub-noise
separations over a genuinely informative probe.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from active_dendrite import (
    monte_carlo_localization,
    random_balanced_probe_sets,
    response_bank,
    six_arm,
    unique_probe_codes,
    unit_energy_subset,
)
from planner import greedy_probe_set, greedy_probe_set_legacy


NOISE_STD = 0.003
TRIALS = 5000
MAX_PAIR_ERROR = 0.05


def accuracy(models, probes, seed):
    return monte_carlo_localization(
        response_bank(models, probes),
        noise_std=NOISE_STD,
        trials=TRIALS,
        seed=seed,
    )


def min_distance(models, probes):
    templates = response_bank(models, probes).reshape(6, -1)
    return float(
        min(
            np.linalg.norm(templates[i] - templates[j])
            for i in range(6)
            for j in range(i)
        )
    )


def main() -> None:
    models = [six_arm(changed_region=i) for i in range(6)]

    designed_subsets = [(0, 1, 2), (0, 3, 4), (1, 3, 5)]
    designed = [unit_energy_subset(6, s) for s in designed_subsets]
    designed_accuracy = accuracy(models, designed, 41)

    # Reproduce the original Gate-1 random comparison exactly.
    random_sets = random_balanced_probe_sets(
        count=100,
        probes_per_set=3,
        subset_size=3,
        seed=4,
    )
    random_accuracy = [
        accuracy(models, probes, 1000 + i)
        for i, probes in enumerate(random_sets)
    ]
    random_codes = [unique_probe_codes(p) for p in random_sets]

    unique6_accuracy = [
        a for a, c in zip(random_accuracy, random_codes) if c == 6
    ]

    # Enumerate all unordered triples of distinct 3-of-6 patterns.
    subsets = list(itertools.combinations(range(6), 3))
    probes = [unit_energy_subset(6, s) for s in subsets]
    all_triples = list(itertools.combinations(range(len(probes)), 3))

    code_valid_triples = []
    code_valid_min_distances = []
    rounded_distance_spectra = set()

    for triple in all_triples:
        pset = [probes[i] for i in triple]
        if unique_probe_codes(pset) != 6:
            continue
        code_valid_triples.append(triple)
        bank = response_bank(models, pset).reshape(6, -1)
        distances = sorted(
            float(np.linalg.norm(bank[i] - bank[j]))
            for i in range(6)
            for j in range(i)
        )
        code_valid_min_distances.append(distances[0])
        rounded_distance_spectra.add(
            tuple(round(d, 12) for d in distances)
        )

    # Synthetic scoring attack. Probe 0 makes all three templates technically
    # different but only by 1e-6 under sigma=1e-2 noise. Probe 1 leaves one pair
    # exactly aliased but robustly separates the third hypothesis by 0.1.
    synthetic_noise = 0.01
    tiny_all_pairs = np.array([[0.0], [1e-6], [2e-6]], dtype=float)
    useful_one_split = np.array([[0.0], [0.0], [0.1]], dtype=float)
    synthetic_bank = np.stack([tiny_all_pairs, useful_one_split], axis=0)

    legacy_choice, legacy_trace = greedy_probe_set_legacy(
        synthetic_bank,
        budget=1,
    )
    fixed_choice, fixed_trace = greedy_probe_set(
        synthetic_bank,
        budget=1,
        noise_std=synthetic_noise,
        max_pair_error=MAX_PAIR_ERROR,
    )

    result = {
        "gate": "planner_claim_and_scoring_audit",
        "symmetric_six_arm_random_design_audit": {
            "designed_accuracy": designed_accuracy,
            "all_100_random_mean_accuracy": float(np.mean(random_accuracy)),
            "random_unique_code_histogram": {
                str(k): int(sum(c == k for c in random_codes))
                for k in sorted(set(random_codes))
            },
            "random_designs_with_6_unique_codes": len(unique6_accuracy),
            "mean_accuracy_conditioned_on_6_unique_codes": float(
                np.mean(unique6_accuracy)
            ),
            "median_accuracy_conditioned_on_6_unique_codes": float(
                np.median(unique6_accuracy)
            ),
            "min_accuracy_conditioned_on_6_unique_codes": float(
                np.min(unique6_accuracy)
            ),
            "max_accuracy_conditioned_on_6_unique_codes": float(
                np.max(unique6_accuracy)
            ),
            "designed_minus_unique_random_mean_accuracy": float(
                designed_accuracy - np.mean(unique6_accuracy)
            ),
        },
        "exhaustive_distinct_three_pattern_audit": {
            "candidate_3_of_6_patterns": len(probes),
            "unordered_distinct_pattern_triples": len(all_triples),
            "triples_with_6_unique_branch_codes": len(code_valid_triples),
            "fraction_code_valid": float(
                len(code_valid_triples) / len(all_triples)
            ),
            "min_of_code_valid_min_template_distances": float(
                min(code_valid_min_distances)
            ),
            "max_of_code_valid_min_template_distances": float(
                max(code_valid_min_distances)
            ),
            "distinct_pairwise_distance_spectra_after_rounding_1e-12": int(
                len(rounded_distance_spectra)
            ),
        },
        "synthetic_scoring_attack": {
            "noise_std": synthetic_noise,
            "probe_0": "three sub-noise but nonzero template differences",
            "probe_1": "one exact alias plus one robust 0.1 separation",
            "legacy_choice": int(legacy_choice[0]),
            "legacy_trace": legacy_trace,
            "noise_aware_choice": int(fixed_choice[0]),
            "noise_aware_trace": fixed_trace,
        },
        "interpretation": (
            "The early 99% result survives as a coding/identifiability toy, but the "
            "strong designed-vs-random claim does not. Among the original random "
            "sets, the ones that happen to assign six distinct branch codes already "
            "perform essentially as well as the hand-designed set. Exhaustively, all "
            "code-valid triples have the same pairwise distance spectrum in the exact "
            "symmetric model. The old planner also has a real noise-blind scoring flaw: "
            "it can prefer many numerically nonzero sub-noise distinctions over a "
            "useful split. The revised planner scores separation relative to declared "
            "measurement noise. A meaningful planner advantage must now be shown after "
            "breaking exact symmetry and adding nuisance/model uncertainty."
        ),
    }

    assert result["symmetric_six_arm_random_design_audit"][
        "random_designs_with_6_unique_codes"
    ] == 42
    assert abs(
        result["symmetric_six_arm_random_design_audit"][
            "mean_accuracy_conditioned_on_6_unique_codes"
        ]
        - 0.9915761904761903
    ) < 1e-12
    assert result["exhaustive_distinct_three_pattern_audit"][
        "triples_with_6_unique_branch_codes"
    ] == 480
    assert result["exhaustive_distinct_three_pattern_audit"][
        "distinct_pairwise_distance_spectra_after_rounding_1e-12"
    ] == 1
    assert result["synthetic_scoring_attack"]["legacy_choice"] == 0
    assert result["synthetic_scoring_attack"]["noise_aware_choice"] == 1

    out = Path("results/gate2a_planner_audit.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
