"""Gate 1: six-region localization from a single soma sensor.

The hidden change is one of six equal distal regions. Every measurement is a
positive, unit-energy stimulation pattern and one soma voltage trace.

Three balanced multi-site patterns assign each branch a unique three-bit code.
This is a deliberately transparent designed benchmark, not yet an adaptive
planner. We compare it with three individual-site probes, six individual-site
probes, and 100 random balanced three-pattern designs.
"""

from __future__ import annotations
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

NOISE_STD = 0.003
TRIALS = 5000
SEED = 41


def accuracy(models, probes, seed):
    bank = response_bank(models, probes)
    return monte_carlo_localization(
        bank,
        noise_std=NOISE_STD,
        trials=TRIALS,
        seed=seed,
    )


def min_template_distance(models, probes):
    bank = response_bank(models, probes).reshape(6, -1)
    return float(
        min(
            np.linalg.norm(bank[i] - bank[j])
            for i in range(6)
            for j in range(i)
        )
    )


def main() -> None:
    models = [six_arm(changed_region=i) for i in range(6)]

    designed_subsets = [
        (0, 1, 2),
        (0, 3, 4),
        (1, 3, 5),
    ]
    designed = [unit_energy_subset(6, s) for s in designed_subsets]

    individual3 = [unit_energy_subset(6, (i,)) for i in range(3)]
    individual6 = [unit_energy_subset(6, (i,)) for i in range(6)]

    acc_designed = accuracy(models, designed, SEED)
    acc_ind3 = accuracy(models, individual3, SEED + 1)
    acc_ind6 = accuracy(models, individual6, SEED + 2)

    random_sets = random_balanced_probe_sets(
        count=100,
        probes_per_set=3,
        subset_size=3,
        seed=4,
    )
    random_acc = [
        accuracy(models, probes, 1000 + i)
        for i, probes in enumerate(random_sets)
    ]
    random_unique = [unique_probe_codes(p) for p in random_sets]

    result = {
        "gate": "six_region_soma_only_localization",
        "model": "soma plus six mirror-equal two-compartment arms",
        "hidden_change": "one distal region has increased passive leak",
        "recording_sites": 1,
        "recording_site": "soma",
        "noise_std": NOISE_STD,
        "trials_per_design": TRIALS,
        "designed_balanced_patterns": [list(s) for s in designed_subsets],
        "designed_pattern_unique_codes": unique_probe_codes(designed),
        "designed_3_trial_accuracy": acc_designed,
        "designed_min_pairwise_template_l2": min_template_distance(models, designed),
        "first_3_individual_sites_accuracy": acc_ind3,
        "all_6_individual_sites_accuracy": acc_ind6,
        "random_balanced_3_pattern_designs": {
            "count": len(random_acc),
            "mean_accuracy": float(np.mean(random_acc)),
            "median_accuracy": float(np.median(random_acc)),
            "min_accuracy": float(np.min(random_acc)),
            "max_accuracy": float(np.max(random_acc)),
            "unique_code_histogram": {
                str(k): int(sum(u == k for u in random_unique))
                for k in sorted(set(random_unique))
            },
        },
        "interpretation": (
            "With a known symmetric morphology and one soma recording site, "
            "designed multi-site stimulation can encode six candidate hidden branch "
            "changes into distinct response signatures in three trials. This is a "
            "toy identification benchmark, not full dendritic tomography. The next "
            "step is an adaptive planner on a non-symmetric Operaattori morphology "
            "with model mismatch and calibrated uncertainty."
        ),
    }

    assert result["designed_pattern_unique_codes"] == 6
    assert result["designed_3_trial_accuracy"] > 0.985
    assert result["first_3_individual_sites_accuracy"] < 0.75
    assert result["all_6_individual_sites_accuracy"] > 0.995
    assert result["random_balanced_3_pattern_designs"]["mean_accuracy"] < result["designed_3_trial_accuracy"]

    out = Path("results/gate1_six_region.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
