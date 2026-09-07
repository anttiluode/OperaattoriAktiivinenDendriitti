"""Gate 2: the computer chooses a three-trial soma-only stimulation set."""

from __future__ import annotations
import itertools
import json
from pathlib import Path
import numpy as np

from active_dendrite import (
    monte_carlo_localization,
    response_bank,
    six_arm,
    unit_energy_subset,
)
from planner import greedy_probe_set

NOISE_STD = 0.003
TRIALS = 5000
SEED = 55


def main() -> None:
    models = [six_arm(changed_region=i) for i in range(6)]
    subsets = list(itertools.combinations(range(6), 3))
    probes = [unit_energy_subset(6, s) for s in subsets]

    single = np.stack(
        [response_bank(models, [p])[:, 0, :] for p in probes],
        axis=0,
    )

    chosen, trace = greedy_probe_set(single, budget=3)
    chosen_probes = [probes[i] for i in chosen]
    chosen_subsets = [subsets[i] for i in chosen]

    bank = response_bank(models, chosen_probes)
    accuracy = monte_carlo_localization(
        bank,
        noise_std=NOISE_STD,
        trials=TRIALS,
        seed=SEED,
    )

    templates = bank.reshape(6, -1)
    distance_matrix = np.zeros((6, 6), dtype=float)
    for i in range(6):
        for j in range(6):
            distance_matrix[i, j] = np.linalg.norm(
                templates[i] - templates[j]
            )

    result = {
        "gate": "greedy_stimulation_planner",
        "candidate_probes": len(probes),
        "candidate_probe_family": "all positive unit-energy 3-of-6 branch subsets",
        "recording_sites": 1,
        "recording_site": "soma",
        "budget_trials": 3,
        "chosen_subsets": [list(s) for s in chosen_subsets],
        "planner_trace": trace,
        "all_15_hypothesis_pairs_separated_after_step": next(
            (
                row["step"]
                for row in trace
                if row["separated_pairs"] == 15
            ),
            None,
        ),
        "noise_std": NOISE_STD,
        "monte_carlo_trials": TRIALS,
        "localization_accuracy": accuracy,
        "final_min_pairwise_template_distance": float(
            np.min(distance_matrix[np.triu_indices(6, k=1)])
        ),
        "distance_matrix": distance_matrix.tolist(),
        "interpretation": (
            "Without being handed the three-bit code, a deterministic stimulation "
            "planner searches physically positive multi-site probes and chooses "
            "three trials that separate all six hidden branch-change hypotheses "
            "at a single soma sensor. This is the first reusable planning primitive "
            "for the future Operaattori bridge; it still assumes a perfect known "
            "forward model and a finite candidate hypothesis set."
        ),
    }

    assert result["all_15_hypothesis_pairs_separated_after_step"] == 3
    assert result["localization_accuracy"] > 0.985
    assert result["final_min_pairwise_template_distance"] > 0.015

    out = Path("results/gate2_planner.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
