"""Gate 2: the computer chooses a three-trial soma-only stimulation set.

Revision: the planner is now explicitly noise-aware. The original implementation
counted any numerically nonzero template difference as a separated hypothesis
pair; that can prefer sub-noise distinctions. See `gate2a_planner_audit.py`.
"""

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
from planner import greedy_probe_set, noise_aware_pairwise_score

NOISE_STD = 0.003
MAX_PAIR_ERROR = 0.05
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

    chosen, trace = greedy_probe_set(
        single,
        budget=3,
        noise_std=NOISE_STD,
        max_pair_error=MAX_PAIR_ERROR,
    )
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

    final_score = noise_aware_pairwise_score(
        templates,
        noise_std=NOISE_STD,
        max_pair_error=MAX_PAIR_ERROR,
    )

    result = {
        "gate": "noise_aware_greedy_stimulation_planner",
        "candidate_probes": len(probes),
        "candidate_probe_family": "all positive unit-energy 3-of-6 branch subsets",
        "recording_sites": 1,
        "recording_site": "soma",
        "budget_trials": 3,
        "noise_std": NOISE_STD,
        "max_pair_error_for_resolved_pair": MAX_PAIR_ERROR,
        "chosen_subsets": [list(s) for s in chosen_subsets],
        "planner_trace": trace,
        "all_15_hypothesis_pairs_resolved_after_step": next(
            (
                row["step"]
                for row in trace
                if row["resolved_pairs"] == 15
            ),
            None,
        ),
        "monte_carlo_trials": TRIALS,
        "localization_accuracy": accuracy,
        "final_pairwise_score": final_score,
        "final_min_pairwise_template_distance": float(
            np.min(distance_matrix[np.triu_indices(6, k=1)])
        ),
        "distance_matrix": distance_matrix.tolist(),
        "interpretation": (
            "The planner now scores separations relative to the declared soma-noise "
            "level instead of treating every nonzero numerical difference as useful. "
            "In this exactly symmetric six-arm toy, the main job is still combinatorial: "
            "find three positive multi-site probes that assign all six branch-change "
            "hypotheses distinct response codes. Gate 2a audits that limitation before "
            "we claim an electrophysiological design advantage on a real morphology."
        ),
    }

    assert result["all_15_hypothesis_pairs_resolved_after_step"] == 3
    assert result["localization_accuracy"] > 0.985
    assert result["final_pairwise_score"]["resolved_pairs"] == 15
    assert result["final_pairwise_score"]["worst_pair_error"] < 0.01

    out = Path("results/gate2_planner.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
