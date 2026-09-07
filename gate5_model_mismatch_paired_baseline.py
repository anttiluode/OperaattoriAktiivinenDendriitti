"""Gate 5: model mismatch and the value/cost of paired baselines.

Gate 4 still used the same reduced model family for design and evaluation. This
gate deliberately breaks that shared assumption.

Planner / decoder model:
    the Gate-4 two-compartment-per-arm passive network.

Synthetic "experimental" generator:
    * three compartments per arm (the distal cable is split),
    * heterogeneous capacitances unknown to the reduced model,
    * a fixed, unmodelled branch-specific distal-leak offset,
    * the same continuous specimen-level nuisance variables as Gate 4.

The hidden intervention is still an added distal leak on one branch.

Two measurement protocols are compared:

ABSOLUTE
    classify from the post-change soma traces directly.

PAIRED
    measure the same stimulation before and after the intervention and classify
    from the difference trace. Baseline and post measurements each receive
    independent sigma noise, so the difference is charged sqrt(2)*sigma noise.

The paired protocol is important because static model error shared by before and
after measurements can cancel even when the reduced forward model is wrong.
It is not free: it costs an extra recording and increases noise if the baseline
is not averaged.

The gate also checks whether reduced-model stimulation rankings survive the
richer generator. They do not perfectly: paired differencing improves robustness,
but model mismatch can still reorder which stimulation set is best.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from active_dendrite import compile_passive, unit_energy_subset
from gate4_nuisance_wall import (
    EVAL_NOISE_SEED,
    HETERO_LOG_STD,
    HETERO_SEED,
    NOISE_STD,
    NUISANCE_STD,
    TARGET_DELTA_LEAK,
    TIME_INDEX,
    candidate_probes,
    heterogeneous_model,
    nuisance_draws,
    response_library,
)
from planner import gaussian_localization_accuracy_estimate


TRAIN_SAMPLES = 160
TRAIN_SEED = 700
EVAL_SAMPLES = 800
EVAL_SEED = 1702
MISMATCH_SEED = 992
CAPACITANCE_SEED = 991
MISMATCH_DISTAL_LEAK_STD = 0.04
PAIRED_NOISE_STD = float(np.sqrt(2.0) * NOISE_STD)
DESIGN_MC = 5000
DESIGN_SEED = 212
RANDOM_DESIGNS = 200
RANDOM_SEED = 288


class CapacitiveNetwork:
    def __init__(self, T, U, soma, stimulation_nodes):
        self.T = np.asarray(T, dtype=float)
        self.U = np.asarray(U, dtype=float)
        self.soma = int(soma)
        self.stimulation_nodes = tuple(int(x) for x in stimulation_nodes)

    def response(self, amplitudes, *, steps=70, pulse_steps=4):
        amplitudes = np.asarray(amplitudes, dtype=float)
        state = np.zeros(self.T.shape[0], dtype=float)
        out = np.zeros(int(steps), dtype=float)
        drive = np.zeros_like(state)
        for amp, node in zip(amplitudes, self.stimulation_nodes):
            drive[node] = float(amp)
        for t in range(int(steps)):
            u = drive if t < int(pulse_steps) else np.zeros_like(drive)
            state = self.T @ state + self.U @ u
            out[t] = state[self.soma]
        return out


def compile_capacitive(
    n,
    edges,
    leak,
    capacitance,
    *,
    stimulation_nodes,
    soma=0,
    dt=0.25,
):
    G = np.zeros((n, n), dtype=float)
    for i, j, g in edges:
        G[i, i] += g
        G[j, j] += g
        G[i, j] -= g
        G[j, i] -= g
    K = G + np.diag(np.asarray(leak, dtype=float))
    D = np.diag(np.asarray(capacitance, dtype=float) / float(dt))
    M = D + K
    Minv = np.linalg.inv(M)
    return CapacitiveNetwork(
        Minv @ D,
        Minv,
        soma,
        stimulation_nodes,
    )


def fixed_mismatch_offsets():
    rng = np.random.default_rng(MISMATCH_SEED)
    offsets = rng.normal(0.0, MISMATCH_DISTAL_LEAK_STD, size=6)
    offsets -= np.mean(offsets)
    return offsets


def rich_ground_truth(
    *,
    hidden_branch=None,
    common_log=0.0,
    split_log=0.0,
    axial_log=0.0,
):
    """Three-compartment-per-arm ground truth unknown to the planner."""
    rng = np.random.default_rng(HETERO_SEED)
    variation = rng.normal(0.0, HETERO_LOG_STD, size=(6, 4))
    cap_rng = np.random.default_rng(CAPACITANCE_SEED)
    cap_variation = cap_rng.normal(0.0, 0.10, size=(6, 3))
    mismatch = fixed_mismatch_offsets()

    n = 19
    edges = []
    leak = np.zeros(n, dtype=float)
    capacitance = np.zeros(n, dtype=float)
    leak[0] = 0.35 * np.exp(float(common_log))
    capacitance[0] = 1.12
    stimulation_nodes = []
    split_sign = np.array([1, 1, 1, -1, -1, -1], dtype=float)

    for branch in range(6):
        prox = 1 + 3 * branch
        mid = 2 + 3 * branch
        dist = 3 + 3 * branch
        stimulation_nodes.append(dist)

        g_soma = 0.75 * np.exp(variation[branch, 0] + axial_log)
        g_dist_reduced = 0.55 * np.exp(variation[branch, 1] + axial_log)

        # Split the reduced distal axial resistance into two equal half-segments.
        edges.extend(
            [
                (0, prox, float(g_soma)),
                (prox, mid, float(2.0 * g_dist_reduced)),
                (mid, dist, float(2.0 * g_dist_reduced)),
            ]
        )

        leak[prox] = 0.22 * np.exp(
            variation[branch, 2]
            + common_log
            + 0.5 * split_log * split_sign[branch]
        )
        distal_total = 0.18 * np.exp(
            variation[branch, 3]
            + common_log
            + split_log * split_sign[branch]
        )
        leak[mid] = 0.48 * distal_total
        leak[dist] = 0.52 * distal_total + mismatch[branch]

        capacitance[prox] = np.exp(cap_variation[branch, 0])
        capacitance[mid] = 0.48 * np.exp(cap_variation[branch, 1])
        capacitance[dist] = 0.52 * np.exp(cap_variation[branch, 2])

    if hidden_branch is not None:
        leak[3 + 3 * int(hidden_branch)] += TARGET_DELTA_LEAK

    if np.min(leak) <= 0.0:
        raise RuntimeError("mismatch attacker produced non-positive leak")

    return compile_capacitive(
        n,
        edges,
        leak,
        capacitance,
        stimulation_nodes=tuple(stimulation_nodes),
    )


def sampled_trace(model, probe):
    return model.response(probe, steps=70, pulse_steps=4)[TIME_INDEX]


def reduced_delta_library(nuisance_values, gain, probes):
    S = len(nuisance_values)
    out = np.empty((len(probes), 6, S, len(TIME_INDEX)), dtype=float)
    basis = np.eye(6, dtype=float)

    for si, (common_log, split_log, axial_log) in enumerate(nuisance_values):
        baseline = heterogeneous_model(
            common_log=float(common_log),
            split_log=float(split_log),
            axial_log=float(axial_log),
        )
        base_single = np.stack(
            [sampled_trace(baseline, basis[b]) for b in range(6)], axis=0
        )

        for hidden in range(6):
            delta = np.zeros(6, dtype=float)
            delta[hidden] = TARGET_DELTA_LEAK
            changed = heterogeneous_model(
                branch_delta=delta,
                common_log=float(common_log),
                split_log=float(split_log),
                axial_log=float(axial_log),
            )
            changed_single = np.stack(
                [sampled_trace(changed, basis[b]) for b in range(6)], axis=0
            )
            difference = changed_single - base_single
            for pi, probe in enumerate(probes):
                out[pi, hidden, si] = float(gain[si]) * (probe @ difference)
    return out


def rich_absolute_and_delta_library(nuisance_values, gain, probes):
    S = len(nuisance_values)
    absolute = np.empty((len(probes), 6, S, len(TIME_INDEX)), dtype=float)
    delta = np.empty_like(absolute)
    basis = np.eye(6, dtype=float)

    for si, (common_log, split_log, axial_log) in enumerate(nuisance_values):
        baseline = rich_ground_truth(
            common_log=float(common_log),
            split_log=float(split_log),
            axial_log=float(axial_log),
        )
        base_single = np.stack(
            [sampled_trace(baseline, basis[b]) for b in range(6)], axis=0
        )

        for hidden in range(6):
            changed = rich_ground_truth(
                hidden_branch=hidden,
                common_log=float(common_log),
                split_log=float(split_log),
                axial_log=float(axial_log),
            )
            changed_single = np.stack(
                [sampled_trace(changed, basis[b]) for b in range(6)], axis=0
            )
            difference = changed_single - base_single
            for pi, probe in enumerate(probes):
                absolute[pi, hidden, si] = float(gain[si]) * (
                    probe @ changed_single
                )
                delta[pi, hidden, si] = float(gain[si]) * (
                    probe @ difference
                )
    return absolute, delta


def fit_lda(library, selected, *, noise_std):
    X = np.concatenate([library[p] for p in selected], axis=2)
    H, S, F = X.shape
    means = np.mean(X, axis=1)
    centered = (X - means[:, None, :]).reshape(H * S, F)
    covariance = (centered.T @ centered) / float(H * S - H)
    covariance += float(noise_std) ** 2 * np.eye(F)
    L = np.linalg.cholesky(covariance + 1e-12 * np.eye(F))
    whitened_means = np.linalg.solve(L, means.T).T
    return L, whitened_means


def evaluate(train_library, eval_library, selected, *, noise_std, seed=8888):
    L, means = fit_lda(train_library, selected, noise_std=noise_std)
    X = np.concatenate([eval_library[p] for p in selected], axis=2)
    H, S, F = X.shape
    rng = np.random.default_rng(seed)
    Y = X + rng.normal(0.0, float(noise_std), size=X.shape)
    Yw = np.linalg.solve(L, Y.reshape(H * S, F).T).T.reshape(H, S, F)
    distance = np.sum(
        (Yw[:, :, None, :] - means[None, None, :, :]) ** 2,
        axis=3,
    )
    predicted = np.argmin(distance, axis=2)
    return float(np.mean(predicted == np.arange(H)[:, None]))


def reduced_paired_score(library, selected):
    _, means = fit_lda(library, selected, noise_std=PAIRED_NOISE_STD)
    return gaussian_localization_accuracy_estimate(
        means,
        noise_std=1.0,
        samples_per_hypothesis=DESIGN_MC,
        seed=DESIGN_SEED + len(selected),
    )


def stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main():
    subsets, probes = candidate_probes()
    triples = list(itertools.combinations(range(len(probes)), 3))

    train_values, train_gain = nuisance_draws(TRAIN_SAMPLES, TRAIN_SEED)
    eval_values, eval_gain = nuisance_draws(EVAL_SAMPLES, EVAL_SEED)

    reduced_absolute = response_library(train_values, train_gain, probes)
    reduced_delta = reduced_delta_library(train_values, train_gain, probes)
    rich_absolute, rich_delta = rich_absolute_and_delta_library(
        eval_values, eval_gain, probes
    )

    # Gate-4 nuisance-aware set and two older attackers.
    gate4_design = [10, 2, 6]
    gate3_design = [18, 10, 7]
    noise_only_design = [18, 0, 13]

    # Design a paired set using only the reduced model. Global exhaustive search
    # is cheap here (1140 sets) and prevents a greedy-search artifact.
    predicted_scores = np.asarray(
        [reduced_paired_score(reduced_delta, list(t)) for t in triples],
        dtype=float,
    )
    paired_index = int(np.argmax(predicted_scores))
    paired_design = list(triples[paired_index])

    named = {
        "gate4_nuisance_aware": gate4_design,
        "gate3_exact_model": gate3_design,
        "noise_only_sensitivity": noise_only_design,
        "reduced_model_paired_design": paired_design,
    }

    absolute_accuracy = {
        name: evaluate(
            reduced_absolute,
            rich_absolute,
            design,
            noise_std=NOISE_STD,
        )
        for name, design in named.items()
    }
    paired_accuracy = {
        name: evaluate(
            reduced_delta,
            rich_delta,
            design,
            noise_std=PAIRED_NOISE_STD,
        )
        for name, design in named.items()
    }

    rng = np.random.default_rng(RANDOM_SEED)
    random_triples = [
        tuple(sorted(rng.choice(len(probes), size=3, replace=False)))
        for _ in range(RANDOM_DESIGNS)
    ]
    random_absolute = np.asarray(
        [
            evaluate(
                reduced_absolute,
                rich_absolute,
                list(t),
                noise_std=NOISE_STD,
            )
            for t in random_triples
        ],
        dtype=float,
    )
    random_paired = np.asarray(
        [
            evaluate(
                reduced_delta,
                rich_delta,
                list(t),
                noise_std=PAIRED_NOISE_STD,
            )
            for t in random_triples
        ],
        dtype=float,
    )

    exhaustive_paired = np.asarray(
        [
            evaluate(
                reduced_delta,
                rich_delta,
                list(t),
                noise_std=PAIRED_NOISE_STD,
            )
            for t in triples
        ],
        dtype=float,
    )
    paired_value = paired_accuracy["reduced_model_paired_design"]
    best_index = int(np.argmax(exhaustive_paired))

    result = {
        "gate": "model_mismatch_and_paired_baseline",
        "planner_model": "Gate-4 two-compartment-per-arm passive model",
        "ground_truth_mismatch": {
            "three_compartments_per_arm": True,
            "heterogeneous_unmodelled_capacitance_log_std": 0.10,
            "fixed_branch_specific_distal_leak_offsets": (
                fixed_mismatch_offsets().tolist()
            ),
            "distal_leak_offset_draw_std": MISMATCH_DISTAL_LEAK_STD,
        },
        "hidden_change": {
            "type": "one distal branch additive passive leak",
            "delta": TARGET_DELTA_LEAK,
        },
        "recording_site": "soma only",
        "stimulation_budget": 3,
        "paired_measurement_cost": (
            "each chosen stimulation is recorded before and after; independent "
            "baseline/post noise makes difference-noise sqrt(2)*sigma"
        ),
        "noise_std": NOISE_STD,
        "paired_difference_noise_std": PAIRED_NOISE_STD,
        "reduced_model_paired_design": {
            "probe_indices": paired_design,
            "subsets": [list(subsets[i]) for i in paired_design],
            "predicted_reduced_model_accuracy": float(
                predicted_scores[paired_index]
            ),
        },
        "rich_ground_truth_accuracy": {
            "absolute_post_change": absolute_accuracy,
            "paired_before_after_difference": paired_accuracy,
        },
        "random_200_baseline": {
            "absolute": stats(random_absolute),
            "paired": stats(random_paired),
        },
        "exhaustive_1140_paired_audit": {
            **stats(exhaustive_paired),
            "reduced_model_paired_design_rank": int(
                1 + np.sum(exhaustive_paired > paired_value)
            ),
            "reduced_model_paired_design_percentile": float(
                np.mean(exhaustive_paired <= paired_value)
            ),
            "best_probe_indices": list(triples[best_index]),
            "best_subsets": [list(subsets[i]) for i in triples[best_index]],
        },
        "interpretation": (
            "Static branch-specific model error can badly corrupt absolute-trace "
            "localization even when the planner handles the Gate-4 nuisance variables. "
            "Paired before/after differencing cancels much of that static mismatch and "
            "recovers localization despite paying sqrt(2) measurement noise. The "
            "reduced-model paired design lands in the top few percent of all equal-budget "
            "sets on the richer generator, but it is not the true optimum and Gate-4's "
            "set slightly outruns it here. Thus baseline differencing is a genuine "
            "robustness mechanism, while stimulation ranking remains model-sensitive."
        ),
        "stopping_line": (
            "Do not call this solved tomography. The mismatch is synthetic and static, "
            "and the paired protocol assumes the hidden intervention occurs between a "
            "baseline and post measurement of the same specimen. The next step is "
            "sequential/adaptive experiment selection and then the real Operaattori "
            "morphology."
        ),
    }

    assert paired_accuracy["gate4_nuisance_aware"] > absolute_accuracy[
        "gate4_nuisance_aware"
    ] + 0.03
    assert paired_value > random_paired.mean() + 0.02
    assert result["exhaustive_1140_paired_audit"][
        "reduced_model_paired_design_percentile"
    ] > 0.94

    out = Path("results/gate5_model_mismatch_paired.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
