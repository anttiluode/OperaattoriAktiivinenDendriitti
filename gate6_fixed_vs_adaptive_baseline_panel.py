"""Gate 6: fixed versus adaptive probing with an explicit baseline panel.

This gate responds to the 2026-09-08 independent review. It freezes the
measurement protocol instead of changing several things at once:

* one soma sensor;
* 14 retained samples per waveform for every strategy;
* the same 20 pre-change baseline recordings for every strategy;
* exactly three post-change stimulation trials;
* seven target classes: unchanged + one of six branch changes;
* one joint particle belief over target and nuisance state across all trials;
* after-only stimulation-gain drift;
* a new held-out mismatch profile for every synthetic specimen.

The baseline panel is deliberately expensive and fully counted. It solves the
chronology problem cleanly: after the change, an adaptive policy may choose any
of the 20 already-baselined probes without pretending it can travel back in time
to acquire a new pre-change response.

The adaptive policy is causal but intentionally simple. It chooses the next
unused probe that maximizes a posterior-weighted discriminant ratio:
between-class predicted response variance divided by within-class nuisance
variance plus measurement/model noise. It cannot inspect the hidden label or
the richer evaluation generator.

A negative adaptive result is allowed. The purpose is to compare fairly, not to
make adaptivity win.
"""

from __future__ import annotations

from collections import Counter
import itertools
import json
import math
from pathlib import Path

import numpy as np

from designs import (
    GATE3_MULTI_HYPOTHESIS,
    GATE4_NUISANCE_AWARE,
    GATE5_REDUCED_PAIRED,
)


NOISE_STD = 0.003
TIME_INDEX = np.arange(0, 70, 5, dtype=int)
HETERO_SEED = 123
HETERO_LOG_STD = 0.10
MODEL_FLOOR_STD = 0.0025

PARTICLE_SEED = 20260908
PARTICLES_PER_CLASS = 128

EVAL_SEED = 20261009
EVAL_PER_CLASS = 150

RANDOM_VALID_CODE_SETS = 40
RANDOM_VALID_CODE_SEED = 20261109

# [common leak, split leak, axial scale, pre gain, after-only gain drift]
NUISANCE_LOG_STD = np.array([0.05, 0.04, 0.03, 0.05, 0.05], dtype=float)
TARGET_MAG_RANGE = (0.04, 0.08)
MODEL_FLOOR_VAR = MODEL_FLOOR_STD**2


class PassiveNetwork:
    def __init__(self, A, dt, soma, stimulation_nodes):
        self.A = np.asarray(A, dtype=float)
        self.dt = float(dt)
        self.soma = int(soma)
        self.stimulation_nodes = tuple(int(x) for x in stimulation_nodes)

    def response(self, amplitudes, *, steps=70, pulse_steps=4):
        amplitudes = np.asarray(amplitudes, dtype=float)
        state = np.zeros(self.A.shape[0], dtype=float)
        drive = np.zeros_like(state)
        for amp, node in zip(amplitudes, self.stimulation_nodes):
            drive[node] = float(amp)
        out = np.empty(int(steps), dtype=float)
        for t in range(int(steps)):
            u = drive if t < int(pulse_steps) else np.zeros_like(drive)
            state = self.A @ (state + self.dt * u)
            out[t] = state[self.soma]
        return out


class CapacitiveNetwork:
    def __init__(self, T, U, soma, stimulation_nodes):
        self.T = np.asarray(T, dtype=float)
        self.U = np.asarray(U, dtype=float)
        self.soma = int(soma)
        self.stimulation_nodes = tuple(int(x) for x in stimulation_nodes)

    def response(self, amplitudes, *, steps=70, pulse_steps=4):
        amplitudes = np.asarray(amplitudes, dtype=float)
        state = np.zeros(self.T.shape[0], dtype=float)
        drive = np.zeros_like(state)
        for amp, node in zip(amplitudes, self.stimulation_nodes):
            drive[node] = float(amp)
        out = np.empty(int(steps), dtype=float)
        for t in range(int(steps)):
            u = drive if t < int(pulse_steps) else np.zeros_like(drive)
            state = self.T @ state + self.U @ u
            out[t] = state[self.soma]
        return out


def compile_passive(n, edges, leak, *, stimulation_nodes, soma=0, dt=0.25):
    G = np.zeros((n, n), dtype=float)
    for i, j, g in edges:
        G[i, i] += g
        G[j, j] += g
        G[i, j] -= g
        G[j, i] -= g
    K = G + np.diag(np.asarray(leak, dtype=float))
    A = np.linalg.inv(np.eye(n) + float(dt) * K)
    return PassiveNetwork(A, dt, soma, stimulation_nodes)


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
    Minv = np.linalg.inv(D + K)
    return CapacitiveNetwork(Minv @ D, Minv, soma, stimulation_nodes)


def candidate_probes():
    subsets = list(itertools.combinations(range(6), 3))
    probes = []
    for subset in subsets:
        p = np.zeros(6, dtype=float)
        p[list(subset)] = 1.0 / np.sqrt(3.0)
        probes.append(p)
    return subsets, np.stack(probes, axis=0)


SUBSETS, PROBES = candidate_probes()


def sampled_basis(model):
    basis = np.eye(6, dtype=float)
    return np.stack(
        [
            model.response(basis[b], steps=70, pulse_steps=4)[TIME_INDEX]
            for b in range(6)
        ],
        axis=0,
    )


def pattern_from_basis(basis):
    return PROBES @ np.asarray(basis, dtype=float)


def reduced_model(
    *,
    branch=None,
    magnitude=0.0,
    common_log=0.0,
    split_log=0.0,
    axial_log=0.0,
):
    rng = np.random.default_rng(HETERO_SEED)
    variation = rng.normal(0.0, HETERO_LOG_STD, size=(6, 4))
    n = 13
    edges = []
    leak = np.zeros(n, dtype=float)
    leak[0] = 0.35 * np.exp(float(common_log))
    stimulation_nodes = []
    split_sign = np.array([1, 1, 1, -1, -1, -1], dtype=float)

    for b in range(6):
        prox = 1 + 2 * b
        dist = 2 + 2 * b
        stimulation_nodes.append(dist)
        g_soma = 0.75 * np.exp(variation[b, 0] + axial_log)
        g_dist = 0.55 * np.exp(variation[b, 1] + axial_log)
        leak[prox] = 0.22 * np.exp(
            variation[b, 2] + common_log + 0.5 * split_log * split_sign[b]
        )
        leak[dist] = 0.18 * np.exp(
            variation[b, 3] + common_log + split_log * split_sign[b]
        )
        if branch is not None and int(branch) == b:
            leak[dist] += float(magnitude)
        edges.extend([(0, prox, float(g_soma)), (prox, dist, float(g_dist))])

    return compile_passive(
        n,
        edges,
        leak,
        stimulation_nodes=tuple(stimulation_nodes),
    )


def rich_model(
    profile,
    *,
    branch=None,
    magnitude=0.0,
    common_log=0.0,
    split_log=0.0,
    axial_log=0.0,
):
    """Three-compartment-per-arm held-out generator."""
    rng = np.random.default_rng(HETERO_SEED)
    variation = rng.normal(0.0, HETERO_LOG_STD, size=(6, 4))
    cap_variation = np.asarray(profile["cap_variation"], dtype=float)
    offsets = np.asarray(profile["distal_leak_offsets"], dtype=float)

    n = 19
    edges = []
    leak = np.zeros(n, dtype=float)
    capacitance = np.zeros(n, dtype=float)
    leak[0] = 0.35 * np.exp(float(common_log))
    capacitance[0] = 1.12
    stimulation_nodes = []
    split_sign = np.array([1, 1, 1, -1, -1, -1], dtype=float)

    for b in range(6):
        prox = 1 + 3 * b
        mid = 2 + 3 * b
        dist = 3 + 3 * b
        stimulation_nodes.append(dist)
        g_soma = 0.75 * np.exp(variation[b, 0] + axial_log)
        g_dist_reduced = 0.55 * np.exp(variation[b, 1] + axial_log)
        edges.extend(
            [
                (0, prox, float(g_soma)),
                (prox, mid, float(2.0 * g_dist_reduced)),
                (mid, dist, float(2.0 * g_dist_reduced)),
            ]
        )
        leak[prox] = 0.22 * np.exp(
            variation[b, 2] + common_log + 0.5 * split_log * split_sign[b]
        )
        distal_total = 0.18 * np.exp(
            variation[b, 3] + common_log + split_log * split_sign[b]
        )
        leak[mid] = 0.48 * distal_total
        leak[dist] = 0.52 * distal_total + offsets[b]
        if branch is not None and int(branch) == b:
            leak[dist] += float(magnitude)
        capacitance[prox] = np.exp(cap_variation[b, 0])
        capacitance[mid] = 0.48 * np.exp(cap_variation[b, 1])
        capacitance[dist] = 0.52 * np.exp(cap_variation[b, 2])

    if np.min(leak) <= 0.0:
        raise RuntimeError("held-out mismatch produced non-positive leak")

    return compile_capacitive(
        n,
        edges,
        leak,
        capacitance,
        stimulation_nodes=tuple(stimulation_nodes),
    )


def make_particles():
    rng = np.random.default_rng(PARTICLE_SEED)
    return np.column_stack(
        [
            rng.normal(0.0, NUISANCE_LOG_STD[0], PARTICLES_PER_CLASS),
            rng.normal(0.0, NUISANCE_LOG_STD[1], PARTICLES_PER_CLASS),
            rng.normal(0.0, NUISANCE_LOG_STD[2], PARTICLES_PER_CLASS),
            rng.normal(0.0, NUISANCE_LOG_STD[3], PARTICLES_PER_CLASS),
            rng.normal(0.0, NUISANCE_LOG_STD[4], PARTICLES_PER_CLASS),
            rng.uniform(TARGET_MAG_RANGE[0], TARGET_MAG_RANGE[1], PARTICLES_PER_CLASS),
        ]
    )


PARTICLE_PARAMS = make_particles()


def build_reduced_delta_library():
    """Return [class(7), particle, probe(20), time(14)]."""
    out = np.zeros(
        (7, PARTICLES_PER_CLASS, len(PROBES), len(TIME_INDEX)), dtype=float
    )
    for q, row in enumerate(PARTICLE_PARAMS):
        common, split, axial, _, _, magnitude = row
        base = reduced_model(common_log=common, split_log=split, axial_log=axial)
        base_patterns = pattern_from_basis(sampled_basis(base))
        for b in range(6):
            changed = reduced_model(
                branch=b,
                magnitude=magnitude,
                common_log=common,
                split_log=split,
                axial_log=axial,
            )
            out[b + 1, q] = pattern_from_basis(sampled_basis(changed)) - base_patterns
    return out


REDUCED_DELTA = build_reduced_delta_library()


def new_mismatch_profile(rng):
    cap_variation = rng.normal(0.0, 0.10, size=(6, 3))
    offsets = rng.normal(0.0, 0.04, size=6)
    offsets -= np.mean(offsets)
    return {"cap_variation": cap_variation, "distal_leak_offsets": offsets}


def generate_specimen(target_class, rng):
    common = rng.normal(0.0, NUISANCE_LOG_STD[0])
    split = rng.normal(0.0, NUISANCE_LOG_STD[1])
    axial = rng.normal(0.0, NUISANCE_LOG_STD[2])
    pre_gain_log = rng.normal(0.0, NUISANCE_LOG_STD[3])
    post_drift_log = rng.normal(0.0, NUISANCE_LOG_STD[4])
    magnitude = 0.0 if int(target_class) == 0 else rng.uniform(*TARGET_MAG_RANGE)

    profile = new_mismatch_profile(rng)
    for _ in range(10):
        try:
            baseline = rich_model(
                profile,
                common_log=common,
                split_log=split,
                axial_log=axial,
            )
            break
        except RuntimeError:
            profile["distal_leak_offsets"] *= 0.75
    else:
        raise RuntimeError("could not make a positive-leak held-out specimen")

    branch = None if int(target_class) == 0 else int(target_class) - 1
    changed = rich_model(
        profile,
        branch=branch,
        magnitude=magnitude,
        common_log=common,
        split_log=split,
        axial_log=axial,
    )
    baseline_patterns = pattern_from_basis(sampled_basis(baseline))
    changed_patterns = pattern_from_basis(sampled_basis(changed))
    g0 = np.exp(pre_gain_log)
    g1 = np.exp(pre_gain_log + post_drift_log)
    baseline_clean = g0 * baseline_patterns
    post_clean = g1 * changed_patterns
    baseline_observed = baseline_clean + rng.normal(0.0, NOISE_STD, baseline_clean.shape)
    post_observed = post_clean + rng.normal(0.0, NOISE_STD, post_clean.shape)
    return {
        "target_class": int(target_class),
        "baseline_observed": baseline_observed,
        "difference_observed": post_observed - baseline_observed,
    }


def particle_prediction(baseline_observed, probe_index):
    _, _, _, pre_gain_log, drift_log, _ = PARTICLE_PARAMS.T
    g1 = np.exp(pre_gain_log + drift_log)
    drift_multiplier = np.expm1(drift_log)
    mu = (
        g1[None, :, None] * REDUCED_DELTA[:, :, probe_index, :]
        + drift_multiplier[None, :, None]
        * baseline_observed[probe_index][None, None, :]
    )
    variance = NOISE_STD**2 * (1.0 + np.exp(2.0 * drift_log)) + MODEL_FLOOR_VAR
    return mu, variance


def normalized_weights(log_weights):
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    return weights


def class_posterior(log_weights):
    weights = normalized_weights(log_weights)
    return np.sum(weights, axis=1), weights


def update_belief(log_weights, specimen, probe_index):
    mu, variance = particle_prediction(specimen["baseline_observed"], probe_index)
    observed = specimen["difference_observed"][probe_index]
    residual = observed[None, None, :] - mu
    ll = (
        -0.5 * np.sum(residual * residual / variance[None, :, None], axis=2)
        - 0.5 * len(TIME_INDEX) * np.log(variance)[None, :]
    )
    out = log_weights + ll
    out -= np.max(out)
    return out


def adaptive_probe_score(log_weights, specimen, probe_index):
    class_prob, weights = class_posterior(log_weights)
    mu, variance = particle_prediction(specimen["baseline_observed"], probe_index)
    class_means = []
    within = 0.0
    for h in range(7):
        class_weights = weights[h]
        ph = float(np.sum(class_weights))
        if ph < 1e-15:
            class_means.append(np.zeros(len(TIME_INDEX), dtype=float))
            continue
        conditional = class_weights / ph
        mean_h = np.sum(conditional[:, None] * mu[h], axis=0)
        class_means.append(mean_h)
        within += ph * float(
            np.sum(conditional[:, None] * (mu[h] - mean_h) ** 2)
        )
    class_means = np.stack(class_means, axis=0)
    overall = np.sum(class_prob[:, None] * class_means, axis=0)
    between = float(np.sum(class_prob[:, None] * (class_means - overall) ** 2))
    noise = float(np.sum(weights * variance[None, :])) * len(TIME_INDEX)
    return between / (within + noise + 1e-30)


def run_policy(specimen, *, fixed_design=None):
    log_weights = np.full(
        (7, PARTICLES_PER_CLASS),
        -math.log(7 * PARTICLES_PER_CLASS),
        dtype=float,
    )
    used = []
    for step in range(3):
        if fixed_design is not None:
            probe_index = int(fixed_design[step])
        else:
            candidates = [p for p in range(len(PROBES)) if p not in used]
            scores = [adaptive_probe_score(log_weights, specimen, p) for p in candidates]
            probe_index = int(candidates[int(np.argmax(scores))])
        used.append(probe_index)
        log_weights = update_belief(log_weights, specimen, probe_index)
    posterior, _ = class_posterior(log_weights)
    return posterior, used


def metrics(truth, posterior):
    truth = np.asarray(truth, dtype=int)
    posterior = np.asarray(posterior, dtype=float)
    predicted = np.argmax(posterior, axis=1)
    correct = predicted == truth
    truth_prob = posterior[np.arange(len(truth)), truth]
    onehot = np.eye(7)[truth]
    confidence = np.max(posterior, axis=1)

    ece = 0.0
    for lo in np.linspace(0.0, 1.0, 11)[:-1]:
        hi = lo + 0.1
        mask = (confidence >= lo) & (confidence < (hi if hi < 1.0 else 1.000001))
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(
                float(np.mean(correct[mask])) - float(np.mean(confidence[mask]))
            )

    abstention = {}
    for threshold in (0.30, 0.40, 0.50, 0.60):
        mask = confidence >= threshold
        abstention[str(threshold)] = {
            "coverage": float(np.mean(mask)),
            "accuracy_when_answered": float(np.mean(correct[mask])) if np.any(mask) else None,
        }

    order = np.argsort(posterior, axis=1)[:, ::-1]
    top2 = np.mean([truth[i] in order[i, :2] for i in range(len(truth))])
    unchanged = truth == 0
    return {
        "accuracy": float(np.mean(correct)),
        "top2_accuracy": float(top2),
        "mean_truth_posterior": float(np.mean(truth_prob)),
        "brier": float(np.mean(np.sum((posterior - onehot) ** 2, axis=1))),
        "log_loss": float(-np.mean(np.log(np.clip(truth_prob, 1e-12, 1.0)))),
        "ece10": float(ece),
        "unchanged_recall": float(np.mean(predicted[unchanged] == 0)),
        "abstention": abstention,
    }


def unique_membership_codes(design):
    matrix = np.stack([PROBES[int(i)] != 0.0 for i in design], axis=0)
    return len({tuple(matrix[:, branch].tolist()) for branch in range(6)})


def summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def evaluate_fixed(specimens, design):
    truth = []
    posterior = []
    for specimen in specimens:
        post, _ = run_policy(specimen, fixed_design=[int(x) for x in design])
        truth.append(specimen["target_class"])
        posterior.append(post)
    return metrics(truth, posterior)


def main():
    rng = np.random.default_rng(EVAL_SEED)
    specimens = []
    for target_class in range(7):
        for _ in range(EVAL_PER_CLASS):
            specimens.append(generate_specimen(target_class, rng))

    fixed_designs = {
        "gate4_nuisance": [int(x) for x in GATE4_NUISANCE_AWARE],
        "gate5_paired": [int(x) for x in GATE5_REDUCED_PAIRED],
        "gate3_actual": [int(x) for x in GATE3_MULTI_HYPOTHESIS],
    }
    result_metrics = {
        name: evaluate_fixed(specimens, design)
        for name, design in fixed_designs.items()
    }

    adaptive_truth = []
    adaptive_posterior = []
    trajectories = []
    for specimen in specimens:
        post, used = run_policy(specimen)
        adaptive_truth.append(specimen["target_class"])
        adaptive_posterior.append(post)
        trajectories.append(tuple(used))
    result_metrics["adaptive"] = metrics(adaptive_truth, adaptive_posterior)

    valid_designs = [
        list(t)
        for t in itertools.combinations(range(len(PROBES)), 3)
        if unique_membership_codes(t) == 6
    ]
    random_rng = np.random.default_rng(RANDOM_VALID_CODE_SEED)
    chosen = random_rng.choice(
        len(valid_designs), size=RANDOM_VALID_CODE_SETS, replace=False
    )
    random_accuracy = np.asarray(
        [
            evaluate_fixed(specimens, valid_designs[int(i)])["accuracy"]
            for i in chosen
        ],
        dtype=float,
    )

    trajectory_counts = Counter(trajectories)
    first_probe_counts = Counter(x[0] for x in trajectories)

    result = {
        "gate": "baseline_panel_fixed_vs_adaptive",
        "classes": ["unchanged"] + [f"branch_{i}_leak_change" for i in range(6)],
        "chance_accuracy": 1.0 / 7.0,
        "measurement_protocol": {
            "candidate_post_probes": len(PROBES),
            "baseline_panel_pre_change_recordings": len(PROBES),
            "post_change_recordings_per_strategy": 3,
            "total_recordings_if_baseline_counted": len(PROBES) + 3,
            "baseline_status": (
                "identical full candidate-probe panel supplied to every strategy before the hidden change"
            ),
            "waveform_samples_per_recording": int(len(TIME_INDEX)),
            "sensor": "soma only",
            "noise_std_each_recording": NOISE_STD,
        },
        "truth_generator": {
            "model": "three-compartment-per-arm passive generator",
            "held_out_mismatch": (
                "new capacitance heterogeneity and static branch leak offsets for every specimen"
            ),
            "target_magnitude": "uniform 0.04..0.08 additive distal leak when changed",
            "shared_nuisance_log_std": {
                "common_leak": float(NUISANCE_LOG_STD[0]),
                "split_leak_bias": float(NUISANCE_LOG_STD[1]),
                "axial_scale": float(NUISANCE_LOG_STD[2]),
                "pre_stimulation_gain": float(NUISANCE_LOG_STD[3]),
                "after_only_gain_drift": float(NUISANCE_LOG_STD[4]),
            },
            "specimens_per_class": EVAL_PER_CLASS,
            "total_specimens": len(specimens),
            "evaluation_seed": EVAL_SEED,
        },
        "inference": {
            "reduced_model": "two-compartment-per-arm model",
            "joint_particles_per_class": PARTICLES_PER_CLASS,
            "joint_classes": 7,
            "joint_particle_states": 7 * PARTICLES_PER_CLASS,
            "shared_state_across_trials": [
                "target class",
                "target magnitude",
                "common leak",
                "split bias",
                "axial scale",
                "pre gain",
                "post-only gain drift",
            ],
            "model_floor_std": MODEL_FLOOR_STD,
            "baseline_use": (
                "stored measured baseline for the selected probe enters the gain-drift prediction; no post-change query asks for a new pre-change measurement"
            ),
            "adaptive_rule": (
                "causal posterior-weighted between-class / within-nuisance discriminant score; no hidden label or rich-generator parameters"
            ),
        },
        "fixed_designs": fixed_designs,
        "metrics": result_metrics,
        "valid_code_random_fixed": {
            "available_valid_code_designs": len(valid_designs),
            "sampled_designs": RANDOM_VALID_CODE_SETS,
            "sampling_seed": RANDOM_VALID_CODE_SEED,
            "condition": (
                "all six branches have distinct three-bit stimulation membership codes"
            ),
            "accuracy": summary(random_accuracy),
        },
        "adaptive_probe_behavior": {
            "distinct_three_probe_trajectories": len(trajectory_counts),
            "first_probe_counts": {
                str(k): int(v) for k, v in sorted(first_probe_counts.items())
            },
            "most_common_trajectories": [
                {"probe_indices": list(k), "count": int(v)}
                for k, v in trajectory_counts.most_common(8)
            ],
        },
        "comparisons": {
            "adaptive_minus_gate4_fixed_accuracy": float(
                result_metrics["adaptive"]["accuracy"]
                - result_metrics["gate4_nuisance"]["accuracy"]
            ),
            "gate4_fixed_minus_code_valid_random_mean_accuracy": float(
                result_metrics["gate4_nuisance"]["accuracy"] - np.mean(random_accuracy)
            ),
            "adaptive_minus_code_valid_random_mean_accuracy": float(
                result_metrics["adaptive"]["accuracy"] - np.mean(random_accuracy)
            ),
        },
        "interpretation": (
            "With an explicit pre-change baseline panel, an unchanged class, after-only gain drift, unknown change magnitude, shared nuisance state, and specimen-specific held-out model mismatch, the strong Gate-4 fixed set remains informative and beats the mean of sampled code-valid random sets. The first causal adaptive heuristic does not beat that fixed set under the same post-change budget. This is a useful negative result: adaptive superiority is not required before transferring the fixed protocol to a real Operaattori morphology. The baseline is an explicit external memory resource and its 20-recording acquisition cost is reported rather than hidden."
        ),
    }

    assert result["measurement_protocol"]["baseline_panel_pre_change_recordings"] == 20
    assert result_metrics["gate4_nuisance"]["accuracy"] > (
        result["valid_code_random_fixed"]["accuracy"]["mean"] + 0.02
    )
    assert result_metrics["adaptive"]["accuracy"] <= (
        result_metrics["gate4_nuisance"]["accuracy"] + 0.01
    )

    out = Path("results/gate6_fixed_vs_adaptive_baseline_panel.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
