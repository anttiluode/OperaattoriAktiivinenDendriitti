"""Gate 4: nuisance wall and sensitivity-subspace rescue.

Gate 3 gave the planner the exact six candidate forward models. This gate adds
four specimen-level nuisance directions that can imitate part of the hidden
branch change:

  * common membrane-leak scale,
  * a known left/right leak-bias mode,
  * global axial-conductance scale,
  * global stimulation-gain calibration.

The hidden target remains: one of six distal branches receives an additive leak
change. We still allow only three positive 3-of-6 stimulation trials and one
soma recording site.

Two local sensitivity planners are compared:

  noise_only:
      optimize branch-change sensitivities assuming only measurement noise;

  nuisance_aware:
      include the nuisance Jacobian in the predicted covariance

          Sigma_p = sigma^2 I + N_p Lambda N_p^T

      and choose stimulation whose branch signatures remain separated after
      whitening by that covariance.

Evaluation is deliberately harder than the design linearization. Continuous
nuisance values are drawn, the full passive network is recompiled, and a
nuisance-trained common-covariance classifier is tested on independent draws.
All methods get the same decoder, sensor, noise, stimulation family and budget.

This is still a synthetic oracle benchmark. The nuisance distribution and the
forward family are known to the experiment designer. Model mismatch comes next.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from active_dendrite import compile_passive, unit_energy_subset
from planner import gaussian_localization_accuracy_estimate


NOISE_STD = 0.003
TARGET_DELTA_LEAK = 0.06
HETERO_SEED = 123
HETERO_LOG_STD = 0.10
TIME_INDEX = np.arange(0, 70, 5, dtype=int)

# Priors are log-standard-deviations for the four nuisance directions.
NUISANCE_STD = np.array([0.05, 0.04, 0.03, 0.05], dtype=float)
# [common leak, split leak, axial scale, stimulation gain]

TRAIN_NUISANCE_SAMPLES = 160
EVAL_NUISANCE_SAMPLES = 1200
TRAIN_SEED = 700
EVAL_SEED = 701
EVAL_NOISE_SEED = 7777
RANDOM_DESIGNS = 200
RANDOM_SEED = 88

FD_EPS = 1e-4
DESIGN_MC = 12000
DESIGN_SEED = 333


def heterogeneous_model(
    *,
    branch_delta: np.ndarray | None = None,
    common_log: float = 0.0,
    split_log: float = 0.0,
    axial_log: float = 0.0,
):
    """Known heterogeneous six-arm passive morphology with nuisance controls."""
    rng = np.random.default_rng(HETERO_SEED)
    variation = rng.normal(0.0, HETERO_LOG_STD, size=(6, 4))

    n = 13
    edges: list[tuple[int, int, float]] = []
    leak = np.zeros(n, dtype=float)
    leak[0] = 0.35 * np.exp(float(common_log))
    stimulation_nodes = []
    split_sign = np.array([1, 1, 1, -1, -1, -1], dtype=float)

    if branch_delta is None:
        branch_delta = np.zeros(6, dtype=float)
    branch_delta = np.asarray(branch_delta, dtype=float)

    for branch in range(6):
        prox = 1 + 2 * branch
        dist = 2 + 2 * branch
        stimulation_nodes.append(dist)

        g_soma = 0.75 * np.exp(variation[branch, 0] + axial_log)
        g_dist = 0.55 * np.exp(variation[branch, 1] + axial_log)
        leak[prox] = 0.22 * np.exp(
            variation[branch, 2]
            + common_log
            + 0.5 * split_log * split_sign[branch]
        )
        leak[dist] = (
            0.18
            * np.exp(
                variation[branch, 3]
                + common_log
                + split_log * split_sign[branch]
            )
            + branch_delta[branch]
        )

        edges.extend(
            [
                (0, prox, float(g_soma)),
                (prox, dist, float(g_dist)),
            ]
        )

    return compile_passive(
        n,
        edges,
        leak,
        stimulation_nodes=tuple(stimulation_nodes),
    )


def candidate_probes():
    subsets = list(itertools.combinations(range(6), 3))
    probes = [unit_energy_subset(6, s) for s in subsets]
    return subsets, probes


def sampled_trace(model, probe: np.ndarray) -> np.ndarray:
    return model.response(probe, steps=70, pulse_steps=4)[TIME_INDEX]


def finite_difference_sensitivities(probes: list[np.ndarray]):
    """Return target J[probe,branch,time] and nuisance N[probe,k,time]."""
    base = heterogeneous_model()
    base_response = np.stack(
        [sampled_trace(base, p) for p in probes],
        axis=0,
    )

    target = np.empty((len(probes), 6, len(TIME_INDEX)), dtype=float)
    for branch in range(6):
        dp = np.zeros(6, dtype=float)
        dp[branch] = FD_EPS
        dm = -dp
        plus = heterogeneous_model(branch_delta=dp)
        minus = heterogeneous_model(branch_delta=dm)
        for pi, probe in enumerate(probes):
            target[pi, branch] = (
                sampled_trace(plus, probe) - sampled_trace(minus, probe)
            ) / (2.0 * FD_EPS)

    nuisance = np.empty((len(probes), 4, len(TIME_INDEX)), dtype=float)
    names = ("common_log", "split_log", "axial_log")
    for ni, name in enumerate(names):
        plus_kwargs = {"common_log": 0.0, "split_log": 0.0, "axial_log": 0.0}
        minus_kwargs = plus_kwargs.copy()
        plus_kwargs[name] = FD_EPS
        minus_kwargs[name] = -FD_EPS
        plus = heterogeneous_model(**plus_kwargs)
        minus = heterogeneous_model(**minus_kwargs)
        for pi, probe in enumerate(probes):
            nuisance[pi, ni] = (
                sampled_trace(plus, probe) - sampled_trace(minus, probe)
            ) / (2.0 * FD_EPS)

    # d y / d log(stimulation gain) at gain=1 is y itself in this passive system.
    nuisance[:, 3, :] = base_response
    return target, nuisance


def sensitivity_score(
    selected: list[int],
    target: np.ndarray,
    nuisance: np.ndarray,
    *,
    nuisance_aware: bool,
) -> float:
    means = np.stack(
        [
            np.concatenate(
                [TARGET_DELTA_LEAK * target[p, h] for p in selected]
            )
            for h in range(6)
        ],
        axis=0,
    )
    features = means.shape[1]

    covariance = (NOISE_STD**2) * np.eye(features)
    if nuisance_aware:
        N = np.stack(
            [
                np.concatenate([nuisance[p, k] for p in selected])
                for k in range(4)
            ],
            axis=1,
        )
        weighted = N * NUISANCE_STD[None, :]
        covariance = covariance + weighted @ weighted.T

    L = np.linalg.cholesky(covariance + 1e-15 * np.eye(features))
    whitened_means = np.linalg.solve(L, means.T).T
    return gaussian_localization_accuracy_estimate(
        whitened_means,
        noise_std=1.0,
        samples_per_hypothesis=DESIGN_MC,
        seed=DESIGN_SEED + len(selected),
    )


def greedy(score_fn, count: int, budget: int = 3):
    selected: list[int] = []
    trace = []
    for step in range(budget):
        best = None
        for probe_index in range(count):
            if probe_index in selected:
                continue
            score = float(score_fn(selected + [probe_index]))
            key = (score, -probe_index)
            if best is None or key > best[0]:
                best = (key, probe_index, score)
        if best is None:
            break
        _, probe_index, score = best
        selected.append(int(probe_index))
        trace.append(
            {
                "step": step + 1,
                "probe_index": int(probe_index),
                "predicted_localization_accuracy": float(score),
            }
        )
    return selected, trace


def response_library(
    nuisance_values: np.ndarray,
    stimulation_gain: np.ndarray,
    probes: list[np.ndarray],
) -> np.ndarray:
    """Return [probe,hypothesis,nuisance_sample,time].

    Superposition is used only to avoid repeating equivalent simulations: for
    each compiled passive model we simulate the six unit branch inputs once and
    form every positive 3-of-6 probe as their exact linear combination.
    """
    S = int(len(nuisance_values))
    P = int(len(probes))
    out = np.empty((P, 6, S, len(TIME_INDEX)), dtype=float)
    basis = np.eye(6, dtype=float)

    for si, (common_log, split_log, axial_log) in enumerate(nuisance_values):
        for hidden in range(6):
            delta = np.zeros(6, dtype=float)
            delta[hidden] = TARGET_DELTA_LEAK
            model = heterogeneous_model(
                branch_delta=delta,
                common_log=float(common_log),
                split_log=float(split_log),
                axial_log=float(axial_log),
            )
            single = np.stack(
                [sampled_trace(model, basis[b]) for b in range(6)],
                axis=0,
            )
            for pi, probe in enumerate(probes):
                out[pi, hidden, si] = (
                    float(stimulation_gain[si]) * (probe @ single)
                )
    return out


def fit_nuisance_lda(library: np.ndarray, selected: list[int]):
    X = np.concatenate([library[p] for p in selected], axis=2)
    H, S, F = X.shape
    means = np.mean(X, axis=1)
    centered = (X - means[:, None, :]).reshape(H * S, F)
    covariance = (centered.T @ centered) / float(H * S - H)
    covariance += (NOISE_STD**2) * np.eye(F)
    L = np.linalg.cholesky(covariance + 1e-12 * np.eye(F))
    whitened_means = np.linalg.solve(L, means.T).T
    return L, whitened_means


def evaluate_design(
    train_library: np.ndarray,
    eval_library: np.ndarray,
    selected: list[int],
    *,
    seed: int = EVAL_NOISE_SEED,
) -> float:
    L, means = fit_nuisance_lda(train_library, selected)
    X = np.concatenate([eval_library[p] for p in selected], axis=2)
    H, S, F = X.shape
    rng = np.random.default_rng(seed)
    Y = X + rng.normal(0.0, NOISE_STD, size=X.shape)
    Yw = np.linalg.solve(L, Y.reshape(H * S, F).T).T.reshape(H, S, F)
    distances = np.sum(
        (Yw[:, :, None, :] - means[None, None, :, :]) ** 2,
        axis=3,
    )
    predicted = np.argmin(distances, axis=2)
    truth = np.arange(H)[:, None]
    return float(np.mean(predicted == truth))


def nuisance_draws(samples: int, seed: int):
    rng = np.random.default_rng(seed)
    values = np.stack(
        [
            rng.normal(0.0, NUISANCE_STD[0], samples),
            rng.normal(0.0, NUISANCE_STD[1], samples),
            rng.normal(0.0, NUISANCE_STD[2], samples),
        ],
        axis=1,
    )
    gain = np.exp(rng.normal(0.0, NUISANCE_STD[3], samples))
    return values, gain


def summary_stats(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--skip-exhaustive",
        action="store_true",
        help="skip the 1140-set validation audit",
    )
    args = ap.parse_args()

    subsets, probes = candidate_probes()
    target, nuisance = finite_difference_sensitivities(probes)

    noise_only, noise_trace = greedy(
        lambda s: sensitivity_score(
            s, target, nuisance, nuisance_aware=False
        ),
        len(probes),
    )
    robust, robust_trace = greedy(
        lambda s: sensitivity_score(
            s, target, nuisance, nuisance_aware=True
        ),
        len(probes),
    )

    # Gate-3 exact-model multi-hypothesis design, carried in as an attacker.
    gate3_exact_model = [18, 10, 7]

    train_values, train_gain = nuisance_draws(
        TRAIN_NUISANCE_SAMPLES, TRAIN_SEED
    )
    eval_values, eval_gain = nuisance_draws(
        EVAL_NUISANCE_SAMPLES, EVAL_SEED
    )
    train_library = response_library(train_values, train_gain, probes)
    eval_library = response_library(eval_values, eval_gain, probes)

    accuracies = {
        "noise_only_sensitivity_greedy": evaluate_design(
            train_library, eval_library, noise_only
        ),
        "nuisance_aware_sensitivity_greedy": evaluate_design(
            train_library, eval_library, robust
        ),
        "gate3_exact_model_design": evaluate_design(
            train_library, eval_library, gate3_exact_model
        ),
    }

    rng = np.random.default_rng(RANDOM_SEED)
    random_triples = [
        tuple(sorted(rng.choice(len(probes), size=3, replace=False)))
        for _ in range(RANDOM_DESIGNS)
    ]
    random_accuracy = np.asarray(
        [
            evaluate_design(
                train_library,
                eval_library,
                list(triple),
            )
            for triple in random_triples
        ],
        dtype=float,
    )

    exhaustive = None
    if not args.skip_exhaustive:
        triples = list(itertools.combinations(range(len(probes)), 3))
        values = np.asarray(
            [
                evaluate_design(
                    train_library,
                    eval_library,
                    list(triple),
                )
                for triple in triples
            ],
            dtype=float,
        )
        robust_accuracy = accuracies["nuisance_aware_sensitivity_greedy"]
        best_index = int(np.argmax(values))
        exhaustive = {
            "sets": len(triples),
            **summary_stats(values),
            "nuisance_aware_greedy_rank": int(
                1 + np.sum(values > robust_accuracy)
            ),
            "nuisance_aware_greedy_percentile": float(
                np.mean(values <= robust_accuracy)
            ),
            "best_probe_indices": list(triples[best_index]),
            "best_subsets": [
                list(subsets[i]) for i in triples[best_index]
            ],
        }

    result = {
        "gate": "nuisance_wall_sensitivity_subspace",
        "model": "known heterogeneous six-arm passive morphology",
        "hidden_change": {
            "type": "one distal branch additive passive leak",
            "delta": TARGET_DELTA_LEAK,
        },
        "recording_site": "soma only",
        "budget_trials": 3,
        "candidate_family": "20 positive unit-energy 3-of-6 stimulation patterns",
        "waveform_samples_per_trial": int(len(TIME_INDEX)),
        "noise_std": NOISE_STD,
        "nuisance_log_std": {
            "common_leak": float(NUISANCE_STD[0]),
            "split_leak_bias": float(NUISANCE_STD[1]),
            "axial_scale": float(NUISANCE_STD[2]),
            "global_stimulation_gain": float(NUISANCE_STD[3]),
        },
        "designs": {
            "noise_only_sensitivity_greedy": {
                "probe_indices": noise_only,
                "subsets": [list(subsets[i]) for i in noise_only],
                "trace": noise_trace,
            },
            "nuisance_aware_sensitivity_greedy": {
                "probe_indices": robust,
                "subsets": [list(subsets[i]) for i in robust],
                "trace": robust_trace,
                "covariance_model": "sigma^2 I + N Lambda N^T",
            },
            "gate3_exact_model_design": {
                "probe_indices": gate3_exact_model,
                "subsets": [
                    list(subsets[i]) for i in gate3_exact_model
                ],
            },
        },
        "independent_continuous_nuisance_evaluation": {
            "train_nuisance_samples": TRAIN_NUISANCE_SAMPLES,
            "eval_nuisance_samples": EVAL_NUISANCE_SAMPLES,
            "hidden_cases": 6 * EVAL_NUISANCE_SAMPLES,
            "classifier": (
                "common-covariance nuisance-trained LDA; same decoder family "
                "for every stimulation design"
            ),
            "accuracy": accuracies,
        },
        "random_200_baseline": {
            "designs": RANDOM_DESIGNS,
            **summary_stats(random_accuracy),
        },
        "exhaustive_equal_budget_audit": exhaustive,
        "interpretation": (
            "The 84% exact-model Gate-3 result does not survive specimen-level "
            "nuisance uncertainty. A local planner that ignores nuisance directions "
            "falls to about 40%, and the old exact-model design is only about 42%. "
            "Whitening branch-change sensitivities by the covariance induced by four "
            "declared nuisance directions recovers a modest but real advantage: about "
            "42.8% under independent continuous nuisance draws, above the roughly "
            "38.7% random-set mean. Exhaustively it ranks near the top few percent but "
            "is not globally optimal. This is the first direct bridge to Operaattori's "
            "tangent machinery: a useful probe should make the target sensitivity not "
            "merely large, but distinguishable from the nuisance sensitivity subspace."
        ),
        "stopping_line": (
            "This is still a known-family oracle benchmark. The planner and decoder "
            "know the nuisance distribution and the same model family generates the "
            "evaluation data. The next mandatory attacker is model mismatch."
        ),
    }

    # Receipts are broad enough to tolerate platform-level floating differences.
    assert accuracies["nuisance_aware_sensitivity_greedy"] > 0.41
    assert accuracies["nuisance_aware_sensitivity_greedy"] > (
        random_accuracy.mean() + 0.025
    )
    assert accuracies["noise_only_sensitivity_greedy"] < 0.42

    out = Path("results/gate4_nuisance_wall.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
