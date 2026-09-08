"""Read-only numerical audit of OperaattoriAktiivinenDendriitti eba6b9c.

Place next to the repository's Python modules at the pinned revision; run with
OPENBLAS_NUM_THREADS=1 python review_current_checks.py. Only review_*.npz/json
files are written. This does not run or modify the original gates' main().
"""
from pathlib import Path
import json
import numpy as np
import gate3_broken_symmetry as g3
import gate4_nuisance_wall as g4
import gate5_model_mismatch_paired_baseline as g5
from active_dendrite import response_bank, unique_probe_codes
from planner import gaussian_localization_accuracy_estimate as estimate

PIN = "eba6b9cf7884dd269240b325f779c82192577bfc"
HERE = Path(__file__).resolve().parent
DESIGNS = {
    "actual_gate3": [18, 17, 11],
    "hardcoded_as_gate3": [18, 10, 7],
    "gate4_nuisance": [10, 2, 6],
    "noise_only": [18, 0, 13],
    "reduced_paired": [4, 11, 16],
}


def cache_library(name, builder):
    path = HERE / ("review_" + name + ".npz")
    if path.exists():
        with np.load(path) as f:
            return tuple(f[k] for k in f.files)
    print("Building " + name, flush=True)
    values = builder()
    if not isinstance(values, tuple):
        values = (values,)
    np.savez_compressed(path, *values)
    return values


def summary(values):
    return {
        "n_designs": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main():
    subsets, probes = g4.candidate_probes()
    report = {
        "repository": "anttiluode/OperaattoriAktiivinenDendriitti",
        "commit": PIN,
        "numpy_version": np.__version__,
        "designs": {k: {"indices": v, "subsets": [subsets[i] for i in v]}
                    for k, v in DESIGNS.items()},
    }
    models = [g3.heterogeneous_six_arm(changed_region=i) for i in range(6)]
    bank = response_bank(models, probes)
    model_error = 0.0
    for h in range(6):
        d = np.zeros(6)
        d[h] = g4.TARGET_DELTA_LEAK
        model_error = max(model_error, float(np.max(np.abs(
            models[h].A - g4.heterogeneous_model(branch_delta=d).A))))
    report["gate3_gate4_zero_nuisance_operator_max_difference"] = model_error
    report["sampling_control"] = {}
    for name in ["actual_gate3", "hardcoded_as_gate3"]:
        a = bank[:, DESIGNS[name], :]
        report["sampling_control"][name] = {
            "70_samples_per_probe": estimate(a.reshape(6, -1),
                noise_std=g3.NOISE_STD, samples_per_hypothesis=100000, seed=1333),
            "14_samples_per_probe": estimate(a[:, :, g4.TIME_INDEX].reshape(6, -1),
                noise_std=g3.NOISE_STD, samples_per_hypothesis=100000, seed=1333),
        }
    report["exact_tie_accuracy_estimator"] = {
        "six_identical_templates_returned": estimate(np.zeros((6, 3)),
            noise_std=0.003, samples_per_hypothesis=1000, seed=1),
        "correct_uniform_prior_accuracy": 1 / 6,
    }
    rng = np.random.default_rng(g3.RANDOM_SEED)
    strata = {}
    for _ in range(g3.RANDOM_DESIGNS):
        selected = sorted(rng.choice(len(probes), size=3, replace=False).tolist())
        codes = unique_probe_codes([probes[i] for i in selected])
        acc = estimate(bank[:, selected, :].reshape(6, -1), noise_std=g3.NOISE_STD,
            samples_per_hypothesis=g3.RANDOM_ESTIMATE_SAMPLES,
            seed=g3.RANDOM_ESTIMATE_SEED)
        strata.setdefault(str(codes), []).append(acc)
    report["gate3_random_designs_stratified_by_membership_code_count"] = {
        k: summary(v) for k, v in strata.items()}
    print(json.dumps({k: report[k] for k in ["sampling_control",
        "gate3_random_designs_stratified_by_membership_code_count",
        "exact_tie_accuracy_estimator"]}, indent=2), flush=True)

    train_values, train_gain = g4.nuisance_draws(g4.TRAIN_NUISANCE_SAMPLES,
                                                g4.TRAIN_SEED)
    eval4_values, eval4_gain = g4.nuisance_draws(g4.EVAL_NUISANCE_SAMPLES,
                                                g4.EVAL_SEED)
    train_abs, = cache_library("train_absolute", lambda: g4.response_library(
        train_values, train_gain, probes))
    eval4, = cache_library("gate4_evaluation", lambda: g4.response_library(
        eval4_values, eval4_gain, probes))
    report["gate4_evaluation"] = {
        k: g4.evaluate_design(train_abs, eval4, v) for k, v in DESIGNS.items()}
    print("Gate 4 " + json.dumps(report["gate4_evaluation"]), flush=True)
    strata4 = {}
    rng = np.random.default_rng(g4.RANDOM_SEED)
    for _ in range(g4.RANDOM_DESIGNS):
        selected = sorted(rng.choice(len(probes), size=3, replace=False).tolist())
        codes = unique_probe_codes([probes[i] for i in selected])
        strata4.setdefault(str(codes), []).append(
            g4.evaluate_design(train_abs, eval4, selected))
    report["gate4_random_designs_stratified_by_membership_code_count"] = {
        k: summary(v) for k, v in strata4.items()}

    train_delta, = cache_library("train_delta", lambda: g5.reduced_delta_library(
        train_values, train_gain, probes))
    eval5_values, eval5_gain = g4.nuisance_draws(g5.EVAL_SAMPLES, g5.EVAL_SEED)
    rich_abs, rich_delta = cache_library("gate5_evaluation", lambda:
        g5.rich_absolute_and_delta_library(eval5_values, eval5_gain, probes))
    report["gate5_evaluation"] = {
        k: {
            "absolute": g5.evaluate(train_abs, rich_abs, v, noise_std=g4.NOISE_STD),
            "paired": g5.evaluate(train_delta, rich_delta, v,
                                  noise_std=g5.PAIRED_NOISE_STD),
        } for k, v in DESIGNS.items()}
    print("Gate 5 " + json.dumps(report["gate5_evaluation"]), flush=True)
    strata5_abs, strata5_paired = {}, {}
    rng = np.random.default_rng(g5.RANDOM_SEED)
    for _ in range(g5.RANDOM_DESIGNS):
        selected = sorted(rng.choice(len(probes), size=3, replace=False).tolist())
        codes = unique_probe_codes([probes[i] for i in selected])
        strata5_abs.setdefault(str(codes), []).append(
            g5.evaluate(train_abs, rich_abs, selected, noise_std=g4.NOISE_STD))
        strata5_paired.setdefault(str(codes), []).append(
            g5.evaluate(train_delta, rich_delta, selected,
                        noise_std=g5.PAIRED_NOISE_STD))
    report["gate5_random_designs_stratified_by_membership_code_count"] = {
        "absolute": {k: summary(v) for k, v in strata5_abs.items()},
        "paired": {k: summary(v) for k, v in strata5_paired.items()},
    }

    # Exact linear after-only gain perturbation, shared across all probes and
    # hypotheses for each nuisance specimen. The reduced decoder is unchanged.
    drift_z = np.random.default_rng(909).normal(size=g5.EVAL_SAMPLES)
    report["after_only_gain_drift_attacker"] = {
        "seed": 909,
        "description": "delta + (exp(log_gain_shift)-1)*absolute_after; same shift across probes",
        "paired_accuracy": {},
    }
    for std in [0.0, 0.01, 0.05]:
        shifted = rich_delta + np.expm1(std * drift_z)[None, None, :, None] * rich_abs
        report["after_only_gain_drift_attacker"]["paired_accuracy"][str(std)] = {
            k: g5.evaluate(train_delta, shifted, DESIGNS[k],
                           noise_std=g5.PAIRED_NOISE_STD)
            for k in ["gate4_nuisance", "noise_only", "actual_gate3"]}
    out = HERE / "review_current_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("Gain drift " + json.dumps(report["after_only_gain_drift_attacker"]), flush=True)
    print("Wrote " + str(out), flush=True)


if __name__ == "__main__":
    main()
