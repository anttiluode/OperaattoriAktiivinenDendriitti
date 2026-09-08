# Operaattori active dendrite: progress review and reproducible checks

Reviewed 2026-09-08. Repository: [anttiluode/OperaattoriAktiivinenDendriitti](https://github.com/anttiluode/OperaattoriAktiivinenDendriitti), pinned commit [eba6b9cf7884dd269240b325f779c82192577bfc](https://github.com/anttiluode/OperaattoriAktiivinenDendriitti/tree/eba6b9cf7884dd269240b325f779c82192577bfc). This is an independent read-only review. No repository changes were pushed.

The project now has a coherent inverse-problem benchmark: choose positive stimulation patterns that expose a hidden local change through a limited recording site, while accounting for uncertain parameters. Gates 4 and 5 are useful progress. The strongest practical development is the move to same-cell change identification using a measured baseline. The current evidence supports this direction, with the corrections below.

## 1. Fix the carried Gate-3 comparator

[Gate 3's receipt](https://github.com/anttiluode/OperaattoriAktiivinenDendriitti/blob/eba6b9cf7884dd269240b325f779c82192577bfc/results/gate3_broken_symmetry.json) selects indices `[18,17,11]`: `(2,4,5), (2,3,5), (1,2,4)`.

[Gate 4](https://github.com/anttiluode/OperaattoriAktiivinenDendriitti/blob/eba6b9cf7884dd269240b325f779c82192577bfc/gate4_nuisance_wall.py) and [Gate 5](https://github.com/anttiluode/OperaattoriAktiivinenDendriitti/blob/eba6b9cf7884dd269240b325f779c82192577bfc/gate5_model_mismatch_paired_baseline.py) instead hardcode `[18,10,7]` and label it Gate 3. This is a different design. The published numbers reproduce for that different design.

Using the actual Gate-3 design gives:

| Evaluation | Hardcoded design labelled Gate 3 | Actual Gate-3 design |
|---|---:|---:|
| Gate 4, nuisance uncertainty | 41.6667% | 37.3750% |
| Gate 5, absolute post-change | 28.7917% | 37.2917% |
| Gate 5, paired before/after | 40.3958% | 41.7292% |

The Gate-4 nuisance-aware design and Gate-5 paired design receipts themselves reproduce. Fix the comparator by reading a versioned design specification or receipt rather than copying indices, then refresh downstream interpretation.

## 2. Separate temporal sampling from nuisance uncertainty

Gate 3 observes all 70 waveform samples. Gate 4 observes `np.arange(0,70,5)`: 14 samples. Both attach independent Gaussian noise of standard deviation 0.003 to each retained sample. This substantially changes the information budget.

With the actual Gate-3 probes, exactly the same known neuron, and no nuisance variation:

| Retained samples per probe | Estimated accuracy |
|---|---:|
| 70 | 84.1875% |
| 14 | 48.4625% |

These estimates use 100,000 Gaussian discriminant draws per hypothesis and seed 1333. The Gate-3 and zero-nuisance Gate-4 propagation matrices are exactly equal in this rerun. Adding nuisance with Gate 4's training and evaluation gives 37.3750% for this design.

Thus the narrative should distinguish **84.19 -> 48.46 from sampling**, then **48.46 -> 37.38 under nuisance and its fitted decoder**. The original 84 -> 42 comparison changes both the design and sampling. The latter comparison is not an isolated measurement of the nuisance penalty.

For future comparisons, hold observation duration, sample selection, and noise covariance fixed. Sampling denser should not create artificial independent information when noise is temporally correlated.

## 3. Preserve the valid-code random baseline beyond Gate 1

Breaking exact symmetry removes equal performance of all valid codebooks. It does not remove coding as an explanation for much of the advantage over unrestricted random designs.

I regenerated each gate's existing 200 random designs using its original seed and grouped them by the number of distinct branch membership codes. The strongest simple geometry-independent control is a random design conditioned on all six codes being distinct.

| Experiment | Chosen design | Random mean, all designs | Random mean, six distinct codes |
|---|---:|---:|---:|
| Gate 3, exact model | 84.19% estimate | 76.50% | 83.02% (79 designs) |
| Gate 4, nuisance | 42.75% | 38.69% | 39.65% (81 designs) |
| Gate 5, absolute mismatch | 37.875% | 31.26% | 32.57% (79 designs) |
| Gate 5, paired mismatch | 42.8125% | 39.04% | 40.50% (79 designs) |

Gate 3's geometry-specific advantage over this stronger baseline is about 1.17 percentage points. The valid-code random designs span approximately 80.60-85.76%, so some outperform its greedy choice.

Gate 4 retains about a 3.10-point advantage over valid-code random designs. This is a useful surviving positive result. Gate 5 retains a smaller paired advantage. These are point estimates under the stated synthetic specimen families; they do not establish a universal ranking of planners.

## 4. Paired measurement helps; nuisance-aware design is not clearly superior after pairing

The Gate-4 set reproduces at 37.875% for absolute mismatch and 42.8125% for paired changes. Static before/after error cancellation is useful despite the correctly included square-root-of-two noise penalty.

However, the noise-only selected set also reaches 42.6667% after pairing. The difference is only seven net correct classifications out of 4,800 cases. This receipt does not establish a reliable superiority of the nuisance-aware selector over that selector after pairing. Inference remains nuisance-aware in the evaluations; “noise-only” describes how that probe set was selected.

The most defensible attribution is that the **measurement protocol** recovers useful information under the tested static mismatch. It is not yet evidence that one sophisticated selector is necessary for that recovery.

## 5. Test baseline stability and pay for its acquisition

Gate 5 holds the nuisance parameters, gain, and structural mismatch identical before and after each hidden intervention. Its model-mismatch and capacitance profiles are single fixed draws shared across nuisance samples.

I added a bounded attacker: multiply the post-change stimulation gain by `exp(z*s)`, with one standard-normal `z` per nuisance specimen, shared across probes and hypotheses. Leave the pre-change gain unchanged and keep the same decoder and sensor-noise model. Linearity makes the altered difference exactly

`delta_drift = delta + (exp(z*s)-1) * absolute_after`.

| After-only log gain standard deviation | Gate-4 set, paired accuracy |
|---|---:|
| 0 | 42.8125% |
| 0.01 | 42.0417% |
| 0.05 | 36.1042% |

These are illustrative synthetic controls, not physiological thresholds. They show why a calibration-drift hypothesis should accompany a neuronal-change hypothesis.

The square-root-of-two penalty counts noise, but trial cost is separate: three paired differences require three before plus three after recordings. The current absolute-versus-paired comparison therefore changes measurement expenditure. A fair instrument comparison must declare whether the baseline is pre-existing and sunk or part of the budget.

There is a chronological constraint on Gate 6. After seeing post-change response 1, the observer cannot request a new *pre-change* response for its newly selected probe 2. It needs a baseline panel acquired in advance, a restricted repertoire, or an explicitly uncertain model-based baseline. In the current passive linear model, six measured single-branch baseline responses can synthesize all candidate pattern baselines, but the resulting noise is shared and correlated between patterns. That shortcut must carry its covariance and must not be assumed for nonlinear stimulation.

## 6. Repair the exact-tie boundary case

`gaussian_localization_accuracy_estimate(np.zeros((6,3)), noise_std=.003)` returns 0.0. Six indistinguishable equally probable hypotheses permit chance accuracy 1/6. The strict inequality test incorrectly counts every exact tie as an error for every hypothesis.

This does not explain the heterogeneous positive results, but it matters for the planned blindness and no-change controls. Handle identical-template equivalence classes with proper tie probabilities, or use an explicit declared tie rule in the classifier.

## Research-program implications

Operaattori provides a forward operator and audited geometric sensitivities. Active Dendrite gives those sensitivities a decision problem: select measurements that distinguish a target from other plausible causes. Sigh motivates why query geometry affects recoverability. Jello and ThinkingJello motivate the further problem of probes changing the material; that is not part of the current static identification gates. The baseline and inference state here are external stored resources, acceptable for a scientific instrument but not automatically an intrinsic neuronal memory mechanism.

The key distinction is between a physical difference, a difference visible through the permitted ports, and a difference identifiable under nuisance uncertainty and finite observations. The current Gaussian covariance `R + N Lambda N.T` implements a prior-weighted penalty for nuisance-like responses. It is not an exact projection eliminating every possible nuisance, nor a proof of structural identifiability.

Current targets are six known alternatives, each a fixed positive distal leak increment. There is no unchanged class, unknown magnitude, sign reversal, multi-region change, or calibrated rejection result yet. At the present noise setting roughly 42% six-way accuracy is evidence of information, above 16.67% chance, while still leaving substantial ambiguity. A ranked region set and an honest inconclusive answer are appropriate next outputs.

## Closely related literature

- [Shababo et al., NIPS 2013](https://proceedings.neurips.cc/paper/2013/hash/17c276c8e723eb46aef576537e9d56d0-Abstract.html) already studied online selection of stimulated presynaptic populations from a single postsynaptic recording, using a simpler inference/design model and a richer synthetic generator. Its target is microcircuit synaptic weights, not this repository's branch leak changes. It is a particularly relevant baseline for Gate 6.
- [Pakman et al., 2013 author manuscript](https://sites.stat.columbia.edu/liam/research/pubs/pakman-huggins-synaptic.pdf) studies synaptic location and strength on known dendritic morphologies from noisy incomplete voltage observations, including designed observation combinations. Its observation access differs from a fixed soma sensor.
- [Jaxley, Nature Methods 2025](https://www.nature.com/articles/s41592-025-02895-w) provides differentiable biophysical simulation. Its 1,390-parameter example records throughout the branches, and even there some parameters remain weakly constrained.

These references establish that active neural inference and differentiable neurons are existing research areas. The opportunity here is a rigorously evaluated integration of morphology, local change targets, uncertainty, restricted interventions, baseline cost, and ambiguity reporting. Whether that specific combination advances the literature requires a direct benchmark, not a general claim of novelty.

## Recommended next experiment

Freeze the target definition, corrected comparators, and measurement protocol. Give fixed and adaptive planners the same pre-change baseline panel and post-change measurement budget. Compare against a strong fixed model-based design and a random design with six distinct branch codes.

Maintain one joint posterior over target and shared nuisance parameters through the sequence; do not redraw independent nuisance at every question. Include unchanged cells, gain drift, and held-out mismatch profiles, and evaluate calibration or coverage as well as forced-choice accuracy. The planner should sometimes spend a trial calibrating or request an additional recording site when branch-only probing cannot resolve the ambiguity.

Then run this fixed protocol on the real Operaattori morphology with a small set of audited length/diameter targets. Adaptive superiority should be tested, not made a requirement for entering the real-cell experiment. A robust fixed planner can already be useful. Explicit conductance or synaptic derivatives can follow once verified.

## Reproduction

This archive contains this review, `review_current_checks.py`, and its `review_current_report.json`. Obtain the repository at the pinned commit, copy the script next to its Python modules, install its NumPy requirement, then run:

```sh
OPENBLAS_NUM_THREADS=1 python review_current_checks.py
```

The script reproduces original-design scores, corrects the comparison design without altering source files, controls sampling, stratifies random baselines, checks exact ties, and evaluates the gain-drift attacker. It writes only `review_*.npz` caches and `review_current_report.json` beside itself. Delete those caches if intentionally changing the underlying model or seeds. The supplied receipt used NumPy 2.3.5. All new statistics are reviewer calculations under declared toy-model assumptions.
