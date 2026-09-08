# OperaattoriAktiivinenDendriitti

**Active Dendritic Identification / Dendritic State Tomography**

This repo asks a narrower and more practical question than "reconstruct the whole neuron":

> **Given a known cell, a stored baseline, a small set of plausible hidden changes, and only one or a few recording sites, which physically realizable experiment makes those hidden states distinguishable?**

The target is a **change detector + stimulation planner + spatial ambiguity map** for a known dendritic morphology.

This is not yet experimental tomography and it does not claim that arbitrary synaptic distributions are recoverable from a soma trace.

The project grows out of three connected threads:

1. [SighImageSuper](https://github.com/anttiluode/SighImageSuper): a distinction may physically exist yet be invisible under one probe/readout geometry; active questioning can reveal it.
2. [Operaattori](https://github.com/anttiluode/Operaattori): morphology can be compiled into an electrical operator with audited response tangents to local geometry.
3. [GeometricNeuronOriginReview](https://github.com/anttiluode/GeometricNeuronOriginReview): the old Perception Lab ECG loop showed, after re-analysis, that changing what a feedback loop measures can change the trajectory subsequently observed.

The original node laboratory is [PerceptionLab](https://github.com/anttiluode/PerceptionLab).

---

# The inverse problem

Let

- `theta` = hidden dendritic/material parameters,
- `p` = a stimulation protocol,
- `F(theta,p)` = the forward neuron model,
- `C` = the available recording operator.

The experimenter sees

```text
y = C F(theta,p) + noise
```

and may choose `p`.

The first toy gates asked which stimulation separates candidate states relative to noise. The later gates add nuisance uncertainty, model mismatch, baseline cost, an unchanged-cell hypothesis, calibration drift, and honest ambiguity reporting.

The current operational question is:

> **Which local change remains identifiable after nuisance, model error, measurement cost, and calibration drift are allowed to imitate it?**

---

# Gate history: what survived the attacks

## Gate 0 — exact blindness caused by symmetry

A mirror-symmetric Y tree has either branch A or mirror branch B changed. With one soma sensor, symmetric stimulation produces identical responses to numerical precision. An asymmetric branch stimulation makes the hidden distinction visible.

> **Unobservable under one question does not mean physically absent.**

This is the direct Sigh -> dendrite bridge.

## Gates 1-2 — the first 99% result was mostly coding

The original six-arm toy appeared to localize one hidden branch with ~99% accuracy using three grouped stimulations.

The audit showed that in the perfectly symmetric toy, most of that result was simply assigning six unique three-bit stimulation codes. Hundreds of random code-valid designs performed essentially the same.

The first planner also had a scoring flaw: it counted arbitrarily tiny nonzero separations without asking whether they exceeded noise.

Both issues are retained as attackers rather than hidden.

## Gate 3 — break exact symmetry

With fixed branch heterogeneity, the trivial three-bit equivalence disappears. A genuine multi-hypothesis design criterion reached about `0.84` in the matched toy, while the controlled random distribution was lower on average.

That result still assumed the planner knew the exact candidate models.

## Gate 4 — nuisance wall

The hidden branch change is mixed with uncertain common leak, split bias, axial scale, and stimulation calibration.

The useful design quantity becomes not merely a large target derivative but a target derivative that cannot be easily imitated by nuisance directions.

For a stimulation panel, write

```text
J = target-sensitivity columns
N = nuisance-sensitivity columns
```

and use the nuisance-induced covariance

```text
Sigma = sigma_noise^2 I + N Lambda N^T.
```

Then compare target signatures after whitening by `Sigma`.

The central handoff to real Operaattori is:

> **Do not maximize a morphology derivative merely because it is large. Choose an experiment in which the target derivative points away from the nuisance derivative subspace.**

## Gate 5 — model mismatch and paired change measurement

The inference model is deliberately simpler than the synthetic evaluator. Absolute post-change traces degrade strongly because static model error can imitate the target.

The useful rescue is to measure the same cell before and after:

```text
delta y = y_after - y_before.
```

Much of the benefit came from **what was measured** — change relative to the same cell — rather than from a sophisticated selector alone.

This changed the practical application from absolute reconstruction to:

> **I know this cell's baseline. Something changed. Which experiment best separates the plausible causes?**

The baseline is therefore treated as a real resource, not a free subtraction.

---

# Gate 6 — fair baseline memory, fixed vs adaptive

`gate6_fixed_vs_adaptive_baseline_panel.py`

The independent review found three important issues in the earlier gates:

- the wrong Gate-3 comparator was accidentally carried forward;
- Gate 3 and Gate 4 used different waveform sample counts, confounding the apparent nuisance penalty;
- exact duplicate templates were mishandled by the old accuracy estimator.

Those are fixed from Gate 6 onward. The response is documented in [`REVIEW_RESPONSE_080926.md`](REVIEW_RESPONSE_080926.md).

Every strategy now gets the same explicit budget:

```text
20 pre-change baseline recordings
 3 post-change recordings
14 samples per waveform
 1 soma sensor
----------------------------
23 recordings total
```

No adaptive method is allowed to invent a pre-change recording after the hidden event.

There are seven target classes:

```text
UNCHANGED
branch 0 changed
branch 1 changed
...
branch 5 changed
```

The target magnitude is unknown. The model includes continuous nuisance, 5% after-only stimulation-gain drift, and a new held-out capacitance/leak mismatch profile for every synthetic specimen. Inference uses a reduced model and one joint particle belief over target + nuisance across all three trials.

On 1,050 held-out specimens:

| strategy | top-1 | top-2 | unchanged recall |
|---|---:|---:|---:|
| **Gate-4 nuisance fixed** | **36.29%** | **59.33%** | 46.67% |
| first causal adaptive planner | 32.86% | 55.24% | **50.67%** |
| Gate-5 paired fixed | 32.95% | 53.43% | 36.67% |
| actual Gate-3 fixed | 30.95% | 51.71% | 36.67% |
| chance | 14.29% | — | — |

The first adaptive planner did **not** beat the strong fixed design. That result is preserved rather than tuned away.

Among sampled random fixed designs restricted to those that already give all six branches distinct stimulation codes, the mean top-1 accuracy was about `32.08%`; the Gate-4 fixed set was about `+4.2` percentage points above that controlled mean.

At only ~36% seven-way top-1 accuracy, ambiguity is part of the answer. Gate 6 therefore also records posterior probability, top-2 accuracy, Brier score, log loss, calibration error, unchanged recall, and abstention curves.

> **A useful instrument must sometimes say AMBIGUOUS or UNIDENTIFIABLE.**

---

# New insight: the baseline panel is an empirical operator fingerprint

Gate 6 pays for 20 pre-change recordings, one for each balanced `3-of-6` stimulation pattern.

But the passive toy models are linear in stimulation. Those 20 traces are not 20 unrelated memories. They are redundant measurements of a lower-dimensional stimulation-to-soma transfer operator.

Let

```text
P  = [20 x 6] stimulation-design matrix
H0 = [6 x time] single-branch baseline transfer waveforms
Y0 = measured 20-probe baseline panel.
```

Then approximately

```text
Y0 = P H0 + noise.
```

Instead of storing each noisy baseline answer independently, estimate

```text
H0_hat = (P^T P)^-1 P^T Y0
```

(or a regularized/generalized least-squares version when covariance is nontrivial).

A future baseline for any stimulation in the calibrated linear input span can then be synthesized as

```text
y_before_hat(p) = p^T H0_hat
```

with declared prediction covariance.

This changes the meaning of baseline memory:

> **The pre-change panel is not merely an episodic lookup table. It can be compressed into an empirical fingerprint of how this particular cell maps stimulation into measurement.**

An independent September 2026 review tested this reuse of the already-paid-for baseline panel and reported an improvement of the existing fixed Gate-6 panel from roughly `36.29%` to `40.00%` on the same cases, and `34.10% -> 38.95%` on a fresh synthetic cohort. Those numbers are a **review result / handoff**, not yet a frozen first-class gate in this repository.

The next implementation should therefore be **Gate 6b: global baseline operator fit**, including the induced correlations in its uncertainty model and fresh-seed validation.

This also relaxes the chronology constraint in the linear regime. An adaptive post-change planner need not necessarily have directly recorded the exact future probe before the change; it may ask a new probe whose pre-change response can be predicted from `H0_hat`, provided that prediction uncertainty is fully counted.

For nonlinear or state-dependent experiments this shortcut will fail unless the baseline model is expanded appropriately.

---

# New bridge: changing the dynamics can itself be an experiment

The old [GeometricNeuronOriginReview](https://github.com/anttiluode/GeometricNeuronOriginReview) finally explains the Perception Lab `ecg.json` loop.

The important correction is that the graph was not sampling four eigenmodes. `ImageToVectorNode` resized a generated checkerboard, flattened it, and the first four values were fed back into a finite-memory homeostatic controller. Changing vector size changed **what spatial region was measured**, and because that measurement was inside feedback, it changed the future trajectory.

That suggests a stronger extension of Active Dendrite.

So far the planner chooses mainly the input `p` in

```text
x_(t+1) = A(theta) x_t + B p_t
y_t     = C x_t.
```

But an experiment may also choose an operating condition `q` that changes the effective dynamics:

```text
A -> A(q).
```

If observation participates in feedback,

```text
u_t = p_t + K_q C_q x_t,
```

then even the closed-loop operator changes:

```text
A_closed(q) = A + B K_q C_q.
```

This motivates a later design question:

> **Can we choose a temporary dynamical regime in which a hidden target sensitivity rotates away from nuisance directions that were inseparable at the default operating point?**

Possible real-neuron controls, depending on what the simulator/experiment can physically support, include stimulation timing, strength, active NMDA state, shunting/inhibitory context, holding condition, or controlled feedback.

This is not yet a gate and should not be mixed into the real-morphology transfer until the simpler fixed-panel protocol is established.

---

# Direct bridge to real Operaattori morphology

The next major transfer should use the audited real [Operaattori](https://github.com/anttiluode/Operaattori) morphology rather than adding ever more detail to the six-arm toy.

Start with parameters whose tangents are already audited:

```text
local LENGTH change
local DIAMETER change
POSE null as a negative control
```

Freeze the Gate-6 accounting:

- explicit baseline resource;
- same post-change trial budget;
- same soma observation window;
- unchanged-cell class;
- calibration drift;
- nuisance directions;
- model mismatch where feasible;
- posterior/ambiguity output;
- strong fixed panel first;
- adaptive superiority as a separate hypothesis, not a prerequisite.

Then add the new baseline-operator idea:

```text
known morphology/model prior
        +
empirical baseline transfer fingerprint
        +
audited local geometry tangents
        +
nuisance sensitivities
        ->
which local changes remain identifiable?
```

The pose null matters especially: if the modeled electrical measurement is intrinsically blind to a pure pose change, the system must report that blindness rather than invent a location.

---

# Connection to the larger program

The project collection is converging on a common object:

> **Which distinctions can a system recover through its available actions, given uncertainty, limited measurements, reference-memory cost, and the possibility that those actions change the system?**

The roles are becoming clearer:

- **Operaattori** — structure/morphology compiles the forward response operator and its tangents.
- **SighImageSuper** — persistence, travelling traces, changed material, active queries, and self-interrogation.
- **GeometricNeuronOriginReview / PerceptionLab ECG** — observation inside feedback can alter the future dynamics being observed.
- **Active Dendrite** — turn those ideas into an external experiment-design problem with explicit budgets, nuisance and ambiguity.
- **Jello / ThinkingJello** — ask how the interrogation itself changes the material and how self-generated effects are separated from new evidence.
- **368** — account for the cost of retaining, refreshing, retrieving, and replacing reference memory.

The larger operational version is now:

> **Which distinctions can a bounded system make observable by choosing its inputs, observations, and temporary dynamics — while paying for uncertainty, baseline memory, and the back-action of asking?**

---

# Files / gates

```bash
python gate0_symmetry_receipt.py
python gate1_six_region_localization.py
python gate2_greedy_planner.py
python gate2a_planner_audit.py
python gate3_broken_symmetry.py
python gate4_nuisance_wall.py
python gate5_model_mismatch_paired_baseline.py
python gate6_fixed_vs_adaptive_baseline_panel.py
```

See also:

- [`LATEST_RESULT.md`](LATEST_RESULT.md)
- [`REVIEW_RESPONSE_080926.md`](REVIEW_RESPONSE_080926.md)
- [`OPERAATTORI_BRIDGE.md`](OPERAATTORI_BRIDGE.md)

---

# Next gates

**Gate 6b — global baseline operator fit**  
Use the full pre-change panel jointly, propagate its covariance correctly, validate on fresh seeds, and compare calibration/unchanged detection as well as localization.

**Gate 7 — real Operaattori morphology**  
Known reconstructed cell, local length/diameter changes, pose null, soma-only measurement first.

**Gate 8 — empirical-baseline + morphology-tangent fusion**  
Combine the model prior with the particular cell's measured transfer fingerprint instead of asking either source to carry the whole inverse problem.

**Gate 9 — synaptic/local conductance parameters**  
Add explicit parameter derivatives or controlled finite differences for the quantities actually inferred.

**Gate 10 — dynamics as a query**  
Under equal budgets, test whether changing timing/drive/active state/feedback reveals target directions hidden at the default operating point.

**Gate 11 — sparse multi-electrode planner**  
Jointly choose stimulation and recording addresses.

---

# Claim boundary

Active experiment design, system identification, optimal design, electrophysiology, synaptic mapping and dendritic parameter fitting are established fields.

This repo does **not** claim to have invented dendritic tomography or to have shown that arbitrary dendritic states are recoverable from soma recordings.

Its current specific question is:

> **Can a morphology-informed, baseline-calibrated experiment planner identify particular local dendritic changes with limited recording sites/trials while explicitly reporting nuisance, model sensitivity, and what remains ambiguous?**
