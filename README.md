# OperaattoriAktiivinenDendriitti

**Active Dendritic Identification / Dendritic State Tomography**

This repo asks a narrower and more practical question than "reconstruct the whole neuron":

> **Given a known morphology, a small set of plausible hidden dendritic changes, and only one or a few recording sites, which stimulation should we apply next to make those hidden states distinguishable?**

It grows directly out of two earlier results:

1. `SighImageSuper`: a hidden distinction can physically exist yet remain invisible under one probe/readout geometry; changing the question can open it.
2. `Operaattori`: morphology can be compiled to an electrical operator, and soma responses already have verified sensitivities to local geometry.

The target tool is a **stimulation planner + uncertainty map** for a known dendritic morphology.

This is not yet experimental tomography and it does not claim that arbitrary synaptic distributions are recoverable from a soma trace. The early toy gates are being attacked before the project is allowed to move onto the real Operaattori morphology.

---

## The inverse problem

Let

- `theta` = hidden dendritic parameters,
- `p` = a stimulation protocol,
- `F(theta, p)` = the forward dendritic model,
- `C` = the available recording operator.

The experimenter sees

```text
y = C F(theta, p) + noise
```

and may choose the next `p`.

For two candidate hidden states, a useful probe makes their predicted recordings separate relative to noise:

```text
D^2(p) = (mu_1(p) - mu_2(p))^T R^-1 (mu_1(p) - mu_2(p))
```

For many candidate states the design objective must be genuinely multi-hypothesis: pairwise nonzero differences are not enough.

The long-term workflow is sequential:

```text
known morphology
      |
candidate hidden changes
      |
choose stimulation
      v
record soma / sparse sites
      |
update uncertainty
      |
choose next stimulation
```

---

## Gate 0 — exact soma blindness caused by symmetry

`gate0_symmetry_receipt.py`

A five-node mirror-symmetric Y tree has two possible hidden changes:

```text
H_A: extra distal leak on branch A
H_B: the same extra leak on mirror branch B
```

The only recording site is the soma.

A symmetric positive stimulus to both branches gives soma traces identical to numerical precision:

```text
max |y_A - y_B| = 4.16e-17
D^2 common mode = 1.22e-27
```

Stimulating branch A alone breaks the symmetry:

```text
||y_A - y_B||_2 = 0.08542
D^2 = 810.7   (sigma = 0.003)
```

A physically realizable differential contrast uses **two positive trials**:

```text
stimulate A -> record soma
stimulate B -> record soma
subtract the recordings
```

The hidden branch states then give opposite contrasts.

> **Same recording site, different question, previously invisible state becomes observable.**

That is the direct Sigh -> dendrite bridge.

---

## Gate 1 — six hidden regions, one soma, three trials

`gate1_six_region_localization.py`

A perfectly symmetric passive model has six equal distal regions. Exactly one has increased leak. There is still only **one soma recording site**.

A transparent three-trial pattern set is

```text
(0, 1, 2)
(0, 3, 4)
(1, 3, 5)
```

which gives every branch a unique three-bit membership code.

The original result was:

| protocol | trials | localization accuracy |
|---|---:|---:|
| first 3 individual branch probes | 3 | 0.6736 |
| random balanced 3-pattern design, mean over 100 | 3 | 0.8444 |
| designed balanced patterns | 3 | 0.9922 |
| all individual branches | 6 | 1.0000 |

That looked stronger than it was.

### Audit correction

`gate2a_planner_audit.py` conditions the random comparison on whether the random design also gives all six branches unique codes.

Of the original 100 random designs:

```text
42 had six unique branch codes
mean accuracy of those 42 = 0.991576
hand-designed accuracy     = 0.992200
```

The gap is only `0.000624` in accuracy.

Exhaustively, among all `1140` unordered triples of distinct `3-of-6` patterns, `480` give six unique codes — and in the exactly symmetric toy **all 480 have the same pairwise response-distance spectrum to 1e-12 rounding**.

So the correct interpretation is:

> **Gate 1 establishes that coded multi-site stimulation can localize six symmetric hidden states in three soma-only trials. It does not establish that our particular design rule is electrophysiologically superior to other code-valid designs.**

This correction matters because the real project must exploit response geometry, not merely assign binary labels to symmetric arms.

---

## Gate 2 — scoring flaw and repaired pairwise planner

`gate2_greedy_planner.py`  
`gate2a_planner_audit.py`

The first planner had another real flaw. It counted every numerically nonzero template difference as a "separated" hypothesis pair.

A synthetic attacker makes the failure exact:

```text
probe 0: separates all 3 pairs only by ~1e-6 under noise sigma=1e-2
probe 1: leaves one pair aliased but separates the other two by 0.1
```

The legacy scorer chooses **probe 0** because `3 nonzero pairs > 2 nonzero pairs`.

The repaired scorer declares a pair resolved only when its predicted Gaussian pair error is below a stated tolerance and otherwise uses noise-scaled confusion/Mahalanobis separation. It chooses **probe 1**.

On the perfectly symmetric six-arm problem the repaired planner still finds a valid three-pattern code, but that remains mostly a combinatorial result. Gate 3 is the more important test.

---

## Gate 3 — break exact symmetry

`gate3_broken_symmetry.py`

The six arms now have fixed 10% log-scale heterogeneity in axial conductance and passive leak. The hidden change is smaller (`delta leak = 0.06`). Exact binary-code equivalence is gone.

Every method gets the same budget: **three stimulation trials, one soma sensor, noise sigma = 0.003**.

The repaired pairwise heuristic is still not enough:

```text
pairwise-confusion greedy accuracy = 0.67296
three individual probes            = 0.66870
```

So merely fixing the sub-noise bug did **not** solve experiment design.

A second planner directly estimates the six-way nearest-template accuracy under the declared Gaussian noise, using the candidate forward models as a design-time oracle. It chooses:

```text
(2, 4, 5)
(2, 3, 5)
(1, 2, 4)
```

and obtains:

```text
design-time estimated accuracy       = 0.84094
independent discriminant check        = 0.84181
direct 50,000-trial localization     = 0.84310
```

For 200 matched-budget random distinct three-pattern designs:

```text
mean   = 0.76501
median = 0.77168
p90    = 0.84017
max    = 0.85763
```

This is the first result in the repo that is not just the exact symmetric three-bit-code trick:

> **When branch response shapes genuinely differ, choosing stimulation from the full multi-hypothesis response geometry improves localization over a random three-trial design on average.**

But it is not a victory lap. Some random sets are still better than this greedy set, and the planner knows the exact candidate models and the exact noise law. The next mandatory wall is nuisance uncertainty/model mismatch.

---

## Why `A - B` needs care

A mathematical signed stimulation vector is useful for analysis, but an excitatory synaptic experiment may not be able to inject a negative current at will.

Therefore this repo distinguishes:

```text
mathematical contrast:  A - B
physical protocol:      positive A trial and positive B trial, then compare outputs
```

For nonlinear dendrites those are not interchangeable. The full planner must optimize realizable experiments, not elegant vectors that the hardware cannot deliver.

---

## What counts as success

There are three increasingly difficult targets.

### Distinguish

Which of a finite set of candidate hidden changes occurred?

### Localize

Which branch or region changed?

### Estimate

Recover several changes and their magnitudes with calibrated uncertainty under nuisance parameters and model mismatch.

Only the first two have toy receipts so far.

---

## Direct bridge to `Operaattori`

The existing `Operaattori` code already has most of the forward-model machinery we need.

In particular, `audits/real_metric_tangent.py` exposes the machinery behind

```text
build_compartment_graph(...)
metric_tangent(...)
simulate(...)
simulate_with_metric_tangents(...)
```

and returns soma response tangents for local geometry directions.

That suggests the first real-morphology planner:

1. load the known reconstructed cell;
2. choose candidate branch regions;
3. define candidate local changes;
4. define physically realizable stimulation protocols;
5. simulate the soma waveform under every `(hidden state, protocol)` pair;
6. score multi-hypothesis localization relative to noise and nuisance uncertainty;
7. choose a protocol;
8. hide the true state and test localization;
9. compare with random, individual-site and conventional stimulation under the same trial/energy budget.

See [`OPERAATTORI_BRIDGE.md`](OPERAATTORI_BRIDGE.md).

Important: the current Operaattori tangent machinery is primarily for **geometry** (`length`, `diameter`, and the pose null). Synaptic-strength or local-channel inference needs additional parameter derivatives or controlled finite differences. Geometry derivatives must not be silently renamed synaptic tomography.

---

## Attackers required before calling this useful

The real project has to survive:

- imperfect morphology;
- uncertain passive membrane parameters;
- stimulation amplitude/calibration error;
- soma noise and filtering;
- hidden changes in more than one region;
- parameters whose soma signatures are nearly collinear;
- nonlinear NMDA/active conductances;
- probe strength changing the operating point;
- physically unrealizable signed probes;
- model mismatch between planner and data generator;
- exact blind directions such as pose changes that preserve intrinsic cable geometry.

A good instrument must sometimes answer **UNIDENTIFIABLE**.

---

## Run

```bash
python gate0_symmetry_receipt.py
python gate1_six_region_localization.py
python gate2_greedy_planner.py
python gate2a_planner_audit.py
python gate3_broken_symmetry.py
```

Only NumPy is required.

---

## Roadmap

**Gate 4 — nuisance-aware planning**  
Hide branch change behind uncertain passive leak/capacitance and stimulation calibration. The planner must marginalize over nuisance rather than pretending the nominal model is exact.

**Gate 5 — model mismatch**  
Generate synthetic "experimental" traces with a richer model and infer/design with the reduced model.

**Gate 6 — sequential/adaptive experiment selection**  
Choose the next stimulation from the remaining posterior ambiguity rather than designing a fixed set offline.

**Gate 7 — real Operaattori morphology**  
Known reconstructed cell, candidate branch regions, soma-only recording, geometry changes first because verified tangents already exist.

**Gate 8 — synaptic/local conductance parameters**  
Add explicit derivatives or controlled finite differences for the parameters actually being inferred.

**Gate 9 — nonlinear operating-point design**  
Choose stimulation location, timing **and strength** because sensitivity rotates with drive.

**Gate 10 — sparse multi-electrode planner**  
Jointly choose stimulation and recording addresses under a budget.

---

## Claim boundary

Active experiment design, system identification, optimal design, electrophysiology, synaptic mapping and dendritic parameter fitting are established fields.

This repo does **not** claim to have invented dendritic tomography or to have shown that arbitrary dendritic states are recoverable from soma recordings.

The specific question is:

> **Can a morphology-informed stimulation planner identify particular hidden dendritic changes with fewer recording sites or fewer trials than ordinary stimulation, while explicitly reporting what remains ambiguous?**

Gate 2a materially weakened the first toy claim. Gate 3 then supplied a harder positive result. That adversarial progression is the intended standard for the rest of the repo.
