# OperaattoriAktiivinenDendriitti

**Active Dendritic Identification / Dendritic State Tomography**

This repo asks a narrower and more practical question than "reconstruct the whole neuron":

> **Given a known morphology, a small set of plausible hidden dendritic changes, and only one or a few recording sites, which stimulation should we apply next to make those hidden states distinguishable?**

It grows directly out of two earlier results:

1. `SighImageSuper`: a hidden distinction can physically exist yet remain invisible under one probe/readout geometry; changing the question can open it.
2. `Operaattori`: morphology can be compiled to an electrical operator, and soma responses already have verified sensitivities to local geometry.

The target tool is a **stimulation planner + uncertainty map** for a known dendritic morphology.

This is not yet experimental tomography and it does not claim that arbitrary synaptic distributions are recoverable from a soma trace. The first gates deliberately isolate identifiability in tiny passive systems before connecting to the full Operaattori compiler.

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

For many candidate states, the planner should choose a small set of physically realizable stimulation trials that breaks as many ambiguities as possible.

The long-term version is sequential:

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
H_B: the same change on mirror branch B
```

The only recording site is the soma.

A symmetric positive stimulus to both branches gives soma traces that are identical to numerical precision:

```text
max |y_A - y_B| = 4.16e-17
D^2 common mode = 1.22e-27
```

The memory/state is not absent. The measurement is blind.

Stimulating branch A alone breaks the symmetry:

```text
||y_A - y_B||_2 = 0.08542
D^2 = 810.7   (for sigma = 0.003)
```

A physically realizable differential contrast uses **two positive trials**:

```text
stimulate A -> record soma
stimulate B -> record soma
subtract the recordings
```

The hidden branch states then give opposite contrasts, with separation `0.17084`.

This is the direct bridge from the Sigh active-query result:

> **same recording site, different question, previously invisible state becomes observable.**

---

## Gate 1 — six hidden regions, one soma, three designed trials

`gate1_six_region_localization.py`

A symmetric passive model has six equal distal regions. Exactly one has increased leak.

There is still only **one soma recording site**.

A hand-transparent three-trial design stimulates these equal-energy positive branch groups:

```text
(0, 1, 2)
(0, 3, 4)
(1, 3, 5)
```

Each branch receives a unique three-bit membership code.

At soma-noise `sigma = 0.003`, 5000 Monte Carlo trials give:

| protocol | trials | localization accuracy |
|---|---:|---:|
| first 3 individual branch probes | 3 | 0.6736 |
| random balanced 3-pattern design, mean over 100 designs | 3 | 0.8444 |
| **designed balanced patterns** | **3** | **0.9922** |
| all individual branches | 6 | 1.0000 |

This is still a toy. But it establishes the core application: **multi-site stimulation can trade stimulation design for fewer recording trials.**

---

## Gate 2 — the computer chooses the stimulation set

`gate2_greedy_planner.py`

Gate 1 was hand-designed. Gate 2 searches all 20 positive unit-energy `3-of-6` stimulation patterns.

The planner greedily prefers probes that split previously aliased candidate models before merely increasing already-large separations.

It chooses:

```text
trial 1: (3, 4, 5)  -> 9 / 15 hypothesis pairs separated
trial 2: (1, 2, 4)  -> 13 / 15
trial 3: (0, 2, 3)  -> 15 / 15
```

With the same soma noise and a three-trial budget:

```text
localization accuracy = 0.9920
```

That is the first reusable piece of the intended instrument:

> **given candidate hidden states and candidate stimulation protocols, select a small protocol set that makes the hidden states distinguishable at the available sensor.**

---

## Why "A - B" needs care

A mathematical signed stimulation vector is useful for analysis, but an excitatory synaptic experiment may not be able to inject a negative current at will.

Therefore this repo distinguishes:

```text
mathematical contrast:  A - B
physical protocol:      positive A trial and positive B trial, then compare outputs
```

For nonlinear dendrites, those are not interchangeable. The full planner must optimize realizable experiments, not elegant vectors that the hardware cannot deliver.

---

## What counts as success

There are three increasingly difficult targets.

### Distinguish

Which of a finite set of candidate hidden changes occurred?

Gates 0-2 start here.

### Localize

Which branch or region changed?

Gate 1 already gives a toy localization receipt.

### Estimate

Recover several changes and their magnitudes with uncertainty under nuisance parameters and model mismatch.

This is the serious tomography problem and comes later.

---

## Direct bridge to `Operaattori`

The existing `Operaattori` code already has most of the forward-model machinery we need.

In particular, its `audits/real_metric_tangent.py` exposes:

```text
build_compartment_graph(...)
metric_tangent(...)
simulate(...)
simulate_with_metric_tangents(...)
```

and returns soma response tangents for local geometry directions.

That suggests the first real-morphology planner:

1. load the known reconstructed cell;
2. choose six candidate branch regions;
3. define candidate local changes;
4. define physically realizable stimulation protocols;
5. simulate the soma waveform under every `(hidden state, protocol)` pair;
6. score response separability relative to noise;
7. choose the next protocol;
8. hide the true state and test localization.

See [`OPERAATTORI_BRIDGE.md`](OPERAATTORI_BRIDGE.md).

Important: the current Operaattori tangent machinery is primarily for **geometry** (`length`, `diameter`, and the pose null). Synaptic-strength or local-channel inference needs additional parameter derivatives or controlled finite differences. We should not silently call geometry derivatives synaptic tomography.

---

## Attackers required before calling this useful

The synthetic gates are intentionally easy. The real project has to survive:

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

A good instrument must say **unidentifiable** when the data do not constrain a direction.

---

## Run

```bash
python gate0_symmetry_receipt.py
python gate1_six_region_localization.py
python gate2_greedy_planner.py
```

Only NumPy is required.

---

## Roadmap

**Gate 3 — adaptive hypothesis elimination**  
Choose the *next* stimulus from the posterior/remaining ambiguity rather than designing the whole set offline.

**Gate 4 — nuisance parameters**  
Hide branch change plus uncertain leak/capacitance/stimulation scale. Require calibrated uncertainty.

**Gate 5 — model mismatch**  
Generate "experimental" traces with a richer simulator and infer with the reduced model.

**Gate 6 — real Operaattori morphology**  
Known cell, six branch regions, soma-only recording, geometry changes first because verified tangents already exist.

**Gate 7 — synaptic/local conductance parameters**  
Add explicit derivatives or controlled finite differences for the parameters actually being inferred.

**Gate 8 — nonlinear operating-point design**  
Choose stimulation location, timing **and strength** because sensitivity rotates with drive.

**Gate 9 — sparse multi-electrode planner**  
Jointly choose stimulation and recording addresses under a budget.

---

## Claim boundary

Active experiment design, system identification, optimal design, electrophysiology, synaptic mapping and dendritic parameter fitting are established fields.

This repo does **not** claim to have invented dendritic tomography or to have shown that arbitrary dendritic states are recoverable from soma recordings.

The specific question is:

> **Can a morphology-informed stimulation planner identify particular hidden dendritic changes with fewer recording sites or fewer trials than ordinary stimulation, while explicitly reporting what remains ambiguous?**

That is testable.
