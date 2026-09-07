# Latest result — from exact-model probing to nuisance and model mismatch

The first three gates established the active-query mechanism, then Astra's audit removed two easy illusions: the symmetric six-arm result was mostly a three-bit coding problem, and the first pairwise planner was noise-blind.

Gates 4 and 5 now push the experiment toward the actual use case.

## Gate 4 — nuisance changes what a good question means

The hidden target is still one of six distal leak changes, but the specimen also has uncertain common leak, a left/right leak-bias mode, axial scale and stimulation gain.

The old exact-model Gate-3 design falls from roughly 84% in its matched world to `0.41667` localization accuracy under these continuous nuisance draws. A sensitivity planner that ignores nuisance reaches `0.40375`.

The useful quantity is not raw target sensitivity. For stimulation `p`, let

```text
J_p = target sensitivity columns
N_p = nuisance sensitivity columns
```

and use

```text
Sigma_p = sigma^2 I + N_p Lambda N_p^T.
```

Whitening the target signatures by this nuisance covariance before choosing stimulation gives `0.42750` accuracy. The mean over 200 matched-budget random sets is `0.38688`.

An exhaustive audit of all 1140 three-pattern sets places the nuisance-aware greedy set at rank `35/1140`, about the 97th percentile. It is not globally optimal.

The important handoff to Operaattori is therefore:

> **Do not maximize a geometry derivative merely because it is large. Choose stimulation for which the target derivative is difficult for the nuisance derivative subspace to imitate.**

## Gate 5 — model mismatch, then paired baseline

Gate 4 still let the planner and evaluator share a model family. Gate 5 breaks that.

The planner remains the reduced two-compartment-per-arm material. The synthetic experimental generator has:

- three compartments per arm;
- unmodelled 10% capacitance heterogeneity;
- a fixed unmodelled branch-specific distal-leak offset with draw scale `0.04`;
- the same continuous nuisance variables as Gate 4.

The hidden intervention is still a `+0.06` distal leak on one branch.

Using only **post-change absolute soma traces**, localization degrades strongly:

```text
Gate-4 nuisance-aware stimulation set  0.37875
Gate-3 exact-model set                  0.28792
noise-only sensitivity set              0.27313
random-set mean                         0.31258
```

The static model error is now large enough to imitate the target.

But the proposed application naturally has another measurement: record the same known neuron before and after the intervention.

Define

```text
delta y = y_after - y_before.
```

The baseline and post recordings are both charged independent measurement noise, so the difference uses `sqrt(2) * sigma`, not a free noiseless baseline.

Even with that cost, paired differencing recovers much of the lost identifiability:

```text
Gate-4 nuisance-aware set, paired       0.42813
Gate-3 set, paired                       0.40396
noise-only set, paired                   0.42667
random-set mean, paired                  0.39037
```

A stimulation set designed only from the reduced model's paired responses lands at `0.42625` on the richer generator and rank `46/1140` (96th percentile) in the exhaustive equal-budget audit.

It is **not** the true best set. In fact the Gate-4 set slightly outruns it. That is useful: baseline subtraction cancels static model error, but model mismatch can still reorder the stimulation ranking.

So the current instrument-shaped principle is becoming:

```text
known morphology/model
       +
baseline measurement
       +
target and nuisance sensitivities
       +
active stimulation design
       ->
which hidden changes remain identifiable?
```

## What this changes for the real Operaattori bridge

The real-cell experiment should probably begin as a **change-identification** problem, not absolute parameter reconstruction.

For an imaged cell:

1. record a baseline response panel;
2. define candidate local geometry/conductance changes;
3. compute target and nuisance tangents from the reduced Operaattori compiler;
4. choose stimulation that separates target directions after nuisance whitening;
5. apply/receive the post-intervention trace;
6. infer from paired change traces where possible;
7. report blind or model-sensitive directions rather than forcing an estimate.

The paired protocol costs measurements, so the eventual planner should be allowed to choose whether a baseline repeat, a new stimulation address, or another recording site is the most valuable next trial.

## Next gate

**Gate 6: sequential/adaptive experiment selection.**

The current designs are fixed sets chosen before any data arrive. The next observer should choose trial 2 after seeing trial 1, and trial 3 after seeing trials 1-2. Its state is a posterior/ambiguity distribution over branch target and nuisance states.

The hard comparison is not adaptive versus a weak random baseline. It is:

- best fixed three-trial set under the same model/budget;
- adaptive three-trial policy;
- both under the richer mismatched generator;
- paired baseline noise and trial cost counted explicitly.

Only after that should the protocol be transplanted onto the real Operaattori morphology and its audited `length` / `diameter` tangents.
