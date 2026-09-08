# Response to `operaattori_review_080926`

The independent review materially changes the interpretation of Gates 3–5. The corrections are accepted rather than explained away.

## 1. Gate-3 comparator bug

The actual Gate-3 multi-hypothesis design is

```text
indices  [18, 17, 11]
subsets  (2,4,5), (2,3,5), (1,2,4)
```

Later scripts accidentally hard-coded `[18,10,7]` and called it the Gate-3 design.

The review reproduced the corrected comparisons:

```text
Gate 4 nuisance evaluation
actual Gate-3 design       0.37375
wrong hard-coded design    0.41667
Gate-4 nuisance design     0.42750

Gate 5 richer generator
                         absolute    paired
actual Gate-3 design      0.37292    0.41729
wrong hard-coded design   0.28792    0.40396
Gate-4 nuisance design    0.37875    0.42813
```

`designs.py` now centralizes the audited design indices so new gates do not copy them by hand.

The old Gate-4/5 result files remain historical receipts for the revisions that produced them. New work must use the centralized audited constants.

## 2. Sampling confound

Gate 3 retained 70 waveform samples per stimulation; Gate 4 retained 14 (`0,5,...,65`) while keeping independent per-sample noise fixed.

The review showed, on the same zero-nuisance neuron and actual Gate-3 probes:

```text
70 samples/probe   0.841875
14 samples/probe   0.484625
```

So the old sentence “84% collapses to 37% because of nuisance” was wrong. A large part of the drop was caused by reducing the measurement itself.

From Gate 6 onward the compared methods use the same 14-sample observation protocol. Sampling density, duration, filtering and noise covariance must be held fixed inside a comparison.

## 3. Stronger random control

Random designs must be stratified by whether they actually give the six branches distinct stimulation codes.

The review found:

```text
Gate 3   selected 0.8419    six-code random mean 0.8302
Gate 4   selected 0.4275    six-code random mean 0.3965
Gate 5 paired, Gate-4 set 0.4281    six-code random mean 0.4050
```

Gate 3 therefore retains only a small design advantage after the coding control. Gate 4 retains a more useful response-geometry advantage.

## 4. Exact-blindness estimator bug

`gaussian_localization_accuracy_estimate` previously used a strict discriminant inequality for exact duplicate templates. Six identical templates therefore returned zero instead of chance `1/6`.

The estimator now groups exact/near-exact template equivalence classes and applies an explicit uniform tie rule. Deliberately unidentifiable controls can therefore report chance rather than a numerical artifact.

## 5. What Gate 5 actually established

The paired before/after measurement is the strongest Gate-5 mechanism.

The nuisance-aware Gate-4 set reached `0.428125` paired accuracy, but the simpler noise-only set reached `0.426667`. That receipt does **not** establish that the sophisticated selector caused the recovery. Much of the recovery comes from measuring a change against the same cell's baseline.

The baseline is therefore treated as a real resource, not a free subtraction.

The review's after-only gain attacker also matters:

```text
post-only log gain drift std
0.00   Gate-4 paired 0.42813
0.01   Gate-4 paired 0.42042
0.05   Gate-4 paired 0.36104
```

A changed measurement apparatus can imitate a changed neuron.

## 6. Chronology: adaptive probing cannot invent a baseline

After the hidden change occurs, an adaptive planner cannot decide on a new stimulation and then request that stimulation's pre-change response.

A legal adaptive protocol must therefore do one of the following:

1. acquire the candidate baseline panel before the change;
2. restrict post-change choices to probes that were baselined;
3. predict missing baselines with declared uncertainty and correct covariance accounting.

Gate 6 uses option 1 deliberately. It is expensive but clean.

---

# Gate 6 — same baseline memory, same budget, fixed vs adaptive

`gate6_fixed_vs_adaptive_baseline_panel.py`

Every strategy receives:

```text
20 pre-change baseline recordings
 3 post-change recordings
14 samples per waveform
 1 soma sensor
```

The 23-recording total is stated explicitly.

The hidden target has seven classes:

```text
unchanged
branch 0 changed
...
branch 5 changed
```

The target magnitude is unknown (`0.04..0.08`). The specimen has shared uncertain leak, split bias, axial scale and pre-gain, plus **5% log-scale after-only gain drift**.

The evaluator is richer than the inference model. Every synthetic specimen gets a new held-out three-compartment morphology mismatch: capacitance heterogeneity and static branch leak offsets. The reduced observer carries one joint particle belief over target and nuisance state across all three post-change trials.

The adaptive policy is causal. It sees its current posterior and the stored baseline panel, then scores unused probes by predicted between-class separation relative to within-class nuisance spread and noise. It never sees the hidden class or rich-generator parameters.

## Gate-6 receipt

On 1,050 held-out specimens (`150` per class):

| strategy | top-1 | top-2 | unchanged recall |
|---|---:|---:|---:|
| Gate-4 nuisance-aware fixed | **0.36286** | **0.59333** | 0.46667 |
| first causal adaptive policy | 0.32857 | 0.55238 | **0.50667** |
| Gate-5 reduced paired fixed | 0.32952 | 0.53429 | 0.36667 |
| actual Gate-3 fixed | 0.30952 | 0.51714 | 0.36667 |

Forty random fixed sets were sampled **only from the 480 designs that give all six branches distinct codes**:

```text
mean    0.32081
median  0.31810
p90     0.34600
max     0.36190
```

The Gate-4 fixed set is `+0.04205` above that controlled random mean.

The first adaptive rule is `-0.03429` below the strong Gate-4 fixed set. That is a useful negative result, not a reason to tune until adaptive wins.

> **A reliable fixed morphology-informed stimulation panel is already worth transferring to the real Operaattori morphology. Adaptive superiority remains a separate hypothesis.**

## Ambiguity is part of the output

Gate 6 also records posterior probability, Brier score, log loss, calibration error, top-2 accuracy, unchanged recall and abstention curves.

The forced-choice accuracy is still low enough that a practical interface must expose ambiguity. The intended instrument is a change detector + stimulation recommender + uncertainty map, not a branch-label oracle.

---

## Next transfer

The next major gate should use the audited real Operaattori morphology rather than making this six-arm toy ever more elaborate.

Start with the parameters already supported by audited tangents:

```text
local length change
local diameter change
pose null as a negative control
```

Freeze the Gate-6 measurement accounting:

- declared baseline panel;
- same number of post-change trials;
- same soma observation window;
- explicit nuisance directions;
- unchanged-cell class;
- calibration drift;
- held-out mismatch where feasible;
- posterior/ambiguity output;
- strong fixed stimulation set first;
- adaptive selection as an additional comparison, not a prerequisite.

The central question is now operational:

> **Given a known cell and a stored baseline, which physically realizable stimulation makes a plausible local change distinguishable from nuisance and model error at the available recording sites?**
