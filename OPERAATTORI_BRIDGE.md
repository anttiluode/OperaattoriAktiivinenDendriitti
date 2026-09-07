# Bridge to Operaattori

This note is the implementation plan for replacing the synthetic passive trees with the existing `anttiluode/Operaattori` morphology compiler.

## What Operaattori already gives us

The current audited path in `Operaattori/audits/real_metric_tangent.py` already constructs the direct cable graph and supports:

```python
build_compartment_graph(cell)
node_for_section_x(graph, section, x)
metric_tangent(graph, node_index, kind)
simulate(G, C, site_nodes, soma_node, ga, gn)
simulate_with_metric_tangents(
    G, C, site_nodes, soma_node, ga, gn, directions
)
```

The tangent simulator returns

```text
soma_mV
soma_tangent_mV_per_logscale
```

for specified local metric directions.

That is nearly the Jacobian object needed by an experiment designer.

## First real-morphology target

Do not begin by reconstructing thousands of unknown parameters.

Start with:

```text
known morphology
6 candidate branch regions
1 hidden local change
known stimulation command
soma-only recording
```

Use geometry first:

```text
hypothesis h = one region's length or diameter changed by a fixed log-scale
```

because the derivative path already exists and is audited.

For every stimulation protocol `p`, construct the waveform Jacobian

```math
J_p(t,i) = d V_soma(t; theta, p) / d theta_i.
```

Columns that are nearly parallel are difficult to distinguish.

A first design score can use noise-whitened column separation:

```math
D_ij^2(p) = (J_p[:,i]-J_p[:,j])^T R^-1 (J_p[:,i]-J_p[:,j]).
```

Then choose the realizable stimulation protocol that maximizes, for example,

```text
minimum pairwise D_ij
```

or expected information gain under a declared prior.

## Why drive strength belongs in the action

Operaattori's existing operating-point audit shows that geometry-gradient signs can change with drive.

Therefore the action should eventually be

```text
p = (location, timing, conductance program, strength)
```

rather than only a branch address.

The planner should pay a cost for strong drive because it can recruit additional nonlinear mechanisms and change the state being measured.

## Synaptic parameters are a separate gate

The current metric tangent differentiates local geometry.

That does **not** automatically give

```text
d soma / d synaptic efficacy
d soma / d ion-channel density
```

For synaptic tomography, add those derivatives explicitly or use finite differences as an initial receipt.

Also remember: an inactive synapse is invisible to a protocol that never activates that pathway.

## Physical stimulation constraints

Do not assume a mathematical signed vector is realizable.

If the mathematical optimum is `A-B`, the physical protocol may need:

```text
trial 1: stimulate A positively
trial 2: stimulate B positively
measurement: compare the two soma traces
```

In a nonlinear model, the difference of two separate trials is not the same experiment as simultaneous signed injection.

The planner therefore needs a protocol generator describing the actual experimental control set.

## Blind directions are outputs, not failures

Operaattori's pose/bending null is exactly the kind of result a tomography tool must surface.

If two candidate parameter changes yield the same recordings for every allowed stimulation, the correct output is:

```text
UNIDENTIFIABLE UNDER CURRENT PROBES / SENSORS
```

not a forced estimate.

## Required attacker

The most important realism test is model mismatch:

```text
data generator: full/richer simulator
planner/inference: reduced Operaattori model
```

If both planner and hidden data use the exact same equations and parameters, the inverse problem can look much easier than a laboratory problem.

## Deliverable

The first serious tool should emit:

```text
recommended next stimulation
predicted response separation
posterior / confidence over candidate regions
blind or weakly constrained directions
expected value of an additional trial
```

That would turn Operaattori from a forward morphology compiler into an active experiment-design instrument.
