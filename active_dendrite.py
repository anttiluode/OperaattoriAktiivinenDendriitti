"""Core machinery for the first Active Dendritic Identification gates.

This is deliberately a small passive linear compartment model. It is not a
biophysical replacement for Operaattori. The point is to isolate the inverse
problem:

    known morphology + hidden local change + chosen stimulation + soma readout

and test whether active stimulation makes a hidden change identifiable.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import numpy as np


@dataclass(frozen=True)
class PassiveNetwork:
    """Implicit-Euler passive network x[t+1] = A (x[t] + dt B u[t])."""

    A: np.ndarray
    dt: float
    soma: int
    stimulation_nodes: tuple[int, ...]

    def response(
        self,
        amplitudes: np.ndarray,
        *,
        steps: int = 80,
        pulse_steps: int = 4,
    ) -> np.ndarray:
        amplitudes = np.asarray(amplitudes, dtype=float)
        if amplitudes.shape != (len(self.stimulation_nodes),):
            raise ValueError("amplitudes must match stimulation_nodes")
        x = np.zeros(self.A.shape[0], dtype=float)
        y = np.zeros(steps, dtype=float)
        drive = np.zeros_like(x)
        for amp, node in zip(amplitudes, self.stimulation_nodes):
            drive[int(node)] = float(amp)
        for t in range(steps):
            u = drive if t < pulse_steps else 0.0 * drive
            x = self.A @ (x + self.dt * u)
            y[t] = x[self.soma]
        return y


def compile_passive(
    n: int,
    edges: list[tuple[int, int, float]],
    leak: np.ndarray,
    *,
    stimulation_nodes: tuple[int, ...],
    soma: int = 0,
    dt: float = 0.25,
) -> PassiveNetwork:
    """Compile a resistor/leak graph to one stable implicit step."""
    leak = np.asarray(leak, dtype=float)
    if leak.shape != (n,):
        raise ValueError("leak shape mismatch")
    G = np.zeros((n, n), dtype=float)
    for i, j, g in edges:
        i, j, g = int(i), int(j), float(g)
        G[i, i] += g
        G[j, j] += g
        G[i, j] -= g
        G[j, i] -= g
    K = G + np.diag(leak)
    A = np.linalg.inv(np.eye(n) + float(dt) * K)
    return PassiveNetwork(
        A=A,
        dt=float(dt),
        soma=int(soma),
        stimulation_nodes=tuple(int(x) for x in stimulation_nodes),
    )


def symmetric_y(*, changed_branch: str | None = None, delta_leak: float = 0.18) -> PassiveNetwork:
    """Five-node mirror-symmetric Y tree: soma, two proximal, two distal."""
    # 0 soma; 1/2 left prox/dist; 3/4 right prox/dist
    edges = [
        (0, 1, 0.8),
        (1, 2, 0.6),
        (0, 3, 0.8),
        (3, 4, 0.6),
    ]
    leak = np.array([0.30, 0.22, 0.18, 0.22, 0.18], dtype=float)
    if changed_branch == "A":
        leak[2] += float(delta_leak)
    elif changed_branch == "B":
        leak[4] += float(delta_leak)
    elif changed_branch is not None:
        raise ValueError("changed_branch must be A, B, or None")
    return compile_passive(
        5,
        edges,
        leak,
        stimulation_nodes=(2, 4),
    )


def six_arm(*, changed_region: int | None = None, delta_leak: float = 0.12) -> PassiveNetwork:
    """Thirteen-node soma + six equal two-compartment arms."""
    n = 13
    edges: list[tuple[int, int, float]] = []
    leak = np.zeros(n, dtype=float)
    leak[0] = 0.35
    stimulation_nodes = []
    for b in range(6):
        prox = 1 + 2 * b
        dist = 2 + 2 * b
        stimulation_nodes.append(dist)
        leak[prox] = 0.22
        leak[dist] = 0.18
        edges.extend([(0, prox, 0.75), (prox, dist, 0.55)])
    if changed_region is not None:
        if not 0 <= int(changed_region) < 6:
            raise ValueError("changed_region outside 0..5")
        leak[2 + 2 * int(changed_region)] += float(delta_leak)
    return compile_passive(
        n,
        edges,
        leak,
        stimulation_nodes=tuple(stimulation_nodes),
    )


def unit_energy_subset(width: int, subset: tuple[int, ...] | list[int] | set[int]) -> np.ndarray:
    """Positive, simultaneously realizable unit-L2 stimulation."""
    subset = tuple(sorted(int(x) for x in subset))
    if not subset:
        raise ValueError("empty subset")
    p = np.zeros(width, dtype=float)
    p[list(subset)] = 1.0 / np.sqrt(len(subset))
    return p


def response_bank(
    models: list[PassiveNetwork],
    probes: list[np.ndarray],
    *,
    steps: int = 70,
    pulse_steps: int = 4,
) -> np.ndarray:
    """Return [hypothesis, probe, time] soma templates."""
    return np.stack(
        [
            np.stack(
                [
                    model.response(p, steps=steps, pulse_steps=pulse_steps)
                    for p in probes
                ],
                axis=0,
            )
            for model in models
        ],
        axis=0,
    )


def nearest_template(y: np.ndarray, templates: np.ndarray) -> int:
    """Euclidean ML classifier for equal isotropic Gaussian noise."""
    d = np.sum((templates - y[None, :]) ** 2, axis=1)
    return int(np.argmin(d))


def monte_carlo_localization(
    bank: np.ndarray,
    *,
    noise_std: float,
    trials: int,
    seed: int,
) -> float:
    """Classification accuracy using concatenated soma traces."""
    templates = bank.reshape(bank.shape[0], -1)
    rng = np.random.default_rng(seed)
    correct = 0
    for _ in range(int(trials)):
        h = int(rng.integers(0, templates.shape[0]))
        y = templates[h] + rng.normal(
            0.0, float(noise_std), size=templates.shape[1]
        )
        correct += nearest_template(y, templates) == h
    return correct / int(trials)


def unique_probe_codes(probes: list[np.ndarray]) -> int:
    """How many hypotheses have a unique on/off membership code."""
    matrix = np.stack([p != 0.0 for p in probes], axis=0)
    return len({tuple(matrix[:, j].tolist()) for j in range(matrix.shape[1])})


def random_balanced_probe_sets(
    *,
    branches: int = 6,
    subset_size: int = 3,
    probes_per_set: int = 3,
    count: int = 100,
    seed: int = 4,
) -> list[list[np.ndarray]]:
    choices = list(itertools.combinations(range(branches), subset_size))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        sets = [
            choices[int(rng.integers(0, len(choices)))]
            for _ in range(probes_per_set)
        ]
        out.append([unit_energy_subset(branches, s) for s in sets])
    return out
