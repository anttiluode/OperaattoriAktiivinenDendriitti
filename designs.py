"""Versioned stimulation designs carried between gates.

Keep these in one place so later attackers do not silently copy the wrong
probe indices. The names describe where each set was actually selected.
"""

GATE3_MULTI_HYPOTHESIS = (18, 17, 11)
GATE4_NUISANCE_AWARE = (10, 2, 6)
GATE4_NOISE_ONLY = (18, 0, 13)
GATE5_REDUCED_PAIRED = (4, 11, 16)


def as_list(design):
    return [int(x) for x in design]
