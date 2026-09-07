"""Gate 0: exact symmetry blindness, then active branch contrast.

Two mirror-related hidden states differ only in whether local leak increased on
branch A or B. A symmetric positive stimulation of both branches gives exactly
the same soma trace. Positive stimulation of one branch breaks the symmetry.

For a physically realizable "differential" measurement we use TWO positive
trials (stimulate A alone; stimulate B alone) and subtract the recorded soma
traces. We do not assume a negative excitatory current is realizable.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from active_dendrite import symmetric_y


def main() -> None:
    h_ab = symmetric_y(changed_branch="A")
    h_ba = symmetric_y(changed_branch="B")

    p_a = np.array([1.0, 0.0])
    p_b = np.array([0.0, 1.0])
    p_common = np.array([1.0, 1.0]) / np.sqrt(2.0)

    y_common_a = h_ab.response(p_common)
    y_common_b = h_ba.response(p_common)
    y_a_a = h_ab.response(p_a)
    y_a_b = h_ba.response(p_a)
    y_b_a = h_ab.response(p_b)
    y_b_b = h_ba.response(p_b)

    common_delta = y_common_a - y_common_b
    a_delta = y_a_a - y_a_b

    contrast_h_a = y_a_a - y_b_a
    contrast_h_b = y_a_b - y_b_b

    sigma = 0.003
    d2_common = float(common_delta @ common_delta / sigma**2)
    d2_a = float(a_delta @ a_delta / sigma**2)
    d2_two_trial_contrast = float(
        ((contrast_h_a - contrast_h_b) @ (contrast_h_a - contrast_h_b))
        / (2.0 * sigma**2)
    )

    result = {
        "gate": "symmetry_blindness_and_active_contrast",
        "model": "five-node mirror-symmetric passive Y tree",
        "hidden_hypotheses": {
            "H_A": "extra distal leak on branch A",
            "H_B": "same extra distal leak on mirror branch B",
        },
        "soma_only": True,
        "noise_std_for_D2": sigma,
        "common_positive_probe": p_common.tolist(),
        "max_abs_common_trace_difference": float(np.max(np.abs(common_delta))),
        "common_probe_D2": d2_common,
        "branch_A_positive_probe": p_a.tolist(),
        "branch_A_trace_difference_l2": float(np.linalg.norm(a_delta)),
        "branch_A_probe_D2": d2_a,
        "two_trial_physical_contrast": "soma(A-positive trial) - soma(B-positive trial)",
        "contrast_H_A_peak": float(contrast_h_a[np.argmax(np.abs(contrast_h_a))]),
        "contrast_H_B_peak": float(contrast_h_b[np.argmax(np.abs(contrast_h_b))]),
        "mirror_contrast_max_error": float(np.max(np.abs(contrast_h_a + contrast_h_b))),
        "two_trial_contrast_separation_l2": float(np.linalg.norm(contrast_h_a - contrast_h_b)),
        "two_trial_contrast_D2_with_independent_noise": d2_two_trial_contrast,
        "interpretation": (
            "A soma trace under common-mode stimulation is exactly blind to which "
            "mirror branch changed. Breaking the input symmetry makes the hidden "
            "branch state observable from the same single soma recording site. "
            "A differential branch contrast is implemented as two positive trials, "
            "not as an assumed signed excitatory stimulus."
        ),
    }

    assert result["max_abs_common_trace_difference"] < 1e-14
    assert result["branch_A_trace_difference_l2"] > 0.08
    assert result["mirror_contrast_max_error"] < 1e-14
    assert result["two_trial_contrast_separation_l2"] > 0.16
    assert result["common_probe_D2"] < 1e-20
    assert result["branch_A_probe_D2"] > 700.0

    out = Path("results/gate0_symmetry.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
