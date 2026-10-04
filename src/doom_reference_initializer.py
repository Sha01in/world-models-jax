"""Read a frozen controller selected from complete prior validation."""

import numpy as np

from src.doom_reference_comparison import independent_frozen_selection
from src.doom_reference_comparison_v2 import archive_info, digest, identity, read


def registered_initializer(protocol):
    """Separate search initialization from public controls and world templates."""
    selected_path = protocol["initializer_selection_frozen"]
    audit_path = protocol["initializer_selection_cpu_audit"]
    for path in (selected_path, audit_path):
        if digest(path) != protocol["frozen_inputs"][path]:
            raise ValueError("Initializer selection evidence changed")
    selection, audit = read(selected_path), read(audit_path)
    parent = read(protocol["initializer_parent_protocol"])
    if digest(protocol["initializer_parent_protocol"]) != selection["protocol_sha256"]:
        raise ValueError("Initializer parent protocol changed")
    # Reconstruct all complete parent validation cohorts and their raw means.
    # Confidence intervals are unnecessary for reproducing the prior choice.
    reconstructed = independent_frozen_selection(parent, selection, resamples=32)
    selected = selection["selected"]
    if (
        audit["selected"] != selected
        or reconstructed["selected"] != selected
        or not audit["validation_selection_recomputed"]
        or not audit["all_validation_records_verified"]
        or selected["controller"] != protocol["initializer_controller"]
        or selected["parameters_sha256"] != protocol["initializer_parameters_sha256"]
    ):
        raise ValueError("Initializer is not the complete validation-selected policy")
    if selected["controller"] is None:
        path = protocol["arguments"]["reference_dir"] + "/controller.json"
        initial = np.asarray(read(path)[0], dtype=np.float64)
    else:
        initial = archive_info(selected["controller"], protocol)["params"]
    if identity(initial) != protocol["initializer_parameters_sha256"]:
        raise ValueError("Registered initializer raw weights changed")
    return initial
