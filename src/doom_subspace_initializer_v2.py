"""Initialize from a projected policy chosen by complete parent validation."""

from src.doom_subspace_comparison import (
    archive_info,
    digest,
    identity,
    independent_frozen_selection,
    public_parameters,
    read,
)


def registered_initializer(protocol):
    """Verify the parent choice without reading its reserved-test outcomes."""
    selection_path = protocol["initializer_selection_frozen"]
    audit_path = protocol["initializer_selection_cpu_audit"]
    parent_path = protocol["initializer_parent_protocol"]
    for path in (selection_path, audit_path, parent_path):
        if digest(path) != protocol["frozen_inputs"][path]:
            raise ValueError("Projected parent initialization evidence changed")
    selection, audit, parent = map(read, (selection_path, audit_path, parent_path))
    if (
        selection["protocol_sha256"] != digest(parent_path)
        or audit["protocol_sha256"] != digest(parent_path)
        or audit["frozen_selection_sha256"] != digest(selection_path)
        or parent["training_method"] != "direct_real_survival_subspace_cma"
    ):
        raise ValueError("Projected parent protocol or choice audit changed")
    reconstructed = independent_frozen_selection(parent, selection, resamples=32)
    selected = selection["selected"]
    if (
        audit["selected"] != selected
        or reconstructed["selected"] != selected
        or not audit["validation_selection_recomputed"]
        or not audit["all_validation_records_verified"]
        or not audit["complete_candidate_eligibility_reconstructed"]
        or selected["controller"] != protocol["initializer_controller"]
        or selected["parameters_sha256"] != protocol["initializer_parameters_sha256"]
    ):
        raise ValueError("Initializer is not the complete validation-selected policy")
    initial = (
        public_parameters(parent)
        if selected["controller"] is None
        else archive_info(
            selected["controller"], parent, direct=selected["kind"] == "candidate"
        )["params"]
    )
    if identity(initial) != protocol["initializer_parameters_sha256"]:
        raise ValueError("Projected parent raw controller changed")
    return initial
