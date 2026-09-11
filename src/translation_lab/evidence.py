from collections import Counter
from typing import Any

from translation_lab.experiment import ExperimentResult
from translation_lab.plans import reconstruct, validate_plan


def validate_result(result: ExperimentResult) -> dict[str, Any]:
    plan = result.plan
    document = result.document
    validate_plan(plan)
    expected = {unit["unit_index"] for unit in plan["translation_units"]}
    request_indices = [request.request_index for request in result.requests]
    if len(request_indices) != len(set(request_indices)):
        raise ValueError("request indices are not unique")

    returned: dict[int, str] = {}
    conflicts: set[int] = set()
    requested_counts: Counter[int] = Counter()
    for request in result.requests:
        if (request.messages is None) == (request.prompt is None):
            raise ValueError(f"request {request.request_index} must contain messages or prompt")
        input_indices = set(request.input_unit_indices)
        requested_indices = set(request.requested_unit_indices)
        if not input_indices <= expected or not requested_indices <= input_indices:
            raise ValueError(f"request {request.request_index} references an unknown unit")
        requested_counts.update(request.requested_unit_indices)
        if request.parse_error is None and len(request.translations) != len(
            request.requested_unit_indices
        ):
            raise ValueError(f"request {request.request_index} has the wrong output count")
        if request.parse_error is not None and request.translations:
            raise ValueError(f"request {request.request_index} retained a failed parse")
        if request.parse_error is None:
            for index, translation in zip(
                request.requested_unit_indices, request.translations, strict=True
            ):
                if index in returned and returned[index] != translation:
                    conflicts.add(index)
                returned[index] = translation

    missing = sorted(expected - returned.keys())
    duplicate_targets = sorted(index for index, count in requested_counts.items() if count > 1)
    if conflicts:
        raise ValueError(f"conflicting translations for units {sorted(conflicts)}")
    if document.reconstructable_from_unit_evidence != (not missing):
        raise ValueError("reconstructability flag disagrees with retained unit outputs")
    if not missing:
        replayed = reconstruct(plan, returned)
        if replayed != document.assembled_translation:
            raise ValueError("published unit evidence does not reproduce the document")
    if (
        document.source_id != plan["source_id"]
        or document.plan_revision != plan["plan_revision"]
        or document.chunk_label != plan["chunk_label"]
        or plan["method"] != (document.method if document.method in {"RAW", "SB"} else "P0")
    ):
        raise ValueError("document identity disagrees with its plan")
    return {
        "valid": True,
        "translation_units": len(expected),
        "requests": len(result.requests),
        "missing_unit_indices": missing,
        "duplicate_target_indices": duplicate_targets,
        "reconstructable": not missing,
        "original_run_complete": document.original_run_complete,
    }
