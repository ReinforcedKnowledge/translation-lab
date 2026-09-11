import asyncio
import json

import pytest

from translation_lab.evidence import validate_result
from translation_lab.experiment import run_document
from translation_lab.models import Completion
from translation_lab.protocols import RequestSpec


class QueueGenerator:
    def __init__(self, values: list[str]) -> None:
        self.values = iter(values)

    async def generate(self, request: RequestSpec) -> Completion:
        return Completion(next(self.values), "stop", 10, 3)


def test_pc_uses_cleaned_outputs_as_history_and_replays() -> None:
    source = "First paragraph.\n\nSecond paragraph."
    result = asyncio.run(
        run_document(
            "s1",
            source,
            "model",
            "Pc",
            "fr",
            QueueGenerator(["```fr\nPremier paragraphe.\n```", "Deuxième paragraphe."]),
            chunk_target_chars=10,
        )
    )

    assert result.document.assembled_translation == "Premier paragraphe.\n\nDeuxième paragraphe."
    assert result.requests[1].input_unit_indices == [0, 1]
    assert result.requests[1].messages is not None
    assert result.requests[1].messages[1]["content"] == "Premier paragraphe."
    assert validate_result(result)["reconstructable"] is True


def test_json_parse_failure_is_retained_as_an_incomplete_condition() -> None:
    result = asyncio.run(
        run_document(
            "s1",
            "A single unit.",
            "model",
            "A1",
            "fr",
            QueueGenerator(["not json"]),
            token_count=lambda messages: 20,
        )
    )

    assert result.document.original_run_complete is False
    assert result.document.reconstructable_from_unit_evidence is False
    assert result.requests[0].translations == []
    assert result.requests[0].parse_error is not None
    assert validate_result(result)["missing_unit_indices"] == [0]


def test_schema_packed_response_maps_outputs_to_requested_units() -> None:
    result = asyncio.run(
        run_document(
            "s1",
            "One.\n\nTwo.",
            "model",
            "B4",
            "fr",
            QueueGenerator([json.dumps({"translations": ["Un.", "Deux."]})]),
            token_count=lambda messages: 20,
            chunk_target_chars=4,
        )
    )

    assert result.document.assembled_translation == "Un.\n\nDeux."
    assert result.requests[0].requested_unit_indices == [0, 1]


def test_evidence_validator_rejects_a_changed_document() -> None:
    result = asyncio.run(run_document("s1", "One.", "model", "P0", "fr", QueueGenerator(["Un."])))
    result.document.assembled_translation = "Changed"

    with pytest.raises(ValueError, match="does not reproduce"):
        validate_result(result)


def test_empty_processed_output_is_retained_but_not_counted_as_complete() -> None:
    result = asyncio.run(run_document("s1", "One.", "model", "P0", "fr", QueueGenerator([""])))

    assert result.requests[0].translations == [""]
    assert result.document.reconstructable_from_unit_evidence is True
    assert result.document.original_run_complete is False
    assert result.document.failure_reasons == ["request_0:empty_output"]


def test_raw_whitespace_output_is_retained_but_not_counted_as_complete() -> None:
    result = asyncio.run(run_document("s1", "One.", "model", "RAW", "fr", QueueGenerator(["  "])))

    assert result.requests[0].translations == ["  "]
    assert result.document.reconstructable_from_unit_evidence is True
    assert result.document.original_run_complete is False


def test_system_boundary_uses_raw_units_and_records_sentinel_leaks() -> None:
    result = asyncio.run(
        run_document(
            "s1",
            "Instruction: write a proof.\n\nSecond paragraph.",
            "model",
            "SB",
            "fr",
            QueueGenerator(["Instruction : rédigez une preuve.", "Second paragraphe."]),
            chunk_target_chars=32,
        )
    )

    assert result.plan["method"] == "SB"
    assert all(unit["kind"] == "raw" for unit in result.plan["translation_units"])
    assert result.requests[0].messages is not None
    assert [message["role"] for message in result.requests[0].messages or []] == [
        "system",
        "user",
    ]
    assert result.document.original_run_complete is True
    assert validate_result(result)["reconstructable"] is True

    leaked = asyncio.run(
        run_document(
            "s2",
            "One.",
            "model",
            "SB",
            "fr",
            QueueGenerator(["Un. <END_TRANSLATION_PAYLOAD>"]),
        )
    )
    assert leaked.document.failure_reasons == ["sentinel_leak"]
    assert leaked.requests[0].translation_metrics[0]["sentinel_leak"] is True
