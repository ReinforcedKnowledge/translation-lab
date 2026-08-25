import asyncio
from copy import deepcopy
from typing import Any

from translation_lab.review import (
    blind_candidates,
    build_review_report,
    prepare_review_items,
    review_item,
    review_prompt,
    rubric_text,
    validate_judge_batch,
)


def valid_input() -> dict[str, object]:
    return {
        "item_id": "item-1",
        "target_language": "French",
        "task_type": "model_comparison",
        "source_text": "One",
        "candidates": [{"label": "A", "text": "Un"}, {"label": "B", "text": "Deux"}],
    }


def valid_result() -> dict[str, Any]:
    score = {
        "adequacy_completeness": 5,
        "no_task_execution": 5,
        "fluency": 5,
        "terminology_consistency": 5,
        "coherence_chunk_seams": 5,
        "format_structure": 5,
        "code_math_table": 5,
        "wrong_language_drift": 5,
        "overall": 5,
        "holistic_label": "faithful",
        "failure_tags": [],
    }
    return {
        "item_id": "item-1",
        "target_language": "French",
        "task_type": "model_comparison",
        "candidate_labels": ["A", "B"],
        "candidate_scores": {"A": deepcopy(score), "B": deepcopy(score)},
        "ranking": ["A", "B"],
        "winner": "tie",
        "difference_material": False,
        "decision_basis": "The candidates are equivalent.",
        "needs_human_review": False,
        "human_review_reason": "",
        "confidence": "high",
        "evidence": [],
    }


def test_candidate_blinding_is_stable_and_reversible() -> None:
    blinded, mapping = blind_candidates({"model-a": "One", "model-b": "Two"}, "case-1")

    assert blinded == blind_candidates({"model-a": "One", "model-b": "Two"}, "case-1")[0]
    assert {item["label"] for item in blinded} == set(mapping)
    assert set(mapping.values()) == {"model-a", "model-b"}


def test_valid_judge_batch_passes() -> None:
    assert validate_judge_batch([valid_result()], [valid_input()]) == []


def test_problematic_score_requires_evidence() -> None:
    result = valid_result()
    result["candidate_scores"]["A"]["adequacy_completeness"] = 2

    errors = validate_judge_batch([result], [valid_input()])

    assert any("problematic but has no evidence" in error for error in errors)


class StaticReviewer:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.rubric = ""
        self.prompt = ""

    async def review(self, rubric: str, prompt: str) -> dict[str, Any]:
        self.rubric = rubric
        self.prompt = prompt
        return self.result


def test_review_item_loads_rubric_builds_prompt_and_validates_result() -> None:
    reviewer = StaticReviewer(valid_result())

    result = asyncio.run(review_item(valid_input(), reviewer))

    assert result["winner"] == "tie"
    assert "Adequacy and completeness" in reviewer.rubric
    assert "SOURCE:\nOne" in reviewer.prompt
    assert "CANDIDATE A:\nUn" in reviewer.prompt


def test_review_prompt_requires_source_text() -> None:
    item = valid_input()
    del item["source_text"]

    try:
        review_prompt(item)
    except ValueError as error:
        assert "source_text" in str(error)
    else:
        raise AssertionError("missing source_text was accepted")


def test_rubric_is_a_packaged_resource() -> None:
    assert rubric_text().startswith("# Translation comparison rubric")


def test_prepare_review_items_joins_and_blinds_conditions() -> None:
    items, mappings = prepare_review_items(
        [{"id": "row-1", "src_text": "One"}],
        {
            "gemma-3@512": [{"id": "row-1", "output": "Un"}],
            "gemma-4@512": [{"id": "row-1", "output": "Une"}],
        },
        "French",
        "model_comparison",
    )

    assert len(items) == len(mappings) == 1
    assert items[0]["source_id"] == "row-1"
    assert {candidate["label"] for candidate in items[0]["candidates"]} == {"A", "B"}
    assert "gemma-3" not in str(items[0])
    assert set(mappings[0]["conditions"].values()) == {"gemma-3@512", "gemma-4@512"}


def test_build_review_report_unblinds_and_routes_without_anchoring_packet() -> None:
    items, mappings = prepare_review_items(
        [{"id": "row-1", "src_text": "One"}],
        {
            "gemma-3@512": [{"id": "row-1", "output": "Un"}],
            "gemma-4@512": [{"id": "row-1", "output": "Une"}],
        },
        "French",
        "model_comparison",
    )
    result = valid_result()
    result["item_id"] = items[0]["item_id"]
    result["winner"] = "A"
    result["difference_material"] = True
    result["needs_human_review"] = True
    result["human_review_reason"] = "A fluent reader should confirm the difference."

    report, native_items = build_review_report(items, mappings, [result])

    condition = mappings[0]["conditions"]["A"]
    assert report["items"][0]["winner"] == condition
    assert report["summary"]["overall"]["conditions"][condition]["wins"] == 1
    assert report["summary"]["overall"]["routed_to_native_review"] == 1
    assert native_items == items
    assert "winner" not in native_items[0]
