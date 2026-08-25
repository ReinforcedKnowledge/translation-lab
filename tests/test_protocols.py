import json

from translation_lab.plans import TranslationUnit
from translation_lab.protocols import (
    common_windows,
    json_requests,
    p0_request,
    parse_json_response,
)


def units() -> list[TranslationUnit]:
    return [
        {"unit_index": 0, "kind": "prose", "source_text": "One."},
        {"unit_index": 1, "kind": "prose", "source_text": "Two."},
        {"unit_index": 2, "kind": "prose", "source_text": "Three."},
    ]


def test_pc_request_records_all_visible_history_units() -> None:
    request = p0_request(
        2,
        "fr",
        units()[2],
        history=[(0, "One.", "Un."), (1, "Two.", "Deux.")],
    )

    assert request["input_unit_indices"] == [0, 1, 2]
    assert request["requested_unit_indices"] == [2]
    assert [message["role"] for message in request["messages"] or []] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]


def test_json_arms_separate_visible_and_requested_units() -> None:
    typed_units = units()
    windows = common_windows(
        typed_units, lambda messages: len(messages[0]["content"]), prompt_token_limit=10_000
    )

    single = json_requests("B1", "fr", typed_units, windows)
    packed = json_requests("B4", "fr", typed_units, windows)

    assert [request["requested_unit_indices"] for request in single] == [[0], [1], [2]]
    assert all(request["input_unit_indices"] == [0, 1, 2] for request in single)
    assert packed[0]["requested_unit_indices"] == [0, 1, 2]
    schema = packed[0]["response_format"]
    assert schema is not None
    assert schema["json_schema"]["schema"]["properties"]["translations"]["minItems"] == 3


def test_prompt_json_allows_only_a_whole_response_fence() -> None:
    raw = json.dumps({"translations": ["Un."]})

    assert parse_json_response(f"```json\n{raw}\n```", 1, allow_markdown_wrapper=True) == (
        ["Un."],
        None,
    )
    assert parse_json_response(f"prefix {raw}", 1, allow_markdown_wrapper=True)[1]
    assert parse_json_response(f"```json\n{raw}\n```", 1, allow_markdown_wrapper=False)[1] == (
        "markdown_wrapper_not_allowed"
    )
