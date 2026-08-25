from translation_lab.contract import validate_run


def accepted_result(row_id: str) -> dict[str, object]:
    return {
        "id": row_id,
        "output": "Bonjour le monde.",
        "finish_reasons": ["stop"],
        "requests": [
            {
                "finish_reason": "stop",
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "source_characters": 12,
                "max_tokens": 100,
            }
        ],
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "structure_preserved": True,
        "verbatim_blocks_preserved": True,
        "looks_translated": True,
        "truncated_chunks": 0,
        "empty_chunks": 0,
    }


def test_accepted_run_reports_end_to_end_rates() -> None:
    validation = validate_run(
        [{"id": "a"}, {"id": "b"}],
        [accepted_result("a"), accepted_result("b")],
        elapsed_seconds=2.0,
    )

    assert validation.valid
    assert validation.summary["accepted_rows"] == 2
    assert validation.summary["accepted_tokens_per_second"] == 10.0
    assert validation.summary["accepted_rows_per_hour"] == 3600.0


def test_contract_rejects_missing_rows_and_length_stops() -> None:
    result = accepted_result("a")
    result["finish_reasons"] = ["length"]
    result["truncated_chunks"] = 1

    validation = validate_run([{"id": "a"}, {"id": "b"}], [result])

    assert not validation.valid
    assert any("missing result identifiers" in issue for issue in validation.issues)
    assert any("length_stop" in issue for issue in validation.issues)
