import json
from pathlib import Path

from translation_lab.cli import main
from translation_lab.plans import p0_plan


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_dry_comet_command_writes_report(tmp_path: Path) -> None:
    sources = tmp_path / "sources.jsonl"
    results = tmp_path / "results.jsonl"
    report = tmp_path / "comet.json"
    write_jsonl(sources, [{"id": "a", "src_text": "one two three"}])
    write_jsonl(results, [{"id": "a", "pairs": [["one two three", "un deux trois"]]}])

    status = main(["comet", str(sources), str(results), str(report), "--dry-run"])
    content = json.loads(report.read_text())

    assert status == 0
    assert content["summary"]["documents"] == 1
    assert content["summary"]["alignment"] == {"whole": 1}


def test_comet_reports_and_excludes_incomplete_experiment_evidence(tmp_path: Path) -> None:
    sources = tmp_path / "sources.jsonl"
    results = tmp_path / "results.jsonl"
    report = tmp_path / "comet.json"
    plan = p0_plan("a", "one two three")
    write_jsonl(sources, [{"id": "a", "src_text": "one two three"}])
    write_jsonl(
        results,
        [
            {
                "source_id": "a",
                "plan": plan,
                "requests": [
                    {
                        "requested_unit_indices": [0],
                        "translations": [],
                        "parse_error": "json_decode",
                    }
                ],
                "document": {"assembled_translation": ""},
            }
        ],
    )

    status = main(["comet", str(sources), str(results), str(report), "--dry-run"])
    summary = json.loads(report.read_text())["summary"]

    assert status == 0
    assert summary["documents"] == 0
    assert summary["incomplete_documents_with_partial_or_no_units"] == 1


def test_validate_run_command_writes_valid_report(tmp_path: Path) -> None:
    sources = tmp_path / "sources.jsonl"
    results = tmp_path / "results.jsonl"
    report = tmp_path / "run.json"
    write_jsonl(sources, [{"id": "a", "src_text": "Hello"}])
    write_jsonl(
        results,
        [
            {
                "id": "a",
                "output": "Bonjour",
                "finish_reasons": ["stop"],
                "requests": [
                    {
                        "finish_reason": "stop",
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                    }
                ],
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "structure_preserved": True,
                "verbatim_blocks_preserved": True,
                "looks_translated": True,
                "truncated_chunks": 0,
                "empty_chunks": 0,
            }
        ],
    )

    status = main(["validate-run", str(sources), str(results), str(report)])

    assert status == 0
    assert json.loads(report.read_text())["valid"] is True


def test_review_file_workflow_prepares_and_summarizes(tmp_path: Path) -> None:
    sources = tmp_path / "sources.jsonl"
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    items = tmp_path / "items.jsonl"
    mapping = tmp_path / "mapping.jsonl"
    write_jsonl(sources, [{"id": "a", "src_text": "Hello"}])
    write_jsonl(first, [{"id": "a", "output": "Bonjour"}])
    write_jsonl(second, [{"id": "a", "output": "Salut"}])

    status = main(
        [
            "prepare-review",
            str(sources),
            str(items),
            str(mapping),
            "--candidate",
            f"first={first}",
            "--candidate",
            f"second={second}",
            "--language",
            "French",
            "--task-type",
            "model_comparison",
        ]
    )

    assert status == 0
    assert "first" not in items.read_text()
    assert "first" in mapping.read_text()

    item = json.loads(items.read_text())
    private = json.loads(mapping.read_text())
    labels = [candidate["label"] for candidate in item["candidates"]]
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
    results = tmp_path / "judge-results.jsonl"
    report = tmp_path / "review-report.json"
    native = tmp_path / "native-review.jsonl"
    write_jsonl(
        results,
        [
            {
                "item_id": item["item_id"],
                "target_language": "French",
                "task_type": "model_comparison",
                "candidate_labels": labels,
                "candidate_scores": {label: score for label in labels},
                "ranking": labels,
                "winner": labels[0],
                "difference_material": True,
                "decision_basis": "The first candidate is more complete.",
                "evidence": [],
                "needs_human_review": False,
                "human_review_reason": "",
                "confidence": "high",
            }
        ],
    )

    status = main(
        [
            "summarize-review",
            str(items),
            str(mapping),
            str(results),
            str(report),
            "--native-review",
            str(native),
            "--force-language",
            "French",
        ]
    )
    summary = json.loads(report.read_text())

    assert status == 0
    assert summary["items"][0]["winner"] == private["conditions"][labels[0]]
    assert summary["summary"]["overall"]["routed_to_native_review"] == 1
    assert "winner" not in json.loads(native.read_text())
