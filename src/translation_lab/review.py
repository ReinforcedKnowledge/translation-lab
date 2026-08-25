import hashlib
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from importlib.resources import files
from typing import Any, Protocol

SCORE_FIELDS = {
    "adequacy_completeness",
    "no_task_execution",
    "fluency",
    "terminology_consistency",
    "coherence_chunk_seams",
    "format_structure",
    "code_math_table",
    "wrong_language_drift",
    "overall",
}
HOLISTIC_LABELS = {"faithful", "minor", "major", "broken"}
CONFIDENCE_LEVELS = {"low", "medium", "high"}
FAILURE_TAGS = {
    "omission",
    "mistranslation",
    "addition",
    "condensation",
    "untranslated_source",
    "wrong_language",
    "english_reversion",
    "third_language_drift",
    "task_execution",
    "new_answer",
    "new_code",
    "code_corruption",
    "math_corruption",
    "table_corruption",
    "format_corruption",
    "placeholder_or_tag_artifact",
    "chunk_seam",
    "repetition",
    "terminology_drift",
    "coherence_break",
    "overliteral",
    "fluency_issue",
    "cannot_judge",
}
EVIDENCE_DIMENSIONS = SCORE_FIELDS - {"overall"}
EVIDENCE_SEVERITIES = {"minor", "major", "critical"}
REQUIRED_FIELDS = {
    "item_id",
    "target_language",
    "task_type",
    "candidate_labels",
    "candidate_scores",
    "ranking",
    "winner",
    "difference_material",
    "decision_basis",
    "evidence",
    "needs_human_review",
    "human_review_reason",
    "confidence",
}


class Reviewer(Protocol):
    async def review(self, rubric: str, prompt: str) -> dict[str, Any]: ...


def rubric_text() -> str:
    return files("translation_lab").joinpath("rubric.md").read_text(encoding="utf-8")


def rubric_sha256() -> str:
    return hashlib.sha256(rubric_text().encode()).hexdigest()


def review_prompt(item: dict[str, Any]) -> str:
    item_id = item.get("item_id")
    target_language = item.get("target_language")
    task_type = item.get("task_type")
    source = item.get("source_text")
    candidates = item.get("candidates")
    if not isinstance(item_id, str) or not item_id:
        raise ValueError("item_id must be a non-empty string")
    if not isinstance(target_language, str) or not target_language:
        raise ValueError(f"{item_id}: target_language must be a non-empty string")
    if not isinstance(task_type, str) or not task_type:
        raise ValueError(f"{item_id}: task_type must be a non-empty string")
    if not isinstance(source, str):
        raise ValueError(f"{item_id}: source_text must be a string")
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= 3:
        raise ValueError(f"{item_id}: candidates must contain two or three items")

    labels: list[str] = []
    sections: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError(f"{item_id}: each candidate must be an object")
        label = candidate.get("label")
        text = candidate.get("text")
        if label not in {"A", "B", "C"} or not isinstance(text, str):
            raise ValueError(f"{item_id}: each candidate needs an A/B/C label and text")
        labels.append(label)
        sections.append(f"CANDIDATE {label}:\n{text}")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{item_id}: candidate labels must be unique")

    score: dict[str, object] = {field: 3 for field in sorted(SCORE_FIELDS)}
    score.update({"holistic_label": "minor", "failure_tags": []})
    template = {
        "item_id": item_id,
        "target_language": target_language,
        "task_type": task_type,
        "candidate_labels": labels,
        "candidate_scores": {label: score for label in labels},
        "ranking": labels,
        "winner": "tie",
        "difference_material": False,
        "decision_basis": "",
        "evidence": [],
        "needs_human_review": False,
        "human_review_reason": "",
        "confidence": "medium",
    }
    return (
        "Apply the supplied translation rubric to this blinded comparison.\n"
        "Score every candidate independently before ranking them. Return one JSON object only.\n"
        "Replace every score and judgment in the template with your evaluation.\n\n"
        f"ITEM ID: {item_id}\n"
        f"TARGET LANGUAGE: {target_language}\n"
        f"TASK TYPE: {task_type}\n\n"
        f"SOURCE:\n{source}\n\n"
        f"{'\n\n'.join(sections)}\n\n"
        f"OUTPUT TEMPLATE:\n{json.dumps(template, ensure_ascii=False, indent=2)}"
    )


def blinded_order(labels: Sequence[str], key: str) -> list[str]:
    return sorted(
        labels,
        key=lambda label: hashlib.sha256(f"{key}|{label}".encode()).hexdigest(),
    )


def blind_candidates(
    candidates: dict[str, str], key: str
) -> tuple[list[dict[str, str]], dict[str, str]]:
    conditions = blinded_order(list(candidates), key)
    labels = [chr(ord("A") + index) for index in range(len(conditions))]
    mapping = dict(zip(labels, conditions, strict=True))
    return (
        [{"label": label, "text": candidates[condition]} for label, condition in mapping.items()],
        mapping,
    )


def _rows_by_id(
    rows: Iterable[dict[str, Any]], id_field: str, source: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if id_field not in row:
            raise ValueError(f"{source}: row is missing {id_field!r}")
        row_id = str(row[id_field])
        if row_id in indexed:
            raise ValueError(f"{source}: duplicate {id_field} {row_id!r}")
        indexed[row_id] = row
    return indexed


def prepare_review_items(
    sources: Iterable[dict[str, Any]],
    candidates: dict[str, Iterable[dict[str, Any]]],
    target_language: str,
    task_type: str,
    id_field: str = "id",
    source_field: str = "src_text",
    output_field: str = "output",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 2 <= len(candidates) <= 3:
        raise ValueError("exactly two or three candidate files are required")
    if not target_language:
        raise ValueError("target_language must not be empty")
    if not task_type:
        raise ValueError("task_type must not be empty")

    source_rows = list(sources)
    indexed_sources = _rows_by_id(source_rows, id_field, "sources")
    indexed_candidates = {
        condition: _rows_by_id(rows, id_field, condition) for condition, rows in candidates.items()
    }
    signature = json.dumps(
        [target_language, task_type, sorted(indexed_candidates)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    items: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    for source_id, source_row in indexed_sources.items():
        source_text = source_row.get(source_field)
        if not isinstance(source_text, str):
            raise ValueError(f"sources: {source_id!r} has no string {source_field!r}")
        texts: dict[str, str] = {}
        for condition, rows in indexed_candidates.items():
            candidate_row = rows.get(source_id)
            if candidate_row is None:
                raise ValueError(f"{condition}: missing result for {source_id!r}")
            text = candidate_row.get(output_field)
            if not isinstance(text, str):
                raise ValueError(f"{condition}: {source_id!r} has no string {output_field!r}")
            texts[condition] = text
        digest = hashlib.sha256(f"{signature}\n{source_id}".encode()).hexdigest()[:16]
        item_id = f"review-{digest}"
        blinded, label_map = blind_candidates(texts, item_id)
        items.append(
            {
                "item_id": item_id,
                "source_id": source_row[id_field],
                "target_language": target_language,
                "task_type": task_type,
                "source_text": source_text,
                "candidates": blinded,
            }
        )
        mappings.append(
            {
                "item_id": item_id,
                "source_id": source_row[id_field],
                "target_language": target_language,
                "task_type": task_type,
                "conditions": label_map,
            }
        )
    return items, mappings


def _validate_score(
    item_id: str,
    label: str,
    score: Any,
    evidence_candidates: set[str],
) -> list[str]:
    errors: list[str] = []
    path = f"{item_id}.candidate_scores.{label}"
    if not isinstance(score, dict):
        return [f"{path}: expected object"]
    required = SCORE_FIELDS | {"holistic_label", "failure_tags"}
    if missing := sorted(required - set(score)):
        errors.append(f"{path}: missing {missing}")
    if extra := sorted(set(score) - required):
        errors.append(f"{path}: unexpected fields {extra}")
    for field in SCORE_FIELDS:
        value = score.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5:
            errors.append(f"{path}.{field}: expected integer 1..5")
    if score.get("holistic_label") not in HOLISTIC_LABELS:
        errors.append(f"{path}.holistic_label: invalid value")
    tags = score.get("failure_tags")
    if not isinstance(tags, list):
        errors.append(f"{path}.failure_tags: expected array")
        tags = []
    elif not all(isinstance(tag, str) for tag in tags):
        errors.append(f"{path}.failure_tags: expected strings")
    else:
        if len(tags) != len(set(tags)):
            errors.append(f"{path}.failure_tags: duplicate values")
        if invalid := sorted(tag for tag in tags if tag not in FAILURE_TAGS):
            errors.append(f"{path}.failure_tags: invalid values {invalid}")
    problematic = (
        any(isinstance(score.get(field), int) and score[field] <= 3 for field in SCORE_FIELDS)
        or score.get("holistic_label") in {"major", "broken"}
        or bool(tags)
    )
    if problematic and label not in evidence_candidates:
        errors.append(f"{item_id}: candidate {label} is problematic but has no evidence")
    return errors


def validate_judge_result(result: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    item_id = str(result.get("item_id", "<missing>"))
    errors: list[str] = []
    if missing := sorted(REQUIRED_FIELDS - set(result)):
        errors.append(f"{item_id}: missing fields {missing}")
    if extra := sorted(set(result) - REQUIRED_FIELDS):
        errors.append(f"{item_id}: unexpected fields {extra}")
    for field in ("item_id", "target_language", "task_type"):
        if result.get(field) != expected.get(field):
            errors.append(f"{item_id}.{field}: does not match input")

    expected_labels = sorted(str(candidate["label"]) for candidate in expected["candidates"])
    labels = result.get("candidate_labels")
    if (
        not isinstance(labels, list)
        or not all(isinstance(label, str) for label in labels)
        or sorted(labels) != expected_labels
    ):
        errors.append(f"{item_id}.candidate_labels: expected {expected_labels}")
        labels = expected_labels

    scores = result.get("candidate_scores")
    if not isinstance(scores, dict):
        errors.append(f"{item_id}.candidate_scores: expected object")
        scores = {}
    if sorted(scores) != sorted(labels):
        errors.append(f"{item_id}.candidate_scores: keys do not match candidate labels")

    ranking = result.get("ranking")
    if (
        not isinstance(ranking, list)
        or not all(isinstance(label, str) for label in ranking)
        or sorted(ranking) != sorted(labels)
    ):
        errors.append(f"{item_id}.ranking: expected a permutation of candidate labels")
    winner = result.get("winner")
    if winner != "tie" and winner not in labels:
        errors.append(f"{item_id}.winner: expected a candidate label or tie")
    if not isinstance(result.get("difference_material"), bool):
        errors.append(f"{item_id}.difference_material: expected boolean")
    if not isinstance(result.get("needs_human_review"), bool):
        errors.append(f"{item_id}.needs_human_review: expected boolean")
    if result.get("confidence") not in CONFIDENCE_LEVELS:
        errors.append(f"{item_id}.confidence: invalid value")
    for field in ("decision_basis", "human_review_reason"):
        if not isinstance(result.get(field), str):
            errors.append(f"{item_id}.{field}: expected string")

    evidence = result.get("evidence")
    evidence_candidates: set[str] = set()
    if not isinstance(evidence, list):
        errors.append(f"{item_id}.evidence: expected array")
    else:
        for index, item in enumerate(evidence):
            candidate = item.get("candidate") if isinstance(item, dict) else None
            if candidate not in labels:
                errors.append(f"{item_id}.evidence[{index}]: invalid candidate")
            else:
                evidence_candidates.add(str(candidate))
            if not isinstance(item, dict):
                continue
            required_evidence = {
                "candidate",
                "severity",
                "dimension",
                "source_span",
                "candidate_span",
                "explanation",
            }
            if missing := sorted(required_evidence - set(item)):
                errors.append(f"{item_id}.evidence[{index}]: missing fields {missing}")
            if extra := sorted(set(item) - required_evidence):
                errors.append(f"{item_id}.evidence[{index}]: unexpected fields {extra}")
            if item.get("severity") not in EVIDENCE_SEVERITIES:
                errors.append(f"{item_id}.evidence[{index}].severity: invalid value")
            if item.get("dimension") not in EVIDENCE_DIMENSIONS:
                errors.append(f"{item_id}.evidence[{index}].dimension: invalid value")
            for field in ("source_span", "candidate_span", "explanation"):
                if not isinstance(item.get(field), str):
                    errors.append(f"{item_id}.evidence[{index}].{field}: expected string")
    for label in labels:
        errors.extend(_validate_score(item_id, label, scores.get(label), evidence_candidates))
    cannot_judge = any(
        "cannot_judge" in score.get("failure_tags", [])
        for score in scores.values()
        if isinstance(score, dict)
    )
    if cannot_judge and result.get("confidence") != "low":
        errors.append(f"{item_id}: cannot_judge requires low confidence")
    if result.get("needs_human_review") and not result.get("human_review_reason"):
        errors.append(f"{item_id}: human_review_reason is required when review is requested")
    return errors


def validate_judge_batch(
    results: Iterable[dict[str, Any]], expected: Iterable[dict[str, Any]]
) -> list[str]:
    expected_by_id: dict[str, dict[str, Any]] = {}
    results_by_id: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for item in expected:
        item_id = str(item.get("item_id", "<missing>"))
        if item_id in expected_by_id:
            errors.append(f"duplicate expected item: {item_id}")
        expected_by_id[item_id] = item
    for result in results:
        item_id = str(result.get("item_id", "<missing>"))
        if item_id in results_by_id:
            errors.append(f"duplicate result: {item_id}")
        results_by_id[item_id] = result
    if missing := sorted(set(expected_by_id) - set(results_by_id)):
        errors.append(f"missing results: {missing}")
    if extra := sorted(set(results_by_id) - set(expected_by_id)):
        errors.append(f"unexpected results: {extra}")
    for item_id in sorted(set(expected_by_id) & set(results_by_id)):
        errors.extend(validate_judge_result(results_by_id[item_id], expected_by_id[item_id]))
    return errors


def _review_reasons(result: dict[str, Any], force_languages: set[str]) -> list[str]:
    reasons: list[str] = []
    if result["target_language"] in force_languages:
        reasons.append("forced_language")
    if result["needs_human_review"]:
        reasons.append("judge_requested")
    if result["confidence"] == "low":
        reasons.append("low_confidence")
    scores = result["candidate_scores"]
    if any("cannot_judge" in score["failure_tags"] for score in scores.values()):
        reasons.append("cannot_judge")
    bad = {
        label for label, score in scores.items() if score["holistic_label"] in {"major", "broken"}
    }
    if result["winner"] != "tie" and result["winner"] in bad:
        reasons.append("selected_winner_bad")
    if bad == set(scores):
        reasons.append("all_candidates_bad")
    return reasons


def _aggregate_reviews(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for condition, score in row["candidate_scores"].items():
            by_condition[condition].append(
                {
                    "score": score,
                    "winner": row["winner"],
                    "material": row["difference_material"],
                }
            )

    conditions: dict[str, object] = {}
    for condition, values in sorted(by_condition.items()):
        score_rows = [value["score"] for value in values]
        conditions[condition] = {
            "appearances": len(values),
            "wins": sum(value["winner"] == condition for value in values),
            "material_wins": sum(
                value["winner"] == condition and value["material"] for value in values
            ),
            "ties": sum(value["winner"] == "tie" for value in values),
            "mean_scores": {
                field: round(statistics.mean(score[field] for score in score_rows), 6)
                for field in sorted(SCORE_FIELDS)
            },
            "holistic_labels": dict(
                sorted(Counter(score["holistic_label"] for score in score_rows).items())
            ),
            "failure_tags": dict(
                sorted(
                    Counter(tag for score in score_rows for tag in score["failure_tags"]).items()
                )
            ),
        }

    return {
        "items": len(rows),
        "material_differences": sum(row["difference_material"] for row in rows),
        "judge_requested_human_review": sum(row["needs_human_review"] for row in rows),
        "routed_to_native_review": sum(row["routed_to_native_review"] for row in rows),
        "confidence": dict(sorted(Counter(row["confidence"] for row in rows).items())),
        "winners": dict(sorted(Counter(row["winner"] for row in rows).items())),
        "conditions": conditions,
    }


def build_review_report(
    expected: Iterable[dict[str, Any]],
    mappings: Iterable[dict[str, Any]],
    results: Iterable[dict[str, Any]],
    force_languages: Iterable[str] = (),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected_rows = list(expected)
    result_rows = list(results)
    if errors := validate_judge_batch(result_rows, expected_rows):
        raise ValueError("; ".join(errors))

    expected_by_id = {str(item["item_id"]): item for item in expected_rows}
    mapping_by_id: dict[str, dict[str, Any]] = {}
    for mapping in mappings:
        item_id = str(mapping.get("item_id", "<missing>"))
        if item_id in mapping_by_id:
            raise ValueError(f"duplicate mapping: {item_id}")
        mapping_by_id[item_id] = mapping
    if missing := sorted(set(expected_by_id) - set(mapping_by_id)):
        raise ValueError(f"missing mappings: {missing}")
    if extra := sorted(set(mapping_by_id) - set(expected_by_id)):
        raise ValueError(f"unexpected mappings: {extra}")

    forced = set(force_languages)
    unblinded: list[dict[str, Any]] = []
    native_items: list[dict[str, Any]] = []
    for result in result_rows:
        item_id = str(result["item_id"])
        expected_item = expected_by_id[item_id]
        mapping = mapping_by_id[item_id]
        label_map = mapping.get("conditions")
        expected_labels = {candidate["label"] for candidate in expected_item["candidates"]}
        if (
            not isinstance(label_map, dict)
            or not all(isinstance(label, str) for label in label_map)
            or not all(isinstance(condition, str) for condition in label_map.values())
            or set(label_map) != expected_labels
        ):
            raise ValueError(f"{item_id}: mapping labels do not match candidates")
        if len(set(label_map.values())) != len(label_map):
            raise ValueError(f"{item_id}: mapping conditions must be unique")

        candidate_scores = {
            str(label_map[label]): score for label, score in result["candidate_scores"].items()
        }
        winner = result["winner"]
        unblinded_winner = "tie" if winner == "tie" else str(label_map[winner])
        reasons = _review_reasons(result, forced)
        evidence = [
            {
                "condition": str(label_map[item["candidate"]]),
                **{key: value for key, value in item.items() if key != "candidate"},
            }
            for item in result["evidence"]
        ]
        row = {
            "item_id": item_id,
            "source_id": mapping.get("source_id"),
            "target_language": result["target_language"],
            "task_type": result["task_type"],
            "candidate_scores": candidate_scores,
            "ranking": [str(label_map[label]) for label in result["ranking"]],
            "winner": unblinded_winner,
            "difference_material": result["difference_material"],
            "decision_basis": result["decision_basis"],
            "evidence": evidence,
            "needs_human_review": result["needs_human_review"],
            "human_review_reason": result["human_review_reason"],
            "confidence": result["confidence"],
            "routed_to_native_review": bool(reasons),
            "routing_reasons": reasons,
        }
        unblinded.append(row)
        if reasons:
            native_items.append(expected_item)

    languages: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_types: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unblinded:
        languages[row["target_language"]].append(row)
        task_types[row["task_type"]].append(row)
    report = {
        "status": "model-assisted triage; final acceptance requires fluent target-language review",
        "rubric_sha256": rubric_sha256(),
        "routing_policy": {
            "automatic": [
                "judge_requested",
                "low_confidence",
                "cannot_judge",
                "selected_winner_bad",
                "all_candidates_bad",
            ],
            "forced_languages": sorted(forced),
        },
        "summary": {
            "overall": _aggregate_reviews(unblinded),
            "by_language": {
                key: _aggregate_reviews(value) for key, value in sorted(languages.items())
            },
            "by_task_type": {
                key: _aggregate_reviews(value) for key, value in sorted(task_types.items())
            },
        },
        "items": unblinded,
    }
    return report, native_items


async def review_item(item: dict[str, Any], reviewer: Reviewer) -> dict[str, Any]:
    result = await reviewer.review(rubric_text(), review_prompt(item))
    if errors := validate_judge_result(result, item):
        raise ValueError("; ".join(errors))
    return result
