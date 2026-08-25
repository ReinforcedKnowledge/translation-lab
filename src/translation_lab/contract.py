from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RunValidation:
    summary: dict[str, object]
    issues: list[str]

    @property
    def valid(self) -> bool:
        return not self.issues


def validate_run(
    source_rows: Iterable[dict[str, Any]],
    result_rows: Iterable[dict[str, Any]],
    id_field: str = "id",
    elapsed_seconds: float | None = None,
) -> RunValidation:
    source_ids = [row.get(id_field) for row in source_rows]
    results = list(result_rows)
    result_ids = [row.get(id_field) for row in results]
    issues: list[str] = []

    if missing_source_ids := [index for index, value in enumerate(source_ids) if value is None]:
        issues.append(f"source rows missing {id_field}: {missing_source_ids}")
    if missing_result_ids := [index for index, value in enumerate(result_ids) if value is None]:
        issues.append(f"result rows missing {id_field}: {missing_result_ids}")

    source_counts = Counter(source_ids)
    result_counts = Counter(result_ids)
    duplicate_sources = sorted(str(value) for value, count in source_counts.items() if count > 1)
    duplicate_results = sorted(str(value) for value, count in result_counts.items() if count > 1)
    if duplicate_sources:
        issues.append(f"duplicate source identifiers: {duplicate_sources}")
    if duplicate_results:
        issues.append(f"duplicate result identifiers: {duplicate_results}")

    missing = sorted(str(value) for value in set(source_ids) - set(result_ids))
    unexpected = sorted(str(value) for value in set(result_ids) - set(source_ids))
    if missing:
        issues.append(f"missing result identifiers: {missing}")
    if unexpected:
        issues.append(f"unexpected result identifiers: {unexpected}")

    accepted = 0
    generated_tokens = 0
    accepted_tokens = 0
    failure_counts: Counter[str] = Counter()
    for row in results:
        row_id = row.get(id_field)
        row_failures: list[str] = []
        if row.get("error"):
            row_failures.append("request_error")
        if not isinstance(row.get("output"), str):
            row_failures.append("missing_output")
        finish_reasons = row.get("finish_reasons")
        if not isinstance(finish_reasons, list):
            row_failures.append("missing_finish_reasons")
        elif "length" in finish_reasons:
            row_failures.append("length_stop")
        completion_tokens = row.get("completion_tokens")
        if not isinstance(completion_tokens, int):
            row_failures.append("missing_completion_tokens")
            completion_token_count = 0
        else:
            generated_tokens += completion_tokens
            completion_token_count = completion_tokens
        if not isinstance(row.get("prompt_tokens"), int):
            row_failures.append("missing_prompt_tokens")
        requests = row.get("requests")
        if not isinstance(requests, list):
            row_failures.append("missing_request_records")
        else:
            if isinstance(finish_reasons, list) and len(finish_reasons) != len(requests):
                row_failures.append("request_finish_reason_count_mismatch")
            request_finish_reasons: list[str] = []
            for request in requests:
                if not isinstance(request, dict):
                    row_failures.append("invalid_request_record")
                    continue
                request_finish_reason = request.get("finish_reason")
                if not isinstance(request_finish_reason, str):
                    row_failures.append("missing_request_finish_reason")
                else:
                    request_finish_reasons.append(request_finish_reason)
                if not isinstance(request.get("prompt_tokens"), int):
                    row_failures.append("missing_request_prompt_tokens")
                if not isinstance(request.get("completion_tokens"), int):
                    row_failures.append("missing_request_completion_tokens")
            if isinstance(finish_reasons, list) and request_finish_reasons != finish_reasons:
                row_failures.append("request_finish_reasons_mismatch")
        for field in ("structure_preserved", "verbatim_blocks_preserved", "looks_translated"):
            if row.get(field) is not True:
                row_failures.append(field)
        if row.get("truncated_chunks") != 0:
            row_failures.append("truncated_chunks")
        if row.get("empty_chunks") != 0:
            row_failures.append("empty_chunks")

        if row_failures:
            failure_counts.update(row_failures)
            issues.append(f"row {row_id}: {', '.join(sorted(set(row_failures)))}")
        else:
            accepted += 1
            accepted_tokens += completion_token_count

    rates: dict[str, float] = {}
    if elapsed_seconds is not None and elapsed_seconds > 0:
        rates = {
            "generated_tokens_per_second": round(generated_tokens / elapsed_seconds, 3),
            "accepted_tokens_per_second": round(accepted_tokens / elapsed_seconds, 3),
            "accepted_rows_per_hour": round(accepted * 3600 / elapsed_seconds, 3),
        }
    summary: dict[str, object] = {
        "source_rows": len(source_ids),
        "result_rows": len(results),
        "accepted_rows": accepted,
        "rejected_rows": len(results) - accepted,
        "generated_tokens": generated_tokens,
        "accepted_tokens": accepted_tokens,
        "failure_counts": dict(failure_counts),
        **rates,
    }
    return RunValidation(summary, issues)
