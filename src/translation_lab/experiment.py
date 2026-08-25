import asyncio
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from translation_lab.interfaces import ModelSpec, native_request
from translation_lab.metrics import audit_request, audit_translation
from translation_lab.models import Completion
from translation_lab.plans import (
    TranslationPlan,
    p0_plan,
    processed_output,
    raw_plan,
    reconstruct,
)
from translation_lab.protocols import (
    JSON_ARMS,
    PROMPT_JSON_ARMS,
    Arm,
    Message,
    RequestSpec,
    common_windows,
    json_requests,
    p0_request,
    parse_json_response,
)


class Generator(Protocol):
    async def generate(self, request: RequestSpec) -> Completion: ...


@dataclass(slots=True)
class RequestEvidence:
    source_id: str
    model: str
    method: Arm
    plan_revision: str
    target_language: str
    chunk_label: int
    generation_index: int
    request_index: int
    input_unit_indices: list[int]
    requested_unit_indices: list[int]
    messages: list[Message] | None
    prompt: str | None
    output_format: str
    max_output_tokens: int
    raw_output: str
    translations: list[str]
    parse_error: str | None
    finish_reason: str
    prompt_tokens: int | None
    completion_tokens: int | None
    translation_metrics: list[dict[str, Any]]


@dataclass(slots=True)
class DocumentEvidence:
    source_id: str
    model: str
    method: Arm
    plan_revision: str
    target_language: str
    chunk_label: int
    generation_index: int
    assembled_translation: str
    original_run_complete: bool
    reconstructable_from_unit_evidence: bool
    failure_reasons: list[str]
    prompt_tokens: int | None
    completion_tokens: int | None
    metrics: dict[str, Any]


@dataclass(slots=True)
class ExperimentResult:
    plan: TranslationPlan
    requests: list[RequestEvidence]
    document: DocumentEvidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.document.source_id,
            "plan": self.plan,
            "requests": [asdict(request) for request in self.requests],
            "document": asdict(self.document),
        }


async def _execute(
    source_id: str,
    model: str,
    method: Arm,
    language_code: str,
    plan: TranslationPlan,
    specification: RequestSpec,
    generator: Generator,
    generation_index: int,
) -> tuple[RequestEvidence, dict[int, str]]:
    completion = await generator.generate(specification)
    requested = specification["requested_unit_indices"]
    if method in JSON_ARMS:
        values, parse_error = parse_json_response(
            completion.text,
            len(requested),
            allow_markdown_wrapper=method in PROMPT_JSON_ARMS,
        )
    else:
        unit_by_index = {unit["unit_index"]: unit for unit in plan["translation_units"]}
        unit = unit_by_index[requested[0]]
        values = [processed_output(plan["method"], unit["kind"], completion.text)]
        parse_error = None
    translations = dict(zip(requested, values, strict=True)) if parse_error is None else {}
    unit_by_index = {unit["unit_index"]: unit for unit in plan["translation_units"]}
    translation_metrics = [
        audit_request(
            unit_by_index[index]["source_text"], value, language_code, completion.finish_reason
        )
        for index, value in translations.items()
    ]
    evidence = RequestEvidence(
        source_id=source_id,
        model=model,
        method=method,
        plan_revision=plan["plan_revision"],
        target_language=language_code,
        chunk_label=plan["chunk_label"],
        generation_index=generation_index,
        request_index=specification["request_index"],
        input_unit_indices=specification["input_unit_indices"],
        requested_unit_indices=requested,
        messages=specification["messages"],
        prompt=specification["prompt"],
        output_format=specification["output_format"],
        max_output_tokens=specification["max_output_tokens"],
        raw_output=completion.text,
        translations=values if parse_error is None else [],
        parse_error=parse_error,
        finish_reason=completion.finish_reason,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        translation_metrics=translation_metrics,
    )
    return evidence, translations


def _sum_optional(values: Sequence[int | None]) -> int | None:
    return (
        sum(value for value in values if value is not None)
        if all(value is not None for value in values)
        else None
    )


async def run_document(
    source_id: str,
    source_text: str,
    model: str,
    method: Arm,
    language_code: str,
    generator: Generator,
    *,
    token_count: Callable[[list[Message]], int] | None = None,
    chunk_label: int = 512,
    chunk_target_chars: int = 2048,
    generation_index: int = 0,
    model_spec: ModelSpec | None = None,
    renderer: Any | None = None,
    max_output_tokens: int | None = None,
) -> ExperimentResult:
    plan = (
        raw_plan(
            source_id,
            source_text,
            chunk_label=chunk_label,
            chunk_target_chars=chunk_target_chars,
        )
        if method == "RAW"
        else p0_plan(
            source_id,
            source_text,
            chunk_label=chunk_label,
            chunk_target_chars=chunk_target_chars,
        )
    )
    evidence: list[RequestEvidence] = []
    translations: dict[int, str] = {}
    units = plan["translation_units"]

    if method in JSON_ARMS:
        if token_count is None:
            raise ValueError("JSON window methods require the model tokenizer")
        windows = common_windows(units, token_count)
        specifications = json_requests(
            method,
            language_code,
            units,
            windows,
            max_output_tokens=max_output_tokens or 16384,
        )
        results = await asyncio.gather(
            *(
                _execute(
                    source_id,
                    model,
                    method,
                    language_code,
                    plan,
                    specification,
                    generator,
                    generation_index,
                )
                for specification in specifications
            )
        )
        for request, returned in results:
            evidence.append(request)
            translations.update(returned)
    elif method == "Pc":
        history: list[tuple[int, str, str]] = []
        for unit in units:
            max_tokens = 512 if unit["kind"] == "table_cell" else max_output_tokens or 8192
            specification = p0_request(
                unit["unit_index"],
                language_code,
                unit,
                history=history if unit["kind"] == "prose" else (),
                max_output_tokens=max_tokens,
            )
            request, returned = await _execute(
                source_id,
                model,
                method,
                language_code,
                plan,
                specification,
                generator,
                generation_index,
            )
            evidence.append(request)
            translations.update(returned)
            if unit["kind"] == "prose":
                history.append(
                    (
                        unit["unit_index"],
                        unit["source_text"],
                        returned[unit["unit_index"]],
                    )
                )
    else:
        specifications = [
            (
                native_request(
                    model_spec,
                    language_code,
                    unit,
                    request_index=unit["unit_index"],
                    renderer=renderer,
                    max_output_tokens=max_output_tokens,
                )
                if model_spec is not None
                else p0_request(
                    unit["unit_index"],
                    language_code,
                    unit,
                    max_output_tokens=(
                        512 if unit["kind"] == "table_cell" else max_output_tokens or 8192
                    ),
                )
            )
            for unit in units
        ]
        results = await asyncio.gather(
            *(
                _execute(
                    source_id,
                    model,
                    method,
                    language_code,
                    plan,
                    specification,
                    generator,
                    generation_index,
                )
                for specification in specifications
            )
        )
        for request, returned in results:
            evidence.append(request)
            translations.update(returned)

    expected = {unit["unit_index"] for unit in units}
    missing = sorted(expected - translations.keys())
    assembled = reconstruct(plan, {index: translations.get(index, "") for index in expected})
    failures = [f"missing_translation_unit:{index}" for index in missing]
    failures.extend(
        f"request_{request.request_index}:{request.parse_error}"
        for request in evidence
        if request.parse_error is not None
    )
    failures.extend(
        f"request_{request.request_index}:empty_output"
        for request in evidence
        if any(not translation.strip() for translation in request.translations)
    )
    if any(request.finish_reason == "length" for request in evidence):
        failures.append("finish_reason_length")
    metrics = audit_translation(source_text, assembled, language_code)
    metrics["severe_composite"] = bool(failures) or any(
        bool(item.get("severe_composite"))
        for request in evidence
        for item in request.translation_metrics
    )
    complete = not failures
    document = DocumentEvidence(
        source_id=source_id,
        model=model,
        method=method,
        plan_revision=plan["plan_revision"],
        target_language=language_code,
        chunk_label=chunk_label,
        generation_index=generation_index,
        assembled_translation=assembled,
        original_run_complete=complete,
        reconstructable_from_unit_evidence=not missing,
        failure_reasons=failures,
        prompt_tokens=_sum_optional([request.prompt_tokens for request in evidence]),
        completion_tokens=_sum_optional([request.completion_tokens for request in evidence]),
        metrics=metrics,
    )
    return ExperimentResult(plan, evidence, document)
