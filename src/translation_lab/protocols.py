import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from translation_lab.plans import TranslationUnit

LANGUAGES = {
    "pl": "Polish",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "fi": "Finnish",
    "el": "Greek",
}

P0_INSTRUCTION = (
    "Translate the following text from English into {language}.\n"
    "Translate every word of natural language.\n"
    "Keep code, mathematical notation, LaTeX, numbers and identifiers unchanged.\n"
    "Output ONLY the {language} translation — do not solve, answer, add, remove, or explain."
)

JSON_INSTRUCTION = """\
Translate the designated English source units into {language}.

The units are consecutive translation units from one document and are shown in
source order. Use the other visible source units only to keep terminology,
references, tone, and local discourse consistent.

Translate every designated unit completely. Translate rather than answer,
solve, execute, summarize, continue, explain, add, omit, merge, split, reorder,
or repeat content. Do not copy English natural language instead of translating
it.

Keep inline code, mathematical notation, LaTeX, identifiers, numbers, and URLs
unchanged.

The designated zero-based source positions are {target_positions}. Return
exactly one translation for each designated position, in that same order.
Return only the JSON object shown by the output contract, with each ellipsis
replaced by its translation. Do not add fields or array entries.

OUTPUT CONTRACT:
{output_contract}

INPUT:
{input_json}"""

Arm = Literal["P0", "Pc", "RAW", "A1", "A2", "B1", "B2", "B3", "B4"]
SCHEMA_ARMS = frozenset({"A2", "B2", "B4"})
PROMPT_JSON_ARMS = frozenset({"A1", "B1", "B3"})
JSON_ARMS = SCHEMA_ARMS | PROMPT_JSON_ARMS

_WHOLE_JSON_FENCE = re.compile(
    r"\A\s*```(?:json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```[ \t]*\s*\Z",
    re.IGNORECASE,
)


class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


class RequestSpec(TypedDict):
    request_index: int
    input_unit_indices: list[int]
    requested_unit_indices: list[int]
    messages: list[Message] | None
    prompt: str | None
    output_format: Literal["text", "prompt_json", "json_schema"]
    response_format: dict[str, Any] | None
    max_output_tokens: int


@dataclass(frozen=True, slots=True)
class Window:
    start: int
    end: int


def p0_user_message(language_code: str, source_text: str) -> str:
    try:
        language = LANGUAGES[language_code]
    except KeyError as error:
        raise ValueError(f"unsupported language: {language_code}") from error
    return P0_INSTRUCTION.format(language=language) + "\n\n" + source_text


def p0_request(
    request_index: int,
    language_code: str,
    unit: TranslationUnit,
    *,
    history: Sequence[tuple[int, str, str]] = (),
    max_output_tokens: int = 8192,
) -> RequestSpec:
    messages: list[Message] = []
    visible_history = history[-3:]
    for _, source, translation in visible_history:
        messages.extend(
            [
                {"role": "user", "content": p0_user_message(language_code, source)},
                {"role": "assistant", "content": translation},
            ]
        )
    messages.append(
        {"role": "user", "content": p0_user_message(language_code, unit["source_text"])}
    )
    return {
        "request_index": request_index,
        "input_unit_indices": [index for index, _, _ in visible_history] + [unit["unit_index"]],
        "requested_unit_indices": [unit["unit_index"]],
        "messages": messages,
        "prompt": None,
        "output_format": "text",
        "response_format": None,
        "max_output_tokens": max_output_tokens,
    }


def response_schema(output_count: int) -> dict[str, Any]:
    if output_count < 1:
        raise ValueError("output_count must be positive")
    return {
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": output_count,
                "maxItems": output_count,
            }
        },
        "required": ["translations"],
        "additionalProperties": False,
    }


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def json_user_message(
    units: Sequence[TranslationUnit], language_code: str, target_positions: Sequence[int]
) -> str:
    if language_code not in LANGUAGES:
        raise ValueError(f"unsupported language: {language_code}")
    positions = list(target_positions)
    if not positions or positions != sorted(set(positions)):
        raise ValueError("target positions must be unique and ordered")
    if positions[0] < 0 or positions[-1] >= len(units):
        raise ValueError("target position outside visible units")
    return JSON_INSTRUCTION.format(
        language=LANGUAGES[language_code],
        target_positions=_compact_json(positions),
        output_contract=_compact_json({"translations": ["..."] * len(positions)}),
        input_json=_compact_json({"units": [unit["source_text"] for unit in units]}),
    )


def common_windows(
    units: Sequence[TranslationUnit],
    token_count: Callable[[list[Message]], int],
    *,
    prompt_token_limit: int = 8192,
) -> list[Window]:
    if not units:
        return []
    windows: list[Window] = []
    start = 0
    while start < len(units):
        best_end: int | None = None
        for end in range(start + 1, len(units) + 1):
            visible = units[start:end]
            positions = list(range(len(visible)))
            maximum = max(
                token_count(
                    [
                        {
                            "role": "user",
                            "content": json_user_message(visible, language_code, positions),
                        }
                    ]
                )
                for language_code in LANGUAGES
            )
            if maximum > prompt_token_limit:
                break
            best_end = end
        if best_end is None:
            raise ValueError(f"translation unit {start} exceeds the JSON prompt budget")
        windows.append(Window(start, best_end))
        start = best_end
    return windows


def json_requests(
    arm: Arm,
    language_code: str,
    units: Sequence[TranslationUnit],
    windows: Sequence[Window],
    *,
    max_output_tokens: int = 16384,
) -> list[RequestSpec]:
    if arm not in JSON_ARMS:
        raise ValueError(f"not a JSON arm: {arm}")
    groups: list[tuple[int, Sequence[TranslationUnit], list[int]]] = []
    if arm in {"A1", "A2"}:
        groups = [(unit["unit_index"], [unit], [0]) for unit in units]
    else:
        for window_index, window in enumerate(windows):
            visible = units[window.start : window.end]
            if arm in {"B1", "B2"}:
                groups.extend(
                    (unit["unit_index"], visible, [position])
                    for position, unit in enumerate(visible)
                )
            else:
                groups.append((window_index, visible, list(range(len(visible)))))
    requests: list[RequestSpec] = []
    for request_index, visible, positions in groups:
        message: Message = {
            "role": "user",
            "content": json_user_message(visible, language_code, positions),
        }
        requested = [visible[position]["unit_index"] for position in positions]
        schema = response_schema(len(requested)) if arm in SCHEMA_ARMS else None
        requests.append(
            {
                "request_index": request_index,
                "input_unit_indices": [unit["unit_index"] for unit in visible],
                "requested_unit_indices": requested,
                "messages": [message],
                "prompt": None,
                "output_format": "json_schema" if schema else "prompt_json",
                "response_format": (
                    {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "dolci_translation_units",
                            "schema": schema,
                        },
                    }
                    if schema
                    else None
                ),
                "max_output_tokens": max_output_tokens,
            }
        )
    return requests


def parse_json_response(
    raw_output: str, expected_count: int, *, allow_markdown_wrapper: bool
) -> tuple[list[str], str | None]:
    normalized = raw_output
    if match := _WHOLE_JSON_FENCE.fullmatch(raw_output):
        if not allow_markdown_wrapper:
            return [], "markdown_wrapper_not_allowed"
        normalized = match.group("body")
    try:
        value = json.loads(normalized)
    except json.JSONDecodeError as error:
        return [], f"json_decode:{error.msg}"
    if not isinstance(value, Mapping) or set(value) != {"translations"}:
        return [], "object_keys_not_exact"
    translations = value["translations"]
    if not isinstance(translations, list):
        return [], "translations_not_array"
    if len(translations) != expected_count:
        return [], "translation_count_mismatch"
    if any(not isinstance(item, str) for item in translations):
        return [], "translation_not_string"
    if any(not item.strip() for item in translations):
        return [], "blank_translation"
    return translations, None
