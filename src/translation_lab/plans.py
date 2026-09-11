import re
from collections.abc import Mapping
from typing import Literal, TypedDict

from translation_lab.blocks import parse_blocks, split_prose, split_table, translatable_cell

P0_PLAN_REVISION = "p0_sectioned_v1"
RAW_PLAN_REVISION = "raw_separators_v1"

UnitKind = Literal["prose", "table_cell", "raw"]


class TranslationUnit(TypedDict):
    unit_index: int
    kind: UnitKind
    source_text: str


class ReconstructionPiece(TypedDict):
    kind: Literal["literal", "translation_unit"]
    text: str | None
    unit_index: int | None


class ReconstructionSection(TypedDict):
    section_index: int
    include_if_nonempty: bool
    prefix: str
    suffix: str
    strip_content: bool
    pieces: list[ReconstructionPiece]


class Reconstruction(TypedDict):
    section_separator: str
    sections: list[ReconstructionSection]


class TranslationPlan(TypedDict):
    source_id: str
    method: Literal["P0", "RAW", "SB"]
    plan_revision: str
    chunk_label: int
    chunk_target_chars: int
    translation_units: list[TranslationUnit]
    reconstruction: Reconstruction


_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_BLANK_LINE = re.compile(r"(?:[ \t]*\r?\n){2,}[ \t]*")
_SENTENCE_SPACE = re.compile(r"(?<=[.!?])([ \t\r\n]+)")
_LINE_BREAK = re.compile(r"\r?\n")
_OTHER_SPACE = re.compile(r"[ \t\f\v]+")


def _piece(
    kind: Literal["literal", "translation_unit"],
    *,
    text: str | None = None,
    unit_index: int | None = None,
) -> ReconstructionPiece:
    return {"kind": kind, "text": text, "unit_index": unit_index}


def _split_think(text: str) -> tuple[str, str, str]:
    match = _THINK.search(text)
    if match is None:
        return "", "", text
    return text[: match.start()], match.group(1), text[match.end() :]


def _p0_section(
    section_index: int,
    source_text: str,
    reasoning: bool,
    units: list[TranslationUnit],
    chunk_target_chars: int,
) -> ReconstructionSection:
    pieces: list[ReconstructionPiece] = []
    for kind, block in parse_blocks(source_text):
        if kind == "prose":
            leading, chunks, separators, trailing = split_prose(block, chunk_target_chars)
            if leading:
                pieces.append(_piece("literal", text=leading))
            for index, chunk in enumerate(chunks):
                unit_index = len(units)
                units.append({"unit_index": unit_index, "kind": "prose", "source_text": chunk})
                pieces.append(_piece("translation_unit", unit_index=unit_index))
                if index < len(separators) and separators[index]:
                    pieces.append(_piece("literal", text=separators[index]))
            if trailing:
                pieces.append(_piece("literal", text=trailing))
        elif kind == "table":
            for token_kind, token in split_table(block):
                if token_kind == "cell" and translatable_cell(token):
                    leading = token[: len(token) - len(token.lstrip())]
                    trailing = token[len(token.rstrip()) :]
                    if leading:
                        pieces.append(_piece("literal", text=leading))
                    unit_index = len(units)
                    units.append(
                        {
                            "unit_index": unit_index,
                            "kind": "table_cell",
                            "source_text": token.strip(),
                        }
                    )
                    pieces.append(_piece("translation_unit", unit_index=unit_index))
                    if trailing:
                        pieces.append(_piece("literal", text=trailing))
                else:
                    pieces.append(_piece("literal", text=token))
        else:
            pieces.append(_piece("literal", text=block))
    return {
        "section_index": section_index,
        "include_if_nonempty": not (reasoning and bool(source_text.strip())),
        "prefix": "<think>\n" if reasoning else "",
        "suffix": "\n</think>" if reasoning else "",
        "strip_content": True,
        "pieces": pieces,
    }


def p0_plan(
    source_id: str,
    source_text: str,
    *,
    chunk_label: int = 512,
    chunk_target_chars: int = 2048,
) -> TranslationPlan:
    before, reasoning, after = _split_think(source_text)
    sections = (
        ((before, False), (reasoning, True), (after, False))
        if _THINK.search(source_text)
        else ((after, False),)
    )
    units: list[TranslationUnit] = []
    reconstruction_sections = [
        _p0_section(index, text, is_reasoning, units, chunk_target_chars)
        for index, (text, is_reasoning) in enumerate(sections)
    ]
    plan: TranslationPlan = {
        "source_id": source_id,
        "method": "P0",
        "plan_revision": P0_PLAN_REVISION,
        "chunk_label": chunk_label,
        "chunk_target_chars": chunk_target_chars,
        "translation_units": units,
        "reconstruction": {
            "section_separator": "\n\n",
            "sections": reconstruction_sections,
        },
    }
    validate_plan(plan, source_text)
    return plan


def _latest_raw_separator(text: str, start: int, limit: int) -> tuple[int, int] | None:
    prefix = text[start:limit]
    for pattern in (_BLANK_LINE, _SENTENCE_SPACE, _LINE_BREAK, _OTHER_SPACE):
        for match in reversed(list(pattern.finditer(prefix))):
            content_end = start + match.start()
            if content_end > start and text[start:content_end].strip():
                return content_end, start + match.end()
    return None


def split_raw_text(source_text: str, target_chars: int = 2048) -> tuple[list[str], list[str]]:
    if target_chars < 1:
        raise ValueError("target_chars must be positive")
    if not source_text:
        return [], []
    units: list[str] = []
    separators: list[str] = []
    position = 0
    while len(source_text) - position > target_chars:
        boundary = _latest_raw_separator(source_text, position, position + target_chars)
        if boundary is None:
            content_end = separator_end = position + target_chars
        else:
            content_end, separator_end = boundary
        unit = source_text[position:content_end]
        if not unit.strip():
            content_end = separator_end = min(len(source_text), position + target_chars)
            unit = source_text[position:content_end]
        units.append(unit)
        separators.append(source_text[content_end:separator_end])
        position = separator_end
    remainder = source_text[position:]
    if remainder:
        if not remainder.strip() and units:
            separators[-1] += remainder
        else:
            units.append(remainder)
    if len(separators) != max(0, len(units) - 1):
        raise AssertionError("RAW unit/separator cardinality mismatch")
    if any(not unit.strip() or len(unit) > target_chars for unit in units):
        raise AssertionError("RAW produced an invalid translation unit")
    if (
        "".join(
            unit + (separators[index] if index < len(separators) else "")
            for index, unit in enumerate(units)
        )
        != source_text
    ):
        raise AssertionError("RAW plan does not cover the source exactly")
    return units, separators


def raw_plan(
    source_id: str,
    source_text: str,
    *,
    chunk_label: int = 512,
    chunk_target_chars: int = 2048,
    method: Literal["RAW", "SB"] = "RAW",
) -> TranslationPlan:
    source_units, separators = split_raw_text(source_text, chunk_target_chars)
    units: list[TranslationUnit] = [
        {"unit_index": index, "kind": "raw", "source_text": text}
        for index, text in enumerate(source_units)
    ]
    pieces: list[ReconstructionPiece] = []
    for index in range(len(units)):
        pieces.append(_piece("translation_unit", unit_index=index))
        if index < len(separators) and separators[index]:
            pieces.append(_piece("literal", text=separators[index]))
    plan: TranslationPlan = {
        "source_id": source_id,
        "method": method,
        "plan_revision": RAW_PLAN_REVISION,
        "chunk_label": chunk_label,
        "chunk_target_chars": chunk_target_chars,
        "translation_units": units,
        "reconstruction": {
            "section_separator": "",
            "sections": [
                {
                    "section_index": 0,
                    "include_if_nonempty": False,
                    "prefix": "",
                    "suffix": "",
                    "strip_content": False,
                    "pieces": pieces,
                }
            ],
        },
    }
    validate_plan(plan, source_text)
    return plan


def reconstruct(plan: TranslationPlan, translations: Mapping[int, str]) -> str:
    sections: list[str] = []
    for section in plan["reconstruction"]["sections"]:
        content: list[str] = []
        for piece in section["pieces"]:
            if piece["kind"] == "literal":
                content.append(piece["text"] or "")
            else:
                unit_index = piece["unit_index"]
                if unit_index is None or unit_index not in translations:
                    raise KeyError(f"missing translation unit {unit_index}")
                content.append(translations[unit_index])
        body = "".join(content)
        if section["strip_content"]:
            body = body.strip()
        if section["include_if_nonempty"] and not body:
            continue
        sections.append(section["prefix"] + body + section["suffix"])
    return plan["reconstruction"]["section_separator"].join(sections)


def validate_plan(plan: TranslationPlan, source_text: str | None = None) -> None:
    units = plan["translation_units"]
    if [unit["unit_index"] for unit in units] != list(range(len(units))):
        raise ValueError("translation-unit indices must be contiguous")
    identity = {unit["unit_index"]: unit["source_text"] for unit in units}
    rebuilt = reconstruct(plan, identity)
    if source_text is not None:
        expected = source_text
        if plan["method"] == "P0":
            before, reasoning, after = _split_think(source_text)
            parts = []
            if before.strip():
                parts.append(before.strip())
            if reasoning.strip():
                parts.append(f"<think>\n{reasoning.strip()}\n</think>")
            if after.strip():
                parts.append(after.strip())
            expected = "\n\n".join(parts)
        if rebuilt != expected:
            raise ValueError("identity reconstruction differs from the source contract")


def processed_output(method: str, kind: UnitKind, raw_output: str) -> str:
    if method in {"RAW", "SB"}:
        return raw_output
    if kind == "prose":
        from translation_lab.blocks import strip_generated_fences

        return strip_generated_fences(raw_output).strip()
    if kind == "table_cell":
        return raw_output.strip()
    raise ValueError(f"unsupported P0 unit kind: {kind}")
