import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

from translation_lab.blocks import parse_blocks

MAX_ENCODER_TOKENS = 512
SPECIAL_AND_SAFETY_TOKENS = 8
CONTENT_BUDGET = MAX_ENCODER_TOKENS - SPECIAL_AND_SAFETY_TOKENS
PER_SIDE_BUDGET = CONTENT_BUDGET // 2

Alignment = Literal["whole", "sentence", "positional"]
Provenance = Literal["recorded-pair", "recovered-document"]
TokenCounter = Callable[[str], int]

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…。！？])\s+|\n{2,}")
_WORD_OR_SPACE = re.compile(r"\S+\s*|\s+")
_CONTROL_MARKER = re.compile(r"</?(?:think|reasoning)>")


@dataclass(frozen=True, slots=True)
class ScoringUnit:
    source: str
    candidate: str
    source_tokens: int
    candidate_tokens: int
    alignment: Alignment
    provenance: Provenance
    chunk: int = 0

    @property
    def weight(self) -> int:
        return self.source_tokens + self.candidate_tokens

    @property
    def fits(self) -> bool:
        return self.weight <= CONTENT_BUDGET


def strip_control_markers(text: str) -> str:
    return _CONTROL_MARKER.sub("", text)


def prose_blocks(text: str) -> list[str]:
    return [value for kind, value in parse_blocks(text) if kind == "prose"]


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        parts.append(text[start : match.end()])
        start = match.end()
    if start < len(text):
        parts.append(text[start:])
    return [part for part in parts if part] or ([text] if text else [])


def _character_windows(text: str, count_tokens: TokenCounter) -> list[str]:
    windows: list[str] = []
    start = 0
    while start < len(text):
        low, high = 1, len(text) - start
        while low < high:
            middle = (low + high + 1) // 2
            if count_tokens(text[start : start + middle]) <= PER_SIDE_BUDGET:
                low = middle
            else:
                high = middle - 1
        windows.append(text[start : start + low])
        start += low
    return windows or [text]


def _atoms(text: str, count_tokens: TokenCounter) -> list[str]:
    atoms: list[str] = []
    for sentence in split_sentences(text):
        if count_tokens(sentence) <= PER_SIDE_BUDGET:
            atoms.append(sentence)
            continue
        current = ""
        for word in _WORD_OR_SPACE.findall(sentence):
            pieces = (
                [word]
                if count_tokens(word) <= PER_SIDE_BUDGET
                else _character_windows(word, count_tokens)
            )
            for piece in pieces:
                if current and count_tokens(current + piece) > PER_SIDE_BUDGET:
                    atoms.append(current)
                    current = ""
                current += piece
        if current:
            atoms.append(current)
    return atoms or ([text] if text else [])


def _split_to_budget(text: str, count_tokens: TokenCounter) -> list[str]:
    pieces: list[str] = []
    current = ""
    for atom in _atoms(text, count_tokens):
        if current and count_tokens(current + atom) > PER_SIDE_BUDGET:
            pieces.append(current)
            current = ""
        current += atom
    if current:
        pieces.append(current)
    return pieces or ([text] if text else [])


def _unit(
    source: str,
    candidate: str,
    count_tokens: TokenCounter,
    alignment: Alignment,
    provenance: Provenance,
    chunk: int = 0,
) -> ScoringUnit:
    return ScoringUnit(
        source=source,
        candidate=candidate,
        source_tokens=count_tokens(source),
        candidate_tokens=count_tokens(candidate),
        alignment=alignment,
        provenance=provenance,
        chunk=chunk,
    )


def pack_pair(
    source: str,
    candidate: str,
    count_tokens: TokenCounter,
    chunk: int = 0,
    provenance: Provenance = "recorded-pair",
) -> list[ScoringUnit]:
    source = strip_control_markers(source)
    candidate = strip_control_markers(candidate)
    if not source.strip() and not candidate.strip():
        return []
    whole = _unit(source, candidate, count_tokens, "whole", provenance, chunk)
    if whole.fits:
        return [whole]

    source_sentences = split_sentences(source)
    candidate_sentences = split_sentences(candidate)
    if len(source_sentences) == len(candidate_sentences) and len(source_sentences) > 1:
        packed: list[ScoringUnit] = []
        current_source = ""
        current_candidate = ""
        for source_sentence, candidate_sentence in zip(
            source_sentences, candidate_sentences, strict=True
        ):
            next_source = current_source + source_sentence
            next_candidate = current_candidate + candidate_sentence
            if current_source and (
                count_tokens(next_source) + count_tokens(next_candidate) > CONTENT_BUDGET
            ):
                packed.append(
                    _unit(
                        current_source,
                        current_candidate,
                        count_tokens,
                        "sentence",
                        provenance,
                        chunk,
                    )
                )
                current_source = ""
                current_candidate = ""
            current_source += source_sentence
            current_candidate += candidate_sentence
        if current_source or current_candidate:
            packed.append(
                _unit(
                    current_source,
                    current_candidate,
                    count_tokens,
                    "sentence",
                    provenance,
                    chunk,
                )
            )
        if all(unit.fits for unit in packed):
            return packed

    source_pieces = _split_to_budget(source, count_tokens)
    candidate_pieces = _split_to_budget(candidate, count_tokens)
    length = max(len(source_pieces), len(candidate_pieces))
    source_pieces += [""] * (length - len(source_pieces))
    candidate_pieces += [""] * (length - len(candidate_pieces))
    return [
        _unit(
            source_piece,
            candidate_piece,
            count_tokens,
            "positional",
            provenance,
            chunk,
        )
        for source_piece, candidate_piece in zip(source_pieces, candidate_pieces, strict=True)
        if source_piece or candidate_piece
    ]


def units_from_pairs(
    pairs: Iterable[tuple[str, str]],
    count_tokens: TokenCounter,
    provenance: Provenance = "recorded-pair",
) -> list[ScoringUnit]:
    units: list[ScoringUnit] = []
    for chunk, (source, candidate) in enumerate(pairs):
        units.extend(pack_pair(source, candidate, count_tokens, chunk, provenance))
    return units


def units_from_documents(
    source: str, candidate: str, count_tokens: TokenCounter
) -> list[ScoringUnit]:
    source_prose = prose_blocks(source)
    candidate_prose = prose_blocks(candidate)
    if len(source_prose) == len(candidate_prose) and source_prose:
        return units_from_pairs(
            zip(source_prose, candidate_prose, strict=True),
            count_tokens,
            "recovered-document",
        )
    return pack_pair(
        "".join(source_prose),
        "".join(candidate_prose),
        count_tokens,
        provenance="recovered-document",
    )


def aggregate_scores(units: Sequence[ScoringUnit], scores: Sequence[float]) -> tuple[float, float]:
    if len(units) != len(scores):
        raise ValueError(f"received {len(scores)} scores for {len(units)} units")
    if not units:
        raise ValueError("cannot aggregate an empty unit list")
    total_weight = sum(unit.weight for unit in units)
    if not total_weight:
        raise ValueError("cannot aggregate zero-token units")
    mean = sum(unit.weight * score for unit, score in zip(units, scores, strict=True))
    return mean / total_weight, min(scores)


def load_comet(
    checkpoint: Path, gpus: int = 1
) -> tuple[Callable[[list[dict[str, str]]], list[float]], TokenCounter]:
    if gpus not in {0, 1}:
        raise ValueError("COMET scoring uses zero or one GPU to preserve prediction order")
    try:
        comet = import_module("comet")
    except ImportError as error:
        raise RuntimeError("install Unbabel COMET in the scoring environment") from error

    checkpoint_file = next(checkpoint.rglob("*.ckpt"), checkpoint)
    model = comet.load_from_checkpoint(str(checkpoint_file))
    tokenizer: Any
    try:
        tokenizer = model.encoder.tokenizer
    except AttributeError:
        transformers = import_module("transformers")
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
    if getattr(tokenizer, "fix_mistral_regex", False) is True:
        raise RuntimeError("fix_mistral_regex=True changes the checkpoint's InfoXLM tokenization")
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise RuntimeError("the COMET checkpoint tokenizer has no encode method")

    def count_tokens(text: str) -> int:
        if not text:
            return 0
        encoded: Any = encode(text, add_special_tokens=False)
        return len(encoded)

    def predict(rows: list[dict[str, str]]) -> list[float]:
        prediction = model.predict(rows, batch_size=16, gpus=gpus)
        return [float(score) for score in prediction.scores]

    return predict, count_tokens
