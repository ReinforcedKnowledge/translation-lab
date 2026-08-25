import bisect
import re
from typing import Literal

BlockKind = Literal["prose", "code", "math", "table"]
type Block = tuple[BlockKind, str]
type TablePart = tuple[Literal["literal", "cell"], str]

_DISPLAY_MATH = re.compile(r"\$\$.*?\$\$|\\\[.*?\\\]", re.DOTALL)
_PIPE_ROW = re.compile(r"^[ \t]*\|.*\|[ \t]*\r?\n?$")
_SEPARATOR_ROW = re.compile(r"^[ \t]*\|[ \t:|]*-[ \t:|-]*\|[ \t]*\r?\n?$")
_WORD = re.compile(r"[^\W\d_]{2,}", re.UNICODE)
_FENCE_OPEN = re.compile(r"^[ \t]*((?P<marker>[`~])(?P=marker){2,})")
_FENCE_CLOSE = re.compile(r"^[ \t]*([`~])\1{2,}[ \t]*\r?\n?$")
_FENCE_LINE = re.compile(r"^[ \t]*(?:`{3,}|~{3,})[^\n]*\n?", re.MULTILINE)
_CONTROL_TAG = re.compile(r"(</?(?:think|reasoning)>)")


def _column_count(line: str) -> int:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return 0
    return len(stripped.split("|")) - 2


def _is_table(rows: list[str]) -> bool:
    if any(_SEPARATOR_ROW.match(row) for row in rows):
        return True
    counts = [_column_count(row) for row in rows]
    mode = max(set(counts), key=counts.count)
    has_word = any(_WORD.search(cell) for row in rows for cell in row.strip().strip("|").split("|"))
    # A numeric separatorless table is structurally indistinguishable from a matrix.
    return mode >= 2 and counts.count(mode) >= 2 and has_word


def _split_tables(text: str) -> list[Block]:
    lines = text.splitlines(keepends=True)
    blocks: list[Block] = []
    prose: list[str] = []
    index = 0
    while index < len(lines):
        if not _PIPE_ROW.match(lines[index]):
            prose.append(lines[index])
            index += 1
            continue
        end = index
        while end < len(lines) and _PIPE_ROW.match(lines[end]):
            end += 1
        rows = lines[index:end]
        if len(rows) >= 2 and _is_table(rows):
            if prose:
                blocks.append(("prose", "".join(prose)))
                prose = []
            blocks.append(("table", "".join(rows)))
        else:
            prose.extend(rows)
        index = end
    if prose:
        blocks.append(("prose", "".join(prose)))
    return blocks


def _split_math_and_tables(text: str) -> list[Block]:
    blocks: list[Block] = []
    last = 0
    for match in _DISPLAY_MATH.finditer(text):
        if match.start() > last:
            blocks.extend(_split_tables(text[last : match.start()]))
        blocks.append(("math", match.group()))
        last = match.end()
    if last < len(text):
        blocks.extend(_split_tables(text[last:]))
    return blocks


def parse_blocks(text: str) -> list[Block]:
    lines = text.splitlines(keepends=True)
    closers: dict[str, list[tuple[int, int]]] = {"`": [], "~": []}
    closer_indexes: dict[str, list[int]] = {"`": [], "~": []}
    for index, line in enumerate(lines):
        if match := _FENCE_CLOSE.match(line):
            run = match.group(0).strip()
            closers[run[0]].append((index, len(run)))
            closer_indexes[run[0]].append(index)
    longest = {
        marker: max((length for _, length in values), default=0)
        for marker, values in closers.items()
    }

    blocks: list[Block] = []
    gap: list[str] = []

    def flush_gap() -> None:
        if gap:
            blocks.extend(_split_math_and_tables("".join(gap)))
            gap.clear()

    index = 0
    while index < len(lines):
        opening = _FENCE_OPEN.match(lines[index])
        closing_index: int | None = None
        if opening:
            run = opening.group(1)
            marker = run[0]
            if len(run) <= longest[marker]:
                candidate = bisect.bisect_right(closer_indexes[marker], index)
                for line_index, length in closers[marker][candidate:]:
                    if length >= len(run):
                        closing_index = line_index
                        break
        if closing_index is None:
            gap.append(lines[index])
            index += 1
            continue
        flush_gap()
        blocks.append(("code", "".join(lines[index : closing_index + 1])))
        index = closing_index + 1
    flush_gap()
    return blocks or [("prose", text)]


def split_control_tags(text: str) -> list[tuple[Literal["text", "tag"], str]]:
    parts = _CONTROL_TAG.split(text)
    return [("tag" if _CONTROL_TAG.fullmatch(part) else "text", part) for part in parts if part]


def split_table(table: str) -> list[TablePart]:
    parts: list[TablePart] = []
    for line in table.splitlines(keepends=True):
        if re.match(r"^[ \t]*\|[\s:\-|]+\|[ \t]*\r?\n?$", line):
            parts.append(("literal", line))
            continue
        for part in re.split(r"(\|)", line):
            kind: Literal["literal", "cell"] = (
                "literal" if part == "|" or not part.strip() else "cell"
            )
            parts.append((kind, part))
    return parts


def _balanced_inline_spans(text: str) -> bool:
    return text.count("$") % 2 == 0 and text.count("`") % 2 == 0


def _hard_split(text: str, target: int) -> tuple[list[str], list[str]]:
    if len(text) <= target * 2:
        return [text], []
    chunks: list[str] = []
    separators: list[str] = []
    start = 0
    while len(text) - start > target * 2:
        window = text[start : start + int(target * 1.5)]
        sentence_ends = [match.end() for match in re.finditer(r"[.!?…]\s", window)]
        cut = next(
            (
                end
                for end in reversed(sentence_ends)
                if _balanced_inline_spans(text[start : start + end])
            ),
            None,
        )
        if cut is None:
            whitespace = [match.start() + 1 for match in re.finditer(r"\s", window)]
            cut = next(
                (
                    end
                    for end in reversed(whitespace)
                    if _balanced_inline_spans(text[start : start + end])
                ),
                None,
            )
        cut = max(1, cut or len(window))
        piece = text[start : start + cut]
        stripped = piece.rstrip()
        chunks.append(stripped)
        separators.append(piece[len(stripped) :])
        start += cut
    chunks.append(text[start:])
    return chunks, separators


def split_prose(text: str, target_chars: int = 2048) -> tuple[str, list[str], list[str], str]:
    if target_chars < 1:
        raise ValueError("target_chars must be positive")
    core = text.strip("\n\r \t")
    if not core:
        return text, [], [], ""
    start = text.index(core[0])
    end = start + len(core)
    leading, trailing = text[:start], text[end:]
    parts = re.split(r"(\n(?:[ \t]*\n)+)", core)
    chunks: list[str] = []
    separators: list[str] = []
    current = ""
    for index, part in enumerate(parts):
        if index % 2:
            if len(current) >= target_chars:
                chunks.append(current)
                separators.append(part)
                current = ""
            else:
                current += part
        else:
            current += part
    if current:
        chunks.append(current)

    bounded_chunks: list[str] = []
    bounded_separators: list[str] = []
    for index, chunk in enumerate(chunks):
        subchunks, subseparators = _hard_split(chunk, target_chars)
        bounded_chunks.extend(subchunks)
        bounded_separators.extend(subseparators)
        if index < len(separators):
            bounded_separators.append(separators[index])
    return leading, bounded_chunks, bounded_separators, trailing


def translatable_cell(cell: str) -> bool:
    stripped = cell.strip()
    if not re.search(r"[A-Za-z]{2}", stripped):
        return False
    return not (" " not in stripped and re.search(r"[\d_@?{}\\/]", stripped))


def pad_like(original: str, translated: str) -> str:
    if not original.strip():
        return original
    leading = original[: len(original) - len(original.lstrip())]
    trailing = original[len(original.rstrip()) :]
    return leading + translated.strip() + trailing


def strip_generated_fences(text: str) -> str:
    return _FENCE_LINE.sub("", text)


def reassemble_prose(leading: str, chunks: list[str], separators: list[str], trailing: str) -> str:
    body = "".join(
        chunk + (separators[index] if index < len(separators) else "")
        for index, chunk in enumerate(chunks)
    )
    return leading + body + trailing
