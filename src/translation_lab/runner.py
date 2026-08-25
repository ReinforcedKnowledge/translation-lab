import asyncio

from translation_lab.blocks import (
    pad_like,
    parse_blocks,
    reassemble_prose,
    split_control_tags,
    split_prose,
    split_table,
    strip_generated_fences,
    translatable_cell,
)
from translation_lab.metrics import audit_translation
from translation_lab.models import (
    Completion,
    RequestRecord,
    TranslationConfig,
    TranslationPair,
    TranslationResult,
    Translator,
)


class _Accumulator:
    def __init__(self) -> None:
        self.pairs: list[TranslationPair] = []
        self.requests: list[RequestRecord] = []
        self.finish_reasons: list[str] = []
        self.truncated = 0
        self.empty = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.has_prompt_tokens = True
        self.has_completion_tokens = True

    def add(self, completion: Completion, source: str, candidate: str, max_tokens: int) -> None:
        self.requests.append(
            RequestRecord(
                completion.finish_reason,
                completion.prompt_tokens,
                completion.completion_tokens,
                len(source),
                max_tokens,
            )
        )
        self.finish_reasons.append(completion.finish_reason)
        self.truncated += completion.finish_reason == "length"
        self.empty += not candidate
        if completion.prompt_tokens is None:
            self.has_prompt_tokens = False
        else:
            self.prompt_tokens += completion.prompt_tokens
        if completion.completion_tokens is None:
            self.has_completion_tokens = False
        else:
            self.completion_tokens += completion.completion_tokens
        if source and candidate:
            self.pairs.append(TranslationPair(source, candidate))


async def _translate_prose(
    text: str,
    translator: Translator,
    config: TranslationConfig,
    accumulator: _Accumulator,
) -> str:
    leading, chunks, separators, trailing = split_prose(text, config.chunk_chars)
    if not chunks:
        return text
    completions = await asyncio.gather(
        *(translator.translate(chunk, config.max_tokens) for chunk in chunks)
    )
    outputs: list[str] = []
    for source, completion in zip(chunks, completions, strict=True):
        candidate = strip_generated_fences(completion.text).strip()
        accumulator.add(completion, source.strip(), candidate, config.max_tokens)
        outputs.append(candidate)
    return reassemble_prose(leading, outputs, separators, trailing)


async def _translate_table(
    text: str,
    translator: Translator,
    config: TranslationConfig,
    accumulator: _Accumulator,
) -> str:
    parts = split_table(text)
    indexes = [
        index
        for index, (kind, value) in enumerate(parts)
        if kind == "cell" and translatable_cell(value)
    ]
    completions = await asyncio.gather(
        *(
            translator.translate(parts[index][1].strip(), config.table_max_tokens)
            for index in indexes
        )
    )
    rendered = [value for _, value in parts]
    for index, completion in zip(indexes, completions, strict=True):
        source = parts[index][1].strip()
        candidate = strip_generated_fences(completion.text).strip()
        accumulator.add(completion, source, candidate, config.table_max_tokens)
        rendered[index] = pad_like(parts[index][1], candidate)
    return "".join(rendered)


async def _translate_tagged_prose(
    text: str,
    translator: Translator,
    config: TranslationConfig,
    accumulator: _Accumulator,
) -> str:
    rendered: list[str] = []
    for kind, part in split_control_tags(text):
        if kind == "tag":
            rendered.append(part)
        else:
            rendered.append(await _translate_prose(part, translator, config, accumulator))
    return "".join(rendered)


async def translate_document(
    text: str, translator: Translator, config: TranslationConfig
) -> TranslationResult:
    accumulator = _Accumulator()
    rendered: list[str] = []
    for kind, part in parse_blocks(text):
        if kind == "prose":
            rendered.append(await _translate_tagged_prose(part, translator, config, accumulator))
        elif kind == "table":
            rendered.append(await _translate_table(part, translator, config, accumulator))
        else:
            rendered.append(part)
    output = "".join(rendered)
    source_verbatim = [value for kind, value in parse_blocks(text) if kind in {"code", "math"}]
    metrics = audit_translation(text, output, config.language_code)
    metrics["verbatim_blocks_preserved"] = all(value in output for value in source_verbatim)
    return TranslationResult(
        output=output,
        pairs=accumulator.pairs,
        requests=accumulator.requests,
        finish_reasons=accumulator.finish_reasons,
        truncated_chunks=accumulator.truncated,
        empty_chunks=accumulator.empty,
        prompt_tokens=accumulator.prompt_tokens if accumulator.has_prompt_tokens else None,
        completion_tokens=(
            accumulator.completion_tokens if accumulator.has_completion_tokens else None
        ),
        metrics=metrics,
    )
