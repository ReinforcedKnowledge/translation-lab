import asyncio

from translation_lab.models import Completion, TranslationConfig
from translation_lab.runner import translate_document


class IdentityTranslator:
    async def translate(self, text: str, max_tokens: int) -> Completion:
        return Completion(text, "stop", 10, 5)


class FencedTranslator:
    async def translate(self, text: str, max_tokens: int) -> Completion:
        return Completion(f"```french\n{text.upper()}\n```", "length", 7, 3)


def config() -> TranslationConfig:
    return TranslationConfig("French", "fr", chunk_chars=40, max_tokens=100)


def test_identity_translation_reconstructs_document_exactly() -> None:
    source = (
        "Before.\n\n<think>Reasoning.\n\n"
        "```python\nprint('keep')\n```\n\n$$x+y$$</think>\n\n"
        "| Label | Value |\n| --- | --- |\n| Natural language | 15 |\n"
    )
    result = asyncio.run(translate_document(source, IdentityTranslator(), config()))

    assert result.output == source
    assert result.metrics["verbatim_blocks_preserved"] is True
    assert result.metrics["structure_preserved"] is True
    assert result.truncated_chunks == 0
    assert result.empty_chunks == 0
    assert result.prompt_tokens == 50
    assert result.completion_tokens == 25
    assert len(result.requests) == 5
    assert all(request.finish_reason == "stop" for request in result.requests)
    assert all("```" not in pair.source and "$$" not in pair.source for pair in result.pairs)


def test_generated_prose_fences_are_removed_and_length_stops_are_counted() -> None:
    result = asyncio.run(translate_document("One short sentence.", FencedTranslator(), config()))

    assert result.output == "ONE SHORT SENTENCE."
    assert result.finish_reasons == ["length"]
    assert result.truncated_chunks == 1
    assert result.prompt_tokens == 7
    assert result.completion_tokens == 3


def test_control_tag_text_inside_code_is_never_sent_to_translator() -> None:
    source = "```xml\n<think>literal</think>\n```"
    result = asyncio.run(translate_document(source, FencedTranslator(), config()))

    assert result.output == source
    assert result.requests == []
