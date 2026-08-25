from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    finish_reason: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class Translator(Protocol):
    async def translate(self, text: str, max_tokens: int) -> Completion: ...


@dataclass(frozen=True, slots=True)
class TranslationConfig:
    language: str
    language_code: str
    chunk_chars: int = 2048
    max_tokens: int = 8192
    table_max_tokens: int = 512


@dataclass(frozen=True, slots=True)
class TranslationPair:
    source: str
    candidate: str


@dataclass(frozen=True, slots=True)
class RequestRecord:
    finish_reason: str
    prompt_tokens: int | None
    completion_tokens: int | None
    source_characters: int
    max_tokens: int


@dataclass(slots=True)
class TranslationResult:
    output: str
    pairs: list[TranslationPair]
    requests: list[RequestRecord]
    finish_reasons: list[str]
    truncated_chunks: int
    empty_chunks: int
    prompt_tokens: int | None
    completion_tokens: int | None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "output": self.output,
            "pairs": [[pair.source, pair.candidate] for pair in self.pairs],
            "requests": [
                {
                    "finish_reason": request.finish_reason,
                    "prompt_tokens": request.prompt_tokens,
                    "completion_tokens": request.completion_tokens,
                    "source_characters": request.source_characters,
                    "max_tokens": request.max_tokens,
                }
                for request in self.requests
            ],
            "finish_reasons": self.finish_reasons,
            "truncated_chunks": self.truncated_chunks,
            "empty_chunks": self.empty_chunks,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            **self.metrics,
        }
