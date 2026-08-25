import asyncio
import json
from typing import Any, cast

from openai import AsyncOpenAI

from translation_lab.models import Completion
from translation_lab.protocols import RequestSpec

PROMPT = (
    "Translate the following text from English into {language}.\n"
    "Translate every word of natural language.\n"
    "Keep code, mathematical notation, LaTeX, numbers, and identifiers unchanged.\n"
    "Output only the {language} translation. Do not solve, answer, add, remove, or explain.\n\n"
    "{text}"
)


class OpenAITranslator:
    def __init__(
        self, client: AsyncOpenAI, model: str, language: str, max_concurrency: int
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        self._client = client
        self._model = model
        self._language = language
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def translate(self, text: str, max_tokens: int) -> Completion:
        async with self._semaphore:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": PROMPT.format(language=self._language, text=text),
                    }
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
        choice = response.choices[0]
        usage = response.usage
        return Completion(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason or "unknown",
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )


class OpenAIReviewer:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        max_tokens: int,
        max_concurrency: int,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def review(self, rubric: str, prompt: str) -> dict[str, Any]:
        async with self._semaphore:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": rubric},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_completion_tokens=self._max_tokens,
                response_format={"type": "json_object"},
            )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("the reviewer returned an empty response")
        result = json.loads(content)
        if not isinstance(result, dict):
            raise ValueError("the reviewer did not return a JSON object")
        return result


class OpenAIGenerator:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        max_concurrency: int,
        *,
        temperature: float = 0.0,
        seed: int = 42,
        extra_body: dict[str, Any] | None = None,
        generation_options: dict[str, Any] | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        self._client = client
        self._model = model
        self._temperature = temperature
        self._seed = seed
        self._extra_body = extra_body or {}
        self._generation_options = generation_options or {}
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def generate(self, request: RequestSpec) -> Completion:
        common: dict[str, Any] = {
            **self._generation_options,
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": request["max_output_tokens"],
            "seed": self._seed,
        }
        extra_body = dict(self._extra_body)
        if request["messages"] is not None:
            if request["response_format"] is not None:
                common["response_format"] = request["response_format"]
            if extra_body:
                common["extra_body"] = extra_body
            async with self._semaphore:
                response = cast(
                    Any,
                    await self._client.chat.completions.create(
                        messages=cast(Any, request["messages"]), **common
                    ),
                )
            choice = response.choices[0]
            content = choice.message.content or ""
        else:
            prompt = request["prompt"]
            if prompt is None:
                raise ValueError("request contains neither messages nor prompt")
            extra_body.setdefault("add_special_tokens", False)
            common["extra_body"] = extra_body
            async with self._semaphore:
                response = cast(
                    Any,
                    await self._client.completions.create(prompt=prompt, **common),
                )
            choice = response.choices[0]
            content = choice.text
        usage = response.usage
        return Completion(
            text=content,
            finish_reason=choice.finish_reason or "unknown",
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )


async def create_translator(
    base_url: str,
    language: str,
    model: str | None = None,
    api_key: str = "unused",
    timeout: float = 1800.0,
    max_concurrency: int = 8,
) -> OpenAITranslator:
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=1)
    if model is None:
        models = await client.models.list()
        if not models.data:
            raise RuntimeError("the inference server returned no models")
        model = models.data[0].id
    return OpenAITranslator(client, model, language, max_concurrency)


async def create_generator(
    base_url: str,
    model: str | None = None,
    api_key: str = "unused",
    timeout: float = 1800.0,
    max_concurrency: int = 8,
    *,
    temperature: float = 0.0,
    seed: int = 42,
    extra_body: dict[str, Any] | None = None,
    generation_options: dict[str, Any] | None = None,
) -> OpenAIGenerator:
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=1)
    if model is None:
        models = await client.models.list()
        if not models.data:
            raise RuntimeError("the inference server returned no models")
        model = models.data[0].id
    return OpenAIGenerator(
        client,
        model,
        max_concurrency,
        temperature=temperature,
        seed=seed,
        extra_body=extra_body,
        generation_options=generation_options,
    )


def create_reviewer(
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 1800.0,
    max_tokens: int = 4096,
    max_concurrency: int = 4,
) -> OpenAIReviewer:
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=1)
    return OpenAIReviewer(client, model, max_tokens, max_concurrency)
