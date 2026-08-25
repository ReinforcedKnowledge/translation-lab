from dataclasses import dataclass
from typing import Any, Literal

from translation_lab.plans import TranslationUnit
from translation_lab.protocols import LANGUAGES, Message, RequestSpec, p0_user_message

ModelInterface = Literal["gemma", "qwen3.8", "translategemma", "milmmt", "hy-mt2"]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    checkpoint: str
    interface: ModelInterface
    temperature: float
    max_output_tokens: int
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    stop_token_ids: tuple[int, ...] = ()
    supported_languages: tuple[str, ...] = tuple(LANGUAGES)
    maximum_input_tokens: int | None = None
    revision: str | None = None


MODEL_SPECS = {
    "gemma3": ModelSpec("RedHatAI/gemma-3-27b-it-FP8-dynamic", "gemma", 0.0, 8192),
    "gemma4": ModelSpec("RedHatAI/gemma-4-31B-it-FP8-Dynamic", "gemma", 0.0, 8192),
    "qwen3.8": ModelSpec(
        "Qwen/Qwen3.8-27B-FP8",
        "qwen3.8",
        0.0,
        8192,
        stop_token_ids=(248046, 248044),
        revision="017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
    ),
    "translategemma": ModelSpec(
        "google/translategemma-27b-it",
        "translategemma",
        0.0,
        4096,
        maximum_input_tokens=2048,
        revision="7d10f0b72f89a2d0f268cea30727d8b77c0d25c2",
    ),
    "milmmt": ModelSpec(
        "xiaomi-research/MiLMMT-46-12B-v0.1",
        "milmmt",
        0.0,
        4096,
        top_k=1,
        revision="17f12a26ad8904efe740b1217fe70d61240bafe9",
    ),
    "hy-mt2": ModelSpec(
        "tencent/Hy-MT2-30B-A3B",
        "hy-mt2",
        0.7,
        4096,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
        supported_languages=("pl", "de", "fr", "es"),
        revision="d3ead4dba61c09aac60a261a96ad1df3e705febb",
    ),
}

_TRANSLATEGEMMA_CODES = {
    "pl": "pl-PL",
    "de": "de-DE",
    "fr": "fr-FR",
    "es": "es-ES",
    "fi": "fi-FI",
    "el": "el-GR",
}


def native_input(interface: ModelInterface, language_code: str, source_text: str) -> Any:
    if language_code not in LANGUAGES:
        raise ValueError(f"unsupported language: {language_code}")
    language = LANGUAGES[language_code]
    if interface in {"gemma", "qwen3.8"}:
        return [{"role": "user", "content": p0_user_message(language_code, source_text)}]
    if interface == "translategemma":
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": "en",
                        "target_lang_code": _TRANSLATEGEMMA_CODES[language_code],
                        "text": source_text,
                    }
                ],
            }
        ]
    if interface == "milmmt":
        return f"Translate this from English to {language}:\nEnglish: {source_text}\n{language}:"
    if interface == "hy-mt2":
        return [
            {
                "role": "user",
                "content": (
                    f"Translate the following text into {language}. "
                    "Note that you should only output the translated result without any "
                    f"additional explanation:\n\n{source_text}"
                ),
            }
        ]
    raise ValueError(f"unknown model interface: {interface}")


def render_native_prompt(interface: ModelInterface, renderer: Any, native: Any) -> str:
    if interface == "milmmt":
        return str(native)
    options = {"enable_thinking": False} if interface == "qwen3.8" else {}
    return str(
        renderer.apply_chat_template(native, tokenize=False, add_generation_prompt=True, **options)
    )


def _input_ids(value: Any) -> list[int]:
    if isinstance(value, dict) or hasattr(value, "keys"):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("unexpected batched template tokenization")
        value = value[0]
    return list(value)


def _verify_rendering(interface: ModelInterface, renderer: Any, native: Any, prompt: str) -> int:
    tokenizer = renderer.tokenizer if interface == "translategemma" else renderer
    options = {"enable_thinking": False} if interface == "qwen3.8" else {}
    template_ids = _input_ids(
        renderer.apply_chat_template(native, tokenize=True, add_generation_prompt=True, **options)
    )
    completion_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    if template_ids != completion_ids:
        raise ValueError("rendered completion tokens differ from native-template tokens")
    return len(template_ids)


def generation_parameters(
    spec: ModelSpec, *, max_output_tokens: int | None = None
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "temperature": spec.temperature,
        "max_tokens": max_output_tokens or spec.max_output_tokens,
        "seed": 42,
    }
    if spec.top_p is not None:
        parameters["top_p"] = spec.top_p
    extra: dict[str, Any] = {"add_special_tokens": False}
    if spec.top_k is not None:
        extra["top_k"] = spec.top_k
    if spec.repetition_penalty is not None:
        extra["repetition_penalty"] = spec.repetition_penalty
    if spec.stop_token_ids:
        extra["stop_token_ids"] = list(spec.stop_token_ids)
    parameters["extra_body"] = extra
    return parameters


def as_messages(value: Any) -> list[Message]:
    if not isinstance(value, list):
        raise TypeError("native input is not a message list")
    result: list[Message] = []
    for message in value:
        if not isinstance(message, dict):
            raise TypeError("native message must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant", "system"} or not isinstance(content, str):
            raise TypeError("native message is not compatible with the chat endpoint")
        result.append({"role": role, "content": content})
    return result


def native_request(
    spec: ModelSpec,
    language_code: str,
    unit: TranslationUnit,
    *,
    request_index: int,
    renderer: Any | None = None,
    max_output_tokens: int | None = None,
) -> RequestSpec:
    native = native_input(spec.interface, language_code, unit["source_text"])
    messages: list[Message] | None = None
    prompt: str | None = None
    if spec.interface in {"gemma", "qwen3.8", "hy-mt2"} and renderer is None:
        messages = as_messages(native)
    elif spec.interface == "milmmt":
        prompt = str(native)
    else:
        if renderer is None:
            raise ValueError("TranslateGemma requires its local AutoProcessor")
        prompt = render_native_prompt(spec.interface, renderer, native)
        token_count = _verify_rendering(spec.interface, renderer, native, prompt)
        if spec.maximum_input_tokens is not None and token_count > spec.maximum_input_tokens:
            raise ValueError(
                f"request has {token_count} input tokens; {spec.checkpoint} declares "
                f"a {spec.maximum_input_tokens}-token input condition"
            )
    return {
        "request_index": request_index,
        "input_unit_indices": [unit["unit_index"]],
        "requested_unit_indices": [unit["unit_index"]],
        "messages": messages,
        "prompt": prompt,
        "output_format": "text",
        "response_format": None,
        "max_output_tokens": (
            512 if unit["kind"] == "table_cell" else max_output_tokens or spec.max_output_tokens
        ),
    }
