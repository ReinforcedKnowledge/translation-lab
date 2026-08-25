import pytest

from translation_lab.interfaces import (
    MODEL_SPECS,
    generation_parameters,
    native_input,
    native_request,
)
from translation_lab.plans import TranslationUnit


class Renderer:
    def apply_chat_template(self, value: object, *, tokenize: bool, **options: object) -> object:
        return [1, 2] if tokenize else "rendered"

    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
        return [1, 2]


class MismatchedRenderer(Renderer):
    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
        return [2, 1]


def test_retained_model_interfaces_keep_their_native_prompts() -> None:
    assert native_input("milmmt", "fr", "Hello") == (
        "Translate this from English to French:\nEnglish: Hello\nFrench:"
    )
    hy = native_input("hy-mt2", "de", "Hello")
    assert "only output the translated result" in hy[0]["content"]
    tg = native_input("translategemma", "fi", "Hello")
    assert tg[0]["content"][0]["target_lang_code"] == "fi-FI"


def test_qwen_generation_disables_thinking_with_retained_stop_tokens() -> None:
    parameters = generation_parameters(MODEL_SPECS["qwen3.8"])

    assert parameters["temperature"] == 0.0
    assert parameters["seed"] == 42
    assert parameters["extra_body"]["stop_token_ids"] == [248046, 248044]


def test_offline_request_requires_template_and_completion_tokens_to_match() -> None:
    unit: TranslationUnit = {"unit_index": 0, "kind": "prose", "source_text": "Hello"}
    request = native_request(
        MODEL_SPECS["qwen3.8"], "fr", unit, request_index=0, renderer=Renderer()
    )

    assert request["messages"] is None
    assert request["prompt"] == "rendered"

    with pytest.raises(ValueError, match="rendered completion tokens differ"):
        native_request(
            MODEL_SPECS["qwen3.8"],
            "fr",
            unit,
            request_index=0,
            renderer=MismatchedRenderer(),
        )


def test_request_can_retain_a_curve_specific_output_ceiling() -> None:
    prose: TranslationUnit = {"unit_index": 0, "kind": "prose", "source_text": "Hello"}
    table: TranslationUnit = {"unit_index": 1, "kind": "table_cell", "source_text": "Header"}

    assert (
        native_request(
            MODEL_SPECS["gemma3"], "fr", prose, request_index=0, max_output_tokens=32768
        )["max_output_tokens"]
        == 32768
    )
    assert (
        native_request(
            MODEL_SPECS["gemma3"], "fr", table, request_index=1, max_output_tokens=32768
        )["max_output_tokens"]
        == 512
    )
