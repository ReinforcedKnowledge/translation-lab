import pytest

from translation_lab.plans import TranslationPlan, p0_plan, raw_plan, reconstruct, split_raw_text


def identity(plan: TranslationPlan) -> dict[int, str]:
    return {unit["unit_index"]: unit["source_text"] for unit in plan["translation_units"]}


def test_p0_plan_keeps_literals_and_reconstructs_its_normalized_source() -> None:
    source = (
        "Opening.\n\n<think>Reasoning.\n\n```python\nprint('x')\n```\n\n$$x+y$$"
        "</think>\n\n| Name | Value |\n| --- | --- |\n| label | 2 |\n"
    )
    plan = p0_plan("s1", source, chunk_target_chars=40)
    units = plan["translation_units"]

    assert {unit["kind"] for unit in units} == {"prose", "table_cell"}
    assert all(
        "```" not in unit["source_text"] and "$$" not in unit["source_text"] for unit in units
    )
    assert reconstruct(plan, identity(plan)) == (
        "Opening.\n\n<think>\nReasoning.\n\n```python\nprint('x')\n```\n\n$$x+y$$\n</think>"
        "\n\n| Name | Value |\n| --- | --- |\n| label | 2 |"
    )


def test_raw_plan_preserves_all_source_characters_and_separators() -> None:
    source = "First sentence.\n\nSecond line with code `x`.\nThird line.  "
    units, separators = split_raw_text(source, 22)
    plan = raw_plan("s1", source, chunk_target_chars=22)

    assert (
        "".join(
            unit + (separators[index] if index < len(separators) else "")
            for index, unit in enumerate(units)
        )
        == source
    )
    assert reconstruct(plan, identity(plan)) == source


def test_reconstruction_refuses_missing_unit_output() -> None:
    plan = raw_plan("s1", "one two three", chunk_target_chars=5)

    with pytest.raises(KeyError, match="missing translation unit"):
        reconstruct(plan, {0: "un"})
