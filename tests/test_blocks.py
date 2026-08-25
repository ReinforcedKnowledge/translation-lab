from translation_lab.blocks import (
    parse_blocks,
    reassemble_prose,
    split_control_tags,
    split_prose,
    split_table,
    translatable_cell,
)


def test_parser_is_lossless_and_typed() -> None:
    source = (
        "Intro.\n\n"
        "```python\nprint('keep')\n```\n\n"
        "$$E = mc^2$$\n\n"
        "| Name | Value |\n| --- | --- |\n| apples | 15 |\n\n"
        "Outro."
    )
    blocks = parse_blocks(source)

    assert "".join(value for _, value in blocks) == source
    assert {kind for kind, _ in blocks} == {"prose", "code", "math", "table"}


def test_fence_scanner_handles_tildes_and_unclosed_fences() -> None:
    closed = "Before\n~~~~python\nx = 1\n~~~~~\nAfter"
    unclosed = "Before\n```python\nx = 1\nAfter"

    assert any(kind == "code" for kind, _ in parse_blocks(closed))
    assert parse_blocks(unclosed) == [("prose", unclosed)]


def test_numeric_pipe_matrix_is_not_treated_as_a_table() -> None:
    matrix = "| 1 | 2 |\n| 3 | 4 |\n"
    table = "| City | Score |\n| Paris | 2 |\n"

    assert parse_blocks(matrix) == [("prose", matrix)]
    assert parse_blocks(table) == [("table", table)]


def test_table_split_is_lossless_and_filters_data_cells() -> None:
    table = "| Label | Value |\n| --- | --- |\n| Nominations received | 15 |\n"
    parts = split_table(table)

    assert "".join(value for _, value in parts) == table
    assert translatable_cell(" Nominations received ")
    assert not translatable_cell(" 15 ")
    assert not translatable_cell(" _Z1hi ")


def test_prose_split_is_lossless_for_paragraphs_and_long_runs() -> None:
    source = "  " + "First sentence. " * 30 + "\n\n" + "Second paragraph. " * 20 + "\n"
    leading, chunks, separators, trailing = split_prose(source, target_chars=80)

    assert reassemble_prose(leading, chunks, separators, trailing) == source
    assert len(chunks) > 2


def test_control_tags_remain_explicit_across_multiple_reasoning_sections() -> None:
    source = "A<think>B</think>C<reasoning>D</reasoning>E"
    parts = split_control_tags(source)

    assert "".join(value for _, value in parts) == source
    assert [value for kind, value in parts if kind == "tag"] == [
        "<think>",
        "</think>",
        "<reasoning>",
        "</reasoning>",
    ]
