import pytest

from translation_lab.comet import (
    CONTENT_BUDGET,
    aggregate_scores,
    pack_pair,
    units_from_documents,
)


def words(text: str) -> int:
    return len(text.split()) if text else 0


def test_small_pair_stays_whole() -> None:
    units = pack_pair("one two", "un deux", words)

    assert len(units) == 1
    assert units[0].alignment == "whole"
    assert units[0].fits


def test_equal_sentence_counts_pack_by_sentence() -> None:
    source = " ".join(["one"] * 180) + ". " + " ".join(["two"] * 180) + ". "
    candidate = " ".join(["un"] * 180) + ". " + " ".join(["deux"] * 180) + ". "
    units = pack_pair(source, candidate, words)

    assert {unit.alignment for unit in units} == {"sentence"}
    assert "".join(unit.source for unit in units) == source
    assert "".join(unit.candidate for unit in units) == candidate
    assert all(unit.weight <= CONTENT_BUDGET for unit in units)


def test_mismatched_sentence_counts_fall_back_to_lossless_position() -> None:
    source = " ".join(["source"] * 700)
    candidate = " ".join(["target"] * 350) + ". Second sentence."
    units = pack_pair(source, candidate, words)

    assert {unit.alignment for unit in units} == {"positional"}
    assert "".join(unit.source for unit in units) == source
    assert "".join(unit.candidate for unit in units) == candidate
    assert all(unit.fits for unit in units)


def test_no_whitespace_atom_is_character_windowed() -> None:
    units = pack_pair("a" * 700, "b" * 700, len)

    assert len(units) == 3
    assert all(unit.fits for unit in units)
    assert "".join(unit.source for unit in units) == "a" * 700
    assert "".join(unit.candidate for unit in units) == "b" * 700


def test_document_recovery_removes_block_non_prose() -> None:
    source = "Intro.\n```python\nx = 1\n```\n$$x$$\nOutro."
    candidate = "Introduction.\n```python\nx = 1\n```\n$$x$$\nFin."
    units = units_from_documents(source, candidate, words)

    assert all("```" not in unit.source and "$$" not in unit.source for unit in units)
    assert {unit.provenance for unit in units} == {"recovered-document"}


def test_length_weighted_aggregation_and_minimum() -> None:
    short = pack_pair("one", "un", words)[0]
    long = pack_pair("one two three four", "un deux trois quatre", words)[0]
    mean, minimum = aggregate_scores([short, long], [0.0, 1.0])

    assert mean == pytest.approx(0.8)
    assert minimum == 0.0


def test_aggregation_rejects_score_count_mismatch() -> None:
    unit = pack_pair("one", "un", words)[0]

    with pytest.raises(ValueError, match="scores"):
        aggregate_scores([unit], [])
