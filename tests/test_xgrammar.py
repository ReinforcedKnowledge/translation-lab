import pytest

from translation_lab.xgrammar import ORIGINAL_BLOCK, REPAIRED_BLOCK, repair_source


def test_stop_token_repair_adds_generation_config_ids_to_xgrammar() -> None:
    source = "prefix\n" + ORIGINAL_BLOCK + "suffix\n"
    repaired = repair_source(source)

    assert ORIGINAL_BLOCK not in repaired
    assert REPAIRED_BLOCK in repaired
    assert "stop_token_ids=stop_token_ids" in repaired


def test_stop_token_repair_refuses_an_unknown_source() -> None:
    with pytest.raises(ValueError, match="expected one"):
        repair_source("unrelated source")
