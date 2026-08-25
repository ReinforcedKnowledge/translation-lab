from translation_lab.metrics import (
    audit_translation,
    language_identification,
    prose_words,
    script_drift,
    source_cjk_fraction,
    structure_report,
)


def test_french_translation_passes_calibrated_tripwire() -> None:
    source = " ".join(
        ["This is a sufficiently long English source sentence containing all necessary words."] * 8
    )
    output = " ".join(
        ["Le texte est dans la langue cible et il est traduit avec les mots nécessaires."] * 8
    )

    metrics = audit_translation(source, output, "French")

    assert metrics["target_present"] is True
    assert metrics["english_reverted"] is False
    assert metrics["looks_translated"] is True


def test_english_reversion_fails_even_with_a_few_target_words() -> None:
    source = " ".join(["The source contains a long reasoning trace."] * 8)
    output = "le texte " + " ".join(["the answer is still entirely in English"] * 8)

    metrics = audit_translation(source, output, "fr")

    assert metrics["english_reverted"] is True
    assert metrics["looks_translated"] is False


def test_short_prose_does_not_claim_target_presence() -> None:
    metrics = audit_translation("Hello", "Bonjour", "fr")

    assert metrics["target_present"] is None
    assert metrics["prose_tokens"] == 1


def test_code_and_math_do_not_dilute_language_density() -> None:
    text = "```python\nthe = 1\n``` $$the + is$$ Le texte est dans la langue cible."

    assert "the" not in prose_words(text)


def test_script_drift_is_target_aware() -> None:
    assert script_drift("text", "texte Ж", "fr") == 1
    assert script_drift("text", "κείμενο", "el") == 0
    assert script_drift("text", "κείμενο Ж", "el") == 1
    assert script_drift("text", "текст", "bg") is None


def test_language_identification_abstains_on_short_prose() -> None:
    assert language_identification("Bonjour !", "fr") == {
        "detected_language": None,
        "detected_language_confidence": None,
        "language_drift": False,
    }


def test_language_identification_uses_stripped_prose() -> None:
    text = "```python\nthe_answer = 1\n``` " + " ".join(
        ["Le résultat est présenté clairement dans cette phrase française."] * 6
    )

    metrics = language_identification(text, "fr")

    assert metrics["detected_language"] == "fr"
    assert metrics["language_drift"] is False


def test_structure_report_names_reasoning_and_wrapper_failures() -> None:
    metrics = structure_report("<think>Reasoning</think>Answer", "<think>Raisonner<text>")

    assert metrics["think_match"] is False
    assert metrics["think_count_ok"] is False
    assert metrics["tag_leak"] is True


def test_source_contamination_uses_the_share_of_real_letters() -> None:
    assert source_cjk_fraction("abc 中文") == 2 / 5
    assert audit_translation("abc 中文", "texte français", "fr")["source_contaminated"] is True
