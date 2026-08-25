import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from typing import Any

from langid.langid import LanguageIdentifier
from langid.langid import model as LANGID_MODEL

TARGET_DENSITY_MIN = 0.10
ENGLISH_DENSITY_WITH_TARGET_MAX = 0.15
MIN_PROSE_TOKENS = 25
OUTPUT_SOURCE_RATIO_MIN = 0.60
OUTPUT_SOURCE_RATIO_MAX = 1.60

LANGUAGE_CODES = {
    "english": "en",
    "finnish": "fi",
    "french": "fr",
    "german": "de",
    "greek": "el",
    "polish": "pl",
    "spanish": "es",
}


def _words(value: str) -> frozenset[str]:
    return frozenset(value.split())


FUNCTION_WORDS = {
    "en": _words(
        "the of and to in is that it for with as was on are this be by an at which or from "
        "but not we you they have can if"
    ),
    "pl": _words(
        "i a w we na się nie że do to jest są z ze o jak ale czy dla po za od przez oraz więc "
        "lub gdy który która które których jako będzie będą był była było były ma mają może mieć "
        "trzeba jeśli jeżeli aby żeby dlatego ponieważ gdzie kiedy bez pod nad przed tylko także "
        "też tego tej tym te ta ten tak nas nam ich jego jej mnie mu my wy oni ona ono on co kto "
        "albo ani lecz jednak niż bardzo przy tych wszystko wszystkie wszystkich sobie już tam tu "
        "więcej można"
    ),
    "fr": _words(
        "le la les de des du un une et est sont que qui dans pour pas sur au aux ce cette ces il "
        "elle ils elles nous vous ne plus avec son sa ses leur leurs se on ou où mais donc car ni "
        "comme si quand parce très tout tous toute toutes mon ma mes ton notre votre être avoir "
        "fait aussi alors entre sans sous vers depuis pendant avant après encore déjà peut doit "
        "faut je tu me te lui eux cela ceci ça y en à"
    ),
    "de": _words(
        "der die das und ist ein eine einen einem einer den dem des zu von mit sich auf für nicht "
        "auch als dass wir sie es im an wird aber oder wenn weil dann doch nur noch schon sehr "
        "hier da dort kann muss soll haben hat sein war waren werden wurde durch über unter vor "
        "nach bei aus um zum zur kein keine sind dieser diese dieses man ich du ihr so wie in am "
        "im vom beim"
    ),
    "es": _words(
        "el la los las de un una unos unas y e o u ni que quien es son ser estar hay en por para "
        "con "
        "no se su sus lo al del como más pero este esta estos estas también entonces porque cuando "
        "si aunque sin sobre entre hasta desde hacia durante antes después puede debe tiene tener "
        "ha han había era fue eran donde yo tú él ella nosotros ellos esto eso muy a"
    ),
    "fi": _words(
        "ja on ei se että oli ovat hän joka mutta kun niin tämä mikä kuin myös vain jos siis koska "
        "sekä tai vai kanssa ilman mukaan jälkeen ennen aikana voi pitää täytyy sitä sen ne me te "
        "he minä sinä nämä kaikki jokainen tässä siellä nyt sitten vielä jo hyvin paljon enemmän "
        "kuten "
        "sillä eli jotta josta jossa jonka joita ovatko olisi tämän niiden"
    ),
    "el": _words(
        "και το η ο του της των στο στη στον στην με σε για από που δεν θα να είναι ήταν έχει "
        "έχουν ως αλλά όταν αν γιατί όπως επίσης μόνο αυτό αυτή αυτός τα οι ένα μια μπορεί πρέπει "
        "είχε όλα "
        "κάθε εδώ εκεί τώρα τότε ακόμα ήδη πολύ πιο ή ούτε καθώς δηλαδή στα στις τους την τη αυτά "
        "αυτές αυτών"
    ),
}

_NON_PROSE = re.compile(
    r"```.*?```|~~~.*?~~~|\$\$.*?\$\$|\$[^$\n]+?\$|\\\[.*?\\\]|\\\(.*?\\\)"
    r"|\\begin\{[^}]*\}.*?\\end\{[^}]*\}|\[asy\].*?\[/asy\]|</?[A-Za-z][^<>\n]*>",
    re.DOTALL,
)
_PLACEHOLDER = re.compile(r"⟦[A-Z]+_\d+⟧")
_BOXED = re.compile(r"\\boxed\{(?:[^{}]|\{[^{}]*\})*\}")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
_CJK = re.compile(r"[가-힣぀-ヿ一-鿿]")
_SCRIPTS = {
    "hangul": re.compile(r"[가-힣]"),
    "kana": re.compile(r"[぀-ヿ]"),
    "cjk": re.compile(r"[一-鿿]"),
    "cyrillic": re.compile(r"[Ѐ-ӿ]"),
    "greek": re.compile(r"[Ͱ-Ͽἀ-῿]"),
}
_FOREIGN_SCRIPTS = {
    "latin": ("hangul", "kana", "cjk", "cyrillic", "greek"),
    "greek": ("hangul", "kana", "cjk", "cyrillic"),
}
_TARGET_SCRIPT = {
    "pl": "latin",
    "fr": "latin",
    "de": "latin",
    "es": "latin",
    "fi": "latin",
    "el": "greek",
}
_LANGUAGE_IDENTIFIER = LanguageIdentifier.from_modelstring(LANGID_MODEL, norm_probs=True)
_FENCED_CODE = re.compile(r"(?ms)^(?:```|~~~)[^\n]*\n.*?^(?:```|~~~)[ \t]*$")
_DISPLAY_MATH_EXACT = re.compile(r"(?s)\$\$.*?\$\$|\\\[.*?\\\]")
_INLINE_CODE = re.compile(r"(?<!`)`[^`\n]+`(?!`)")
_INLINE_MATH = re.compile(r"(?s)(?<!\$)\$(?!\$)[^\n$]+\$(?!\$)|\\\(.*?\\\)")
_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+\*?")
_URL = re.compile(r"https?://[^\s<>\]\[(){}]+")
_XML_TAG = re.compile(r"<\s*(/?)\s*([A-Za-z][\w:.-]*)\b([^<>]*?)(/?)\s*>")
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?(?![\w.])")
_IDENTIFIER = re.compile(
    r"(?<![\w])(?:[A-Za-z_$][\w$]*_[\w$]+|[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+|[A-Za-z_$][\w$]*(?=\())"
)


def canonical_language(language: str) -> str:
    normalized = language.strip().lower()
    return LANGUAGE_CODES.get(normalized, normalized.split("-")[0].split("_")[0])


def prose_words(text: str) -> list[str]:
    return _WORD.findall(_NON_PROSE.sub(" ", text).lower())


def language_identification(text: str, language: str) -> dict[str, str | float | bool | None]:
    prose = _BOXED.sub(" ", _NON_PROSE.sub(" ", _PLACEHOLDER.sub(" ", text))).strip()
    if len(_LETTER.findall(prose)) < 30:
        detected = confidence = None
    else:
        detected, probability = _LANGUAGE_IDENTIFIER.classify(prose)
        confidence = round(float(probability), 3)
    return {
        "detected_language": detected,
        "detected_language_confidence": confidence,
        "language_drift": detected is not None and detected != canonical_language(language),
    }


def source_cjk_fraction(text: str) -> float:
    letters = _LETTER.findall(text)
    return len(_CJK.findall(text)) / len(letters) if letters else 0.0


def script_drift(source: str, output: str, language: str) -> int | None:
    code = canonical_language(language)
    scripts = _FOREIGN_SCRIPTS.get(_TARGET_SCRIPT.get(code, ""))
    if scripts is None:
        return None
    return sum(
        max(0, len(_SCRIPTS[name].findall(output)) - len(_SCRIPTS[name].findall(source)))
        for name in scripts
    )


def seam_run_togethers(text: str) -> int:
    count = 0
    for match in re.finditer(r"\S[.!?…]\S", text):
        value = match.group()
        if value[0].islower() and value[2].isupper():
            count += 1
    return count


def hard_structure_preserved(source: str, output: str) -> bool:
    source_tags = re.findall(r"</?(?:think|reasoning)>", source)
    output_tags = re.findall(r"</?(?:think|reasoning)>", output)
    return source_tags == output_tags


def structure_report(source: str, output: str) -> dict[str, bool | int]:
    source_open = source.count("<think>")
    source_close = source.count("</think>")
    output_open = output.count("<think>")
    output_close = output.count("</think>")
    source_has_think = source_open > 0
    return {
        "think_match": (source_open, source_close) == (output_open, output_close),
        "think_blocks": output_open,
        "think_count_ok": (
            (output_open, output_close) == (source_open, source_close)
            if source_has_think
            else output_open == 0
        ),
        "answer_outside_think": not output.rstrip().endswith("</think>"),
        "fence_parity": output.count("```") % 2 == 0,
        "tag_leak": "<text>" in output or "</text>" in output or "⟦" in output,
    }


def repeated_ngram_excess(text: str, n: int = 8) -> float:
    words = re.findall(r"\S+", text)
    if len(words) < n:
        return 0.0
    grams = [tuple(words[index : index + n]) for index in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def invisible_run_profile(text: str) -> dict[str, int]:
    maximum = current = total = format_total = 0
    for character in text:
        category = unicodedata.category(character)
        suspicious = category in {"Cf", "Zl", "Zp"} or (category == "Zs" and character != " ")
        format_total += category == "Cf"
        total += suspicious
        if suspicious:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return {"max_run": maximum, "total": total, "format_total": format_total}


def _ordered_equal(source_values: list[Any], output_values: list[Any]) -> dict[str, Any]:
    return {
        "exact_ordered": source_values == output_values,
        "missing": list((Counter(source_values) - Counter(output_values)).elements()),
        "added": list((Counter(output_values) - Counter(source_values)).elements()),
    }


def _table_signature(text: str) -> list[tuple[int, int, bool]]:
    result = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.count("|") < 2:
            continue
        cells = stripped.strip("|").split("|")
        separator = all(re.fullmatch(r"\s*:?-{3,}:?\s*", cell) for cell in cells)
        result.append((stripped.count("|"), len(cells), separator))
    return result


def _tags_balanced(tags: Iterable[tuple[str, str, bool]]) -> bool:
    stack: list[str] = []
    for closing, name, self_closing in tags:
        canonical = name.lower()
        if canonical in {"br", "hr", "img", "input", "meta", "link"} or self_closing:
            continue
        if closing:
            if not stack or stack.pop() != canonical:
                return False
        else:
            stack.append(canonical)
    return not stack


def preservation_report(source: str, output: str) -> dict[str, Any]:
    checks = []
    for pattern in (
        _FENCED_CODE,
        _DISPLAY_MATH_EXACT,
        _INLINE_CODE,
        _INLINE_MATH,
        _LATEX_COMMAND,
        _URL,
        _NUMBER,
        _IDENTIFIER,
    ):
        checks.append(_ordered_equal(pattern.findall(source), pattern.findall(output)))
    source_tags = [
        (match.group(1), match.group(2), bool(match.group(4)))
        for match in _XML_TAG.finditer(source)
    ]
    output_tags = [
        (match.group(1), match.group(2), bool(match.group(4)))
        for match in _XML_TAG.finditer(output)
    ]
    tag_check = _ordered_equal(source_tags, output_tags)
    tag_check["balance_preserved"] = _tags_balanced(source_tags) == _tags_balanced(output_tags)
    checks.append(tag_check)
    checks.append(_ordered_equal(_table_signature(source), _table_signature(output)))
    source_fences = len(re.findall(r"(?m)^(?:```|~~~)", source))
    output_fences = len(re.findall(r"(?m)^(?:```|~~~)", output))
    fence_check = {
        "exact_ordered": source_fences == output_fences,
        "output_balanced": output_fences % 2 == 0,
    }
    checks.append(fence_check)
    strict = all(check["exact_ordered"] for check in checks)
    strict = strict and tag_check["balance_preserved"] and fence_check["output_balanced"]
    return {"strict_pass": strict, "checks": checks}


def audit_translation(source: str, output: str, language: str) -> dict[str, Any]:
    code = canonical_language(language)
    words = prose_words(output)
    denominator = max(1, len(words))
    target_words = FUNCTION_WORDS.get(code)
    shared = FUNCTION_WORDS["en"] & target_words if target_words else frozenset()
    english_hits = sum(word in FUNCTION_WORDS["en"] and word not in shared for word in words)
    english_density = english_hits / denominator
    ratio = len(output.strip()) / max(1, len(source))

    if target_words is None:
        target_density: float | None = None
        target_present: bool | None = None
        english_reverted: bool | None = None
    else:
        target_density = sum(word in target_words for word in words) / denominator
        if len(words) < MIN_PROSE_TOKENS:
            target_present = None
            english_reverted = english_density >= TARGET_DENSITY_MIN
        else:
            target_present = target_density >= TARGET_DENSITY_MIN
            english_reverted = english_density >= ENGLISH_DENSITY_WITH_TARGET_MAX or (
                not target_present and english_density >= TARGET_DENSITY_MIN
            )

    emitted_code_fence = ("```" in output and "```" not in source) or (
        "~~~" in output and "~~~" not in source
    )
    looks_translated = (
        OUTPUT_SOURCE_RATIO_MIN <= ratio <= OUTPUT_SOURCE_RATIO_MAX
        and target_present is not False
        and english_reverted is not True
        and not emitted_code_fence
    )
    drift = script_drift(source, output, code)
    preservation = preservation_report(source, output)
    structure = structure_report(source, output)
    language_id = language_identification(output, code)
    cjk_fraction = source_cjk_fraction(source)
    hard_failure = (
        not preservation["strict_pass"]
        or not structure["think_match"]
        or not structure["think_count_ok"]
        or not structure["fence_parity"]
        or structure["tag_leak"]
    )
    return {
        "output_source_ratio": round(ratio, 3),
        "prose_tokens": len(words),
        "target_density": round(target_density, 3) if target_density is not None else None,
        "target_present": target_present,
        "english_density": round(english_density, 3),
        "english_reverted": english_reverted,
        "emitted_code_fence": emitted_code_fence,
        "script_drift": drift,
        "seam_run_togethers": seam_run_togethers(output),
        "structure_preserved": hard_structure_preserved(source, output),
        "preservation": preservation,
        **structure,
        **language_id,
        "document_hard_failure": hard_failure,
        "source_cjk_fraction": round(cjk_fraction, 6),
        "source_contaminated": cjk_fraction >= 0.05,
        "looks_translated": looks_translated,
        "nonempty": bool(output.strip()),
    }


def audit_request(
    source: str, output: str, language: str, finish_reason: str | None
) -> dict[str, Any]:
    metrics = audit_translation(source, output, language)
    ratio = len(output) / max(1, len(source))
    repetition = repeated_ngram_excess(output)
    source_repetition = repeated_ngram_excess(source)
    invisible = invisible_run_profile(output)
    source_invisible = invisible_run_profile(source)
    preservation = preservation_report(source, output)
    flags = {
        "finish_reason_length": finish_reason == "length",
        "empty_output": not output.strip(),
        "english_reversion": bool(metrics["english_reverted"]),
        "new_code_fence": bool(metrics["emitted_code_fence"]),
        "script_drift": metrics["script_drift"],
        "runaway_expansion": ratio > 2.2,
        "undertranslation": len(source) >= 80 and ratio < 0.45,
        "repetition": repetition > max(0.35, source_repetition + 0.05),
        "invisible_run": invisible["max_run"] >= 16,
        "format_character_run": invisible["format_total"] > source_invisible["format_total"],
        "literal_preservation_failure": not preservation["strict_pass"],
    }
    severe = any(
        flags[name]
        for name in (
            "finish_reason_length",
            "empty_output",
            "english_reversion",
            "new_code_fence",
            "runaway_expansion",
            "undertranslation",
            "repetition",
            "invisible_run",
            "format_character_run",
        )
    )
    return {
        **metrics,
        **flags,
        "repeated_8gram_excess": repetition,
        "maximum_invisible_run": invisible["max_run"],
        "preservation": preservation,
        "severe_composite": severe,
    }
