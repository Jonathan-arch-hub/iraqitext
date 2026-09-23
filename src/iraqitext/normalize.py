"""Normalization utilities for Arabic / Iraqi text.

Pure functions with no external dependencies — only ``re`` and
``str.translate`` — so the library stays fast and CPU-only friendly.

The main entry point is :func:`normalize`. Each step is also exposed
(``remove_tatweel``, ``unify_letters``, ...) so you can build your own
pipeline if the defaults are not what you need.
"""
from __future__ import annotations

import re

_TATWEEL = "\u0640"

# Harakat/tashkeel + Quranic marks + Arabic-Extended-A marks.
_TASHKEEL_RE = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED\u08E0-\u08FF]")

# Control / invisible / zero-width characters, soft hyphen, BOM, ALM...
_CONTROL_RE = re.compile(
    r"[\x00-\x1F\x7F-\x9F\u00AD\u061C\u200B-\u200F\u2028-\u202E"
    r"\u2060-\u206F\uFEFF]"
)

_ARABIC_LETTERS = r"\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF"
_REPEATED_LETTERS_RE = re.compile(r"([" + _ARABIC_LETTERS + r"])\1{2,}")

# --- letter unification tables ------------------------------------------
_UNIFY_ALEF = str.maketrans("أإآٱ", "اااا")
_UNIFY_HAMZA_SEAT = str.maketrans({"ؤ": "و", "ئ": "ي"})
_UNIFY_PERSIAN = str.maketrans(
    {
        "ی": "ي",  # U+06CC Persian yeh
        "ک": "ك",  # U+06A9 Persian keheh
        "ے": "ي",  # U+06D2 Urdu yeh
        "ۓ": "ي",  # U+06D3 Urdu yeh (hamza above)
        # Persian-Indic digits -> Arabic-Indic digits
        "۰": "٠", "۱": "١", "۲": "٢", "۳": "٣", "۴": "٤",
        "۵": "٥", "۶": "٦", "۷": "٧", "۸": "٨", "۹": "٩",
    }
)
_UNIFY_TEH_MARBUTA = str.maketrans({"ة": "ه", "ۃ": "ه"})
_UNIFY_ALEF_MAQSURA = str.maketrans({"ى": "ي"})

# --- punctuation ---------------------------------------------------------
_PUNCT_MAP = str.maketrans({",": "،", ";": "؛", "?": "؟"})
_REPEATED_PUNCT_RE = re.compile(r"([،؟؛!])\1+")
_ELLIPSIS_RE = re.compile(r"\.{3,}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([،؟؛!….])")
_WHITESPACE_RE = re.compile(r"\s+")


def remove_tatweel(text: str) -> str:
    """Remove kashida / tatweel characters (ـ)."""
    return text.replace(_TATWEEL, "")


def remove_tashkeel(text: str) -> str:
    """Remove harakat and other diacritic marks."""
    return _TASHKEEL_RE.sub("", text)


def remove_control_chars(text: str) -> str:
    """Remove control, zero-width and other invisible characters."""
    return _CONTROL_RE.sub("", text)


def collapse_repeated_letters(text: str, max_run: int = 1) -> str:
    """Collapse runs of 3+ identical Arabic letters down to ``max_run``.

    Dialect writing often stretches letters for emphasis
    (``ششششلونك``, ``حبييييبي``); this brings them back to a single
    (or ``max_run``) letter. Valid doubled letters (e.g. shadda spellings
    like ``شدّ``) with a run-length of exactly 2 are preserved.
    """
    if max_run < 1:
        return text

    def _rep(match: re.Match) -> str:
        return match.group(1) * max_run

    return _REPEATED_LETTERS_RE.sub(_rep, text)


def unify_letters(
    text: str,
    *,
    alef: bool = True,
    hamza: bool = True,
    persian: bool = True,
    teh_marbuta: bool = False,
    alef_maqsura: bool = False,
) -> str:
    """Unify spelling variants into a single canonical letter.

    * ``alef``: ``أ إ آ ٱ`` -> ``ا``
    * ``hamza``: ``ؤ`` -> ``و``, ``ئ`` -> ``ي`` (standalone ``ء`` is kept)
    * ``persian``: ``ی ک ے ۓ`` -> ``ي ك`` and Persian digits -> Arabic-Indic
    * ``teh_marbuta``: ``ة ۃ`` -> ``ه`` (off by default: it can change meaning)
    * ``alef_maqsura``: ``ى`` -> ``ي`` (off by default)
    """
    if alef:
        text = text.translate(_UNIFY_ALEF)
    if hamza:
        text = text.translate(_UNIFY_HAMZA_SEAT)
    if persian:
        text = text.translate(_UNIFY_PERSIAN)
    if teh_marbuta:
        text = text.translate(_UNIFY_TEH_MARBUTA)
    if alef_maqsura:
        text = text.translate(_UNIFY_ALEF_MAQSURA)
    return text


def normalize_punctuation(
    text: str,
    *,
    unify: bool = True,
    collapse: bool = True,
    spacing: bool = True,
) -> str:
    """Normalize Arabic punctuation.

    * ``unify``: ASCII ``,`` ``;`` ``?`` -> ``،`` ``؛`` ``؟``
    * ``collapse``: ``؟؟؟`` -> ``؟``, ``!!!`` -> ``!``, ``...`` -> ``…``
    * ``spacing``: remove spaces *before* punctuation marks
    """
    if unify:
        text = text.translate(_PUNCT_MAP)
    if collapse:
        text = _REPEATED_PUNCT_RE.sub(r"\1", text)
        text = _ELLIPSIS_RE.sub("…", text)
    if spacing:
        text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse every run of whitespace to a single space and trim."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize(
    text: str,
    *,
    strip_tatweel: bool = True,
    strip_tashkeel: bool = True,
    strip_controls: bool = True,
    unify_alef: bool = True,
    unify_hamza: bool = True,
    unify_persian: bool = True,
    unify_teh_marbuta: bool = False,
    unify_alef_maqsura: bool = False,
    collapse_repeats: bool = False,
    punctuation: bool = True,
    whitespace: bool = True,
    strip: bool = True,
) -> str:
    """Normalize an Arabic / Iraqi string in one call.

    Defaults are deliberately conservative: tatweel, tashkeel and control
    characters are removed, alef/hamza/Persian-variant letters are unified,
    and punctuation + whitespace are tidied up. ``ة -> ه`` and ``ى -> ي``
    are opt-in because they can change meaning in MSA.

    Examples:
        >>> normalize("شــــلونك؟؟؟")
        'شلونك؟'
        >>> normalize("كَيْفَ حَالُكْ")
        'كيف حالك'
    """
    if strip_controls:
        text = remove_control_chars(text)
    if strip_tatweel:
        text = remove_tatweel(text)
    if strip_tashkeel:
        text = remove_tashkeel(text)
    text = unify_letters(
        text,
        alef=unify_alef,
        hamza=unify_hamza,
        persian=unify_persian,
        teh_marbuta=unify_teh_marbuta,
        alef_maqsura=unify_alef_maqsura,
    )
    if collapse_repeats:
        text = collapse_repeated_letters(text, max_run=1)
    if punctuation:
        text = normalize_punctuation(text)
    if whitespace:
        text = normalize_whitespace(text)
    elif strip:
        text = text.strip()
    return text


def light_normalize(text: str) -> str:
    """Minimal, safe normalization used before matching / translation.

    Removes tatweel, tashkeel and control characters and cleans whitespace
    and punctuation, but leaves the letters themselves untouched — the
    matcher in ``rules.py`` already tolerates alif/teh variant spellings,
    so this never has to guess how to rewrite a word.
    """
    return normalize(
        text,
        unify_alef=False,
        unify_hamza=False,
        unify_persian=False,
        collapse_repeats=False,
    )