"""Dictionary-based translation between Iraqi Arabic and MSA.

A lightweight, pattern-based word/phrase matcher with **no dependencies**:

* entries are tried longest-first so ``شكو ماكو`` wins over ``شكو``;
* the default ``(?<!\\w) ... (?!\\w)`` boundaries mean a dictionary entry
  matches even when directly attached to punctuation: ``شلونك؟``,
  ``(شلونك)``, ``شلونك،``. Pass ``strict_spaces=True`` to restore the old
  whitespace-only ``(?<!\\S) ... (?!\\S)`` behaviour;
* ``clitics=True`` (opt-in, experimental) additionally translates words
  attached to the Arabic proclitics ``و / ف / ب / ك / ل``, e.g.
  ``وشخبارك`` -> ``وما أخبارك``.

The public API is unchanged::

    from iraqitext import IraqiTranslator
    translator = IraqiTranslator()
    translator.to_fusha("شلونك")    # كيف حالك
    translator.to_iraqi("كيف حالك")  # إشلونك
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .rules import flexible_pattern
from .tokenizer import ARABIC_LETTERS

BASE_DIR = Path(__file__).parent

# Arabic proclitics handled by the (opt-in) ``clitics`` mode.
CLITIC_LETTERS = "وفبكل"


def _load_dictionary():
    with open(BASE_DIR / "dictionary.json", "r", encoding="utf-8") as f:
        dictionary = json.load(f)
    to_fusha_pairs = sorted(
        dictionary.items(), key=lambda pair: len(pair[0]), reverse=True
    )
    to_iraqi_pairs = sorted(
        ((v, k) for k, v in dictionary.items()),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    return to_fusha_pairs, to_iraqi_pairs


_FUSHA_PAIRS, _IRAQI_PAIRS = _load_dictionary()


def _boundaries(boundary: str) -> tuple[str, str]:
    if boundary == "word":
        return r"(?<!\w)", r"(?!\w)"
    if boundary == "space":
        return r"(?<!\S)", r"(?!\S)"
    raise ValueError(f"unknown boundary: {boundary!r} (use 'word' or 'space')")


def _compile(pairs, boundary: str = "word"):
    pre, post = _boundaries(boundary)
    return [
        (re.compile(pre + flexible_pattern(pattern) + post), _plain_replacer(replacement))
        for pattern, replacement in pairs
    ]


def _plain_replacer(replacement: str):
    return lambda match: replacement


def _apply(text: str, rules) -> str:
    for regex, replacer in rules:
        text = regex.sub(replacer, text)
    return text


_CLITIC_WORD_RE = re.compile(
    r"(?<!\w)([" + CLITIC_LETTERS + r"]+[" + ARABIC_LETTERS + r"]+)(?!\w)"
)


def _clitic_pass(text: str, mapping: dict[str, str]) -> str:
    """Translate the base word of proclitic-attached tokens (opt-in).

    Only tokens that are *not* themselves dictionary entries are touched:
    the longest matching base word is split off and translated while the
    clitic letters are re-attached in front (``وشخبارك`` -> ``وما أخبارك``).
    """

    def _replacer(match: re.Match) -> str:
        token = match.group(0)
        if token in mapping:
            return token  # leave it to the normal engine
        for i in range(1, len(token)):  # longest base first
            clitic, base = token[:i], token[i:]
            if base in mapping:
                return clitic + mapping[base]
        return token

    return _CLITIC_WORD_RE.sub(_replacer, text)


# Module-level defaults kept for backward compatibility; instances compile
# their own rules so constructor options (strict_spaces / clitics) work.
_TO_FUSHA_RULES = _compile(_FUSHA_PAIRS, "word")
_TO_IRAQI_RULES = _compile(_IRAQI_PAIRS, "word")


class IraqiTranslator:
    """Translate between Iraqi dialect and Modern Standard Arabic.

    Parameters
    ----------
    strict_spaces:
        Use the old whitespace-only word boundaries (``(?<!\\S)`` /
        ``(?!\\S)``). Default ``False`` uses Unicode-aware word boundaries
        so translations also work next to punctuation.
    clitics:
        Also translate words attached to the proclitics ``و ف ب ك ل``
        (experimental; e.g. ``وشخبارك`` -> ``وما أخبارك``).
    """

    def __init__(self, *, strict_spaces: bool = False, clitics: bool = False):
        self.boundary = "space" if strict_spaces else "word"
        self.clitics = bool(clitics)
        self._fusha_map = dict(_FUSHA_PAIRS)
        self._iraqi_map = dict(_IRAQI_PAIRS)
        self._to_fusha_rules = _compile(_FUSHA_PAIRS, self.boundary)
        self._to_iraqi_rules = _compile(_IRAQI_PAIRS, self.boundary)

    def to_fusha(self, text: str) -> str:
        """Translate Iraqi dialect text to Modern Standard Arabic."""
        if self.clitics:
            text = _clitic_pass(text, self._fusha_map)
        return _apply(text, self._to_fusha_rules)

    def to_iraqi(self, text: str) -> str:
        """Translate Modern Standard Arabic text to Iraqi dialect."""
        if self.clitics:
            text = _clitic_pass(text, self._iraqi_map)
        return _apply(text, self._to_iraqi_rules)