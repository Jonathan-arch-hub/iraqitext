"""Regex helpers for tolerant Arabic matching.

The translation engine matches dictionary entries with *flexible* patterns:
alef variants (``ا إ أ آ ٱ``) and heh-variants (``ه ة``) inside a word are
treated as equivalent, so a single entry covers common spelling variants
without bloating ``dictionary.json``.
"""
import re

_ALIF_VARIANTS = "اإأآٱ"
_HA_VARIANTS = "هة"


def _char_class(ch):
    if ch in _ALIF_VARIANTS:
        return "[" + _ALIF_VARIANTS + "]"
    if ch in _HA_VARIANTS:
        return "[" + _HA_VARIANTS + "]"
    return re.escape(ch)


def flexible_pattern(phrase):
    """Build a regex fragment matching ``phrase`` with tolerant letters."""
    return "".join(_char_class(ch) for ch in phrase)