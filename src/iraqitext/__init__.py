"""iraqitext: lightweight NLP for the Iraqi Arabic dialect.

Zero mandatory dependencies, works on CPU-only machines. Combines a
dictionary-based Iraqi <-> MSA translation engine with normalization,
tokenization, dialect detection and keyword extraction.

Typical use::

    from iraqitext import IraqiTranslator, IraqiText

    translator = IraqiTranslator()
    translator.to_fusha("شلونك")   # "كيف حالك"

    it = IraqiText()
    it.normalize("شــــلونك؟؟؟")   # "شلونك؟"
    it.detect("شلونك شخبارك؟")     # Detection(dialect='iraqi', ...)
"""
from .detector import Detection, detect, is_arabic, is_iraqi
from .normalize import (
    collapse_repeated_letters,
    light_normalize,
    normalize,
    normalize_punctuation,
    normalize_whitespace,
    remove_control_chars,
    remove_tashkeel,
    remove_tatweel,
    unify_letters,
)
from .text import IraqiText
from .tokenizer import Token, tokenize, word_tokenize
from .translator import CLITIC_LETTERS, IraqiTranslator

__version__ = "0.2.0"

__all__ = [
    "IraqiTranslator",
    "IraqiText",
    # normalization
    "normalize",
    "light_normalize",
    "remove_tatweel",
    "remove_tashkeel",
    "remove_control_chars",
    "unify_letters",
    "normalize_punctuation",
    "normalize_whitespace",
    "collapse_repeated_letters",
    # tokenization
    "Token",
    "tokenize",
    "word_tokenize",
    # detection
    "Detection",
    "detect",
    "is_iraqi",
    "is_arabic",
    # translator helpers
    "CLITIC_LETTERS",
]