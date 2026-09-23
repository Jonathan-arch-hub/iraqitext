"""Unified, friendly API for everything Iraqi-text related.

``IraqiText`` wraps the translation engine, normalization, tokenization,
dialect detection and keyword extraction behind one small object::

    from iraqitext import IraqiText

    it = IraqiText()
    it.normalize("شــــلونك؟؟؟")      # -> "شلونك؟"
    it.tokenize("شلونك حبيبي؟")       # -> ["شلونك", "حبيبي"]
    it.detect("شلونك شخبارك؟")        # -> Detection(dialect='iraqi', ...)
    it.to_fusha("شلونك شخبارك؟")      # -> "كيف حالك ما أخبارك؟"
"""
from __future__ import annotations

from collections import Counter

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
from .stopwords import STOPWORDS
from .tokenizer import Token, tokenize, word_tokenize
from .translator import IraqiTranslator


class IraqiText:
    """One object for Iraqi text: normalize, tokenize, detect, extract and translate."""

    def __init__(self, translator: IraqiTranslator | None = None, **translator_kwargs):
        if translator is None:
            translator = IraqiTranslator(**translator_kwargs)
        self.translator = translator

    # ------------------------------------------------------------------ #
    # Normalization
    # ------------------------------------------------------------------ #
    def normalize(self, text: str, **kwargs) -> str:
        """Normalize Arabic/Iraqi text (tatweel, tashkeel, letters, punctuation)."""
        return normalize(text, **kwargs)

    def remove_tatweel(self, text: str) -> str:
        return remove_tatweel(text)

    def remove_tashkeel(self, text: str) -> str:
        return remove_tashkeel(text)

    def unify_letters(self, text: str, **kwargs) -> str:
        return unify_letters(text, **kwargs)

    # ------------------------------------------------------------------ #
    # Tokenization
    # ------------------------------------------------------------------ #
    def tokenize(self, text: str, *, with_positions: bool = False):
        """Tokenize text.

        By default returns a plain list of word strings
        (``["شلونك", "حبيبي"]``). With ``with_positions=True`` returns a
        list of :class:`~iraqitext.tokenizer.Token` objects (offsets & kinds).
        """
        if with_positions:
            return tokenize(text)
        return word_tokenize(text)

    def word_tokenize(self, text: str) -> list[str]:
        """Return word tokens of ``text``."""
        return word_tokenize(text)

    def words(self, text: str) -> list[str]:
        """Alias for :meth:`word_tokenize`."""
        return word_tokenize(text)

    # ------------------------------------------------------------------ #
    # Dialect detection
    # ------------------------------------------------------------------ #
    def detect(self, text: str, **kwargs) -> Detection:
        """Detect whether ``text`` is Iraqi, MSA, mixed or unknown."""
        return detect(text, **kwargs)

    def is_iraqi(self, text: str, **kwargs) -> bool:
        return is_iraqi(text, **kwargs)

    def is_arabic(self, text: str) -> bool:
        return is_arabic(text)

    # ------------------------------------------------------------------ #
    # Keywords
    # ------------------------------------------------------------------ #
    def keywords(
        self,
        text: str,
        *,
        top_n: int = 10,
        min_length: int = 2,
        stopwords: frozenset[str] | set[str] | None = None,
    ) -> list[str]:
        """Extract the most important words of ``text``.

        Words are normalized, tokenized, stripped of stopwords and ranked by
        frequency (most common first). Pass ``stopwords`` to replace the
        built-in Arabic list.
        """
        if stopwords is None:
            stopwords = STOPWORDS
        # Clean tatweel/tashkeel/punctuation/whitespace but keep letters
        # as-written so stopword matching stays literal and predictable.
        tokens = word_tokenize(
            normalize(
                text,
                strip_tatweel=True,
                strip_tashkeel=True,
                unify_alef=False,
                unify_hamza=False,
                punctuation=True,
                whitespace=True,
            )
        )
        counts = Counter(
            token for token in tokens
            if len(token) >= min_length
            and not token.isdecimal()
            and token not in stopwords
        )
        return [word for word, _ in counts.most_common(top_n)]

    # ------------------------------------------------------------------ #
    # Translation (delegates to IraqiTranslator)
    # ------------------------------------------------------------------ #
    def to_fusha(self, text: str) -> str:
        """Translate Iraqi dialect text to Modern Standard Arabic."""
        return self.translator.to_fusha(text)

    def to_iraqi(self, text: str) -> str:
        """Translate Modern Standard Arabic text to Iraqi dialect."""
        return self.translator.to_iraqi(text)