"""Lightweight tokenizer for Arabic / Iraqi text.

No external dependencies: a single Unicode-aware regex powers everything,
so it stays fast on CPU-only machines. Punctuation and whitespace are
emitted as their own tokens (or skipped), and every token carries its
character offsets for further NLP work.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Arabic *letters* only (no punctuation like ؟ ، ؛ which live inside
# U+0600-U+06FF), including Arabic-Indic/Persian digits handled separately.
ARABIC_LETTERS = (
    r"\u0621-\u063A\u0641-\u064A\u066E\u066F"
    r"\u0671-\u06D3\u06D5\u06EE\u06EF\u06FA-\u06FF"
    r"\u0750-\u077F\u08A0-\u08FF"
)

_TOKEN_RE = re.compile(
    r"(?P<word>["
    + ARABIC_LETTERS
    + r"]+|[A-Za-z]+(?:['’][A-Za-z]+)?)"
    r"|(?P<number>\d+(?:[.,]\d+)*)"
    r"|(?P<space>\s+)"
    r"|(?P<punct>[^\w\s]+)"
    r"|(?P<other>.)"
)


@dataclass(frozen=True)
class Token:
    """A single lexical token with its position in the source string."""

    text: str
    start: int
    end: int
    kind: str  # one of: word | number | space | punct | other

    @property
    def is_word(self) -> bool:
        """True for real (non-numeric) word tokens."""
        return self.kind == "word"

    def __str__(self) -> str:
        return self.text


def tokenize(text: str, *, keep_spaces: bool = False) -> list[Token]:
    """Split ``text`` into :class:`Token` objects with offsets.

    ``kind`` is one of ``word``, ``number``, ``space``, ``punct`` or
    ``other``. Arabic-Indic digit-only runs are classified as ``number``.
    Whitespace tokens are dropped unless ``keep_spaces=True``.
    """
    tokens: list[Token] = []
    for match in _TOKEN_RE.finditer(text):
        kind = match.lastgroup
        value = match.group()
        if kind == "word" and value.isdecimal():
            kind = "number"
        if kind == "space" and not keep_spaces:
            continue
        tokens.append(Token(value, match.start(), match.end(), kind))
    return tokens


def word_tokenize(text: str) -> list[str]:
    """Return only the word tokens of ``text`` (no punctuation/numbers)."""
    return [t.text for t in tokenize(text) if t.kind == "word"]