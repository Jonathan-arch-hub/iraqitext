"""Iraqi dialect detection.

Heuristic, dictionary-assisted detection that runs on the CPU with no ML
models. Instead of counting "words that happen to be in the dictionary",
it weights *evidence* per token across three tiers:

* ``strong_iraqi``  — unambiguously Iraqi items:
  characteristic words (``شلون``, ``شنو``, ``هسه``, ...) and any word
  using the Iraqi-only letters ``چ`` / ``گ`` / ``پ``.
* ``medium_iraqi``  — tokens that appear as keys of ``dictionary.json``
  but never as MSA values: they are Iraqi-only vocabulary by the data
  itself (≈1000 words, no extra dictionary needed).
* ``strong_msa``    — MSA function words that have a direct Iraqi
  counterpart (``ماذا``/``شنو``, ``كيف``/``شلون``, ``إلى``/``على``,
  ``سوف``/``راح``, ...).
* ``medium_msa``    — MSA verb stems Iraqis replace with other roots
  (``ذهب``, ``أتى``, ``فعل`` ...) and softer MSA particles.
* ``neutral``       — shared/common words (``تريد``, ``اليوم``,
  ``السوق``, words that appear on both sides of the dictionary) that do
  **not** push the verdict either way.

The final ``score`` is the share of Iraqi evidence vs total evidence
(``0`` = pure MSA, ``1`` = pure Iraqi, ``0.5`` = perfectly balanced), and
the ``dialect`` label is a transparent projection of that axis:

* only Iraqi evidence            -> ``iraqi``
* only MSA evidence              -> ``msa``
* both, ratio very one-sided     -> that side (default > 0.75 / < 0.25)
* both, balanced                 -> ``mixed``

``details`` keeps the old raw counters (``iraqi_hits``,
``strong_markers``, ``marker_letters``) for backwards compatibility plus
an ``evidence`` map that says, per token, which tier it landed in — so
every decision is inspectable.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .normalize import normalize
from .tokenizer import ARABIC_LETTERS, word_tokenize

BASE_DIR = Path(__file__).parent

ARABIC_RE = re.compile(r"[" + ARABIC_LETTERS + r"]")

# Letters that are (almost) never used in MSA but are common in Iraqi.
IRAQI_LETTERS = "چگپ"

# --------------------------------------------------------------------------- #
# Evidence weights
# --------------------------------------------------------------------------- #
#: Weight of a single strong (unambiguous) piece of evidence.
STRONG_WEIGHT = 3.0
#: Weight of a medium piece of evidence (dictionary-only Iraqi, MSA verbs).
MEDIUM_WEIGHT = 2.0
#: Weight of a light piece of evidence (softer MSA particles).
LIGHT_WEIGHT = 1.5

# Above this ratio of Iraqi(evidence) the text is considered clearly Iraqi.
IRAQI_THRESHOLD = 0.75
# Below this ratio the text is considered clearly MSA.
MIXED_THRESHOLD = 0.25

# --------------------------------------------------------------------------- #
# Curated evidence lists (kept small; the dictionary supplies the rest).
# The words are written in natural spelling and canonicalized at import time,
# so spelling variants (أ/ا/إ, ة/ه, ی/ي ...) are folded automatically.
# --------------------------------------------------------------------------- #

# Unambiguously Iraqi words/signals. Includes the classic markers
# (شلون / شنو / شكد / هسه / ليش / ويا / تسوي / چان / تره / وين) plus a few
# very frequent everyday items that the dictionary does not always cover.
STRONG_IRAQI_WORDS = frozenset(
    """
    شلون شلونك اشلون اشلونك شخبار شخبارك شنو شكو شكد
    هسه هسة ليش ويا وياك وياي تسوي وين وينك
    چان چانت جيت اروح رحت راحوا اكو ماكو تره ترا
    مو هيه زين خوش كلش اگول گال كاعد هذول هذي
    """.split()
)

# MSA function words that have a clear, different Iraqi counterpart.
# They act as the real counterweight to the Iraqi evidence.
STRONG_MSA_WORDS = frozenset(
    """
    ماذا كيف اين لماذا ان الى سوف هل كم لم لن
    كان كانت تكون يكون ثم اذا لقد ايضا ربما يجب ينبغي
    عندما حيث بينما لعل هولاء
    """.split()
)

# MSA verb stems that Iraqi replaces with a different root
# (ذهب -> راح/گال, أتى -> جا, فعل -> سوى, خبر -> گال/خبر). Exact-string
# match, so they can never fire on unrelated words.
MSA_VERB_STEMS = frozenset(
    """
    ذهب يذهب تذهب نذهب اذهب ذهبت ذهبوا ذهبت ذهبنا
    اتى اتت ياتي تاتي تات يات اتوا اتينا اتيت
    يفعل تفعل افعل نفعل يفعلون تفعلون فعلت فعلوا
    يخبر تخبر اخبر اخبرني اخبرت اخبروني
    """.split()
)

# Softer MSA indicators — they count, but weigh less than the strong ones.
LIGHT_MSA_WORDS = frozenset(
    """
    الذي التي الذين ذلك تلك هناك هنا يوجد يمكن كذلك اما
    لو لولا فقط غير سوى قد لكن
    """.split()
)


def _canon(word: str) -> str:
    """Canonical form used for matching dictionary keys against text."""
    return normalize(
        word,
        strip_tatweel=True,
        strip_tashkeel=True,
        strip_controls=True,
        unify_alef=True,
        unify_hamza=True,
        unify_persian=True,
        unify_teh_marbuta=False,
        unify_alef_maqsura=False,
        collapse_repeats=False,
        punctuation=False,
        whitespace=True,
    )


def _canonical_set(words: frozenset[str]) -> frozenset[str]:
    return frozenset(w for word in words if (w := _canon(word)))


STRONG_IRAQI = _canonical_set(STRONG_IRAQI_WORDS)
STRONG_MSA = _canonical_set(STRONG_MSA_WORDS)
MSA_VERBS = _canonical_set(MSA_VERB_STEMS)
LIGHT_MSA = _canonical_set(LIGHT_MSA_WORDS)

del STRONG_IRAQI_WORDS, STRONG_MSA_WORDS, MSA_VERB_STEMS, LIGHT_MSA_WORDS


# --------------------------------------------------------------------------- #
# Dictionary-derived evidence
# --------------------------------------------------------------------------- #
def _load_vocab() -> tuple[set[str], set[str], set[str]]:
    """Split the dictionary into key tokens, value tokens and the overlap.

    Returns ``(keys, values, keys - values)``. A token in ``keys - values``
    is Iraqi-only *by the dictionary's own data*: it is a dialectal source
    entry that never appears as an MSA translation, so it is clean medium
    Iraqi evidence. Tokens on both sides are shared/neutral.
    """
    with open(BASE_DIR / "dictionary.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    keys: set[str] = set()
    values: set[str] = set()
    for key, value in raw.items():
        for word in _canon(key).split():
            if word:
                keys.add(word)
        for word in _canon(value).split():
            if word:
                values.add(word)
    return keys, values, keys - values


_KEY_TOKENS, _VALUE_TOKENS, _DIALECTAL_KEYS = _load_vocab()


#: Arabic proclitics that glue to the following word. When a whole token does
#: not match anything, the stem after one clitic is tried against the curated
#: evidence lists (e.g. ``وهل`` -> ``هل``, ``فماذا`` -> ``ماذا``). The
#: dictionary-derived medium evidence still requires an exact whole-word hit,
#: so no artificial Iraqi evidence is introduced by stripping.
_CLITIC_PREFIXES = "وفبكل"


def _classify_exact(token: str) -> str:
    """Return the evidence tier of a whole token (no clitic handling)."""
    if token in STRONG_IRAQI:
        return "strong_iraqi"
    if token in STRONG_MSA:
        return "strong_msa"
    if token in MSA_VERBS:
        return "msa_verbs"
    if token in LIGHT_MSA:
        return "light_msa"
    # Iraqi-only letters are a decisive signal on their own.
    if any(ch in token for ch in IRAQI_LETTERS):
        return "strong_iraqi"
    if token in _DIALECTAL_KEYS:
        return "medium_iraqi"
    return "neutral"


def _classify(token: str) -> str:
    """Classify a token, falling back to the clitic-stripped stem."""
    tier = _classify_exact(token)
    if tier != "neutral":
        return tier
    if token[:1] in _CLITIC_PREFIXES and len(token) > 1:
        stem = token[1:]
        if ARABIC_RE.search(stem):
            tier = _classify_exact(stem)
            if tier != "neutral":
                return tier
    return "neutral"


_EVIDENCE_WEIGHTS = {
    "strong_iraqi": STRONG_WEIGHT,
    "medium_iraqi": MEDIUM_WEIGHT,
    "strong_msa": STRONG_WEIGHT,
    "msa_verbs": MEDIUM_WEIGHT,
    "light_msa": LIGHT_WEIGHT,
}


@dataclass
class Detection:
    """Result of :func:`detect`."""

    dialect: str  # iraqi | mixed | msa | unknown
    score: float  # 0..1 — share of Iraqi evidence (1 = pure Iraqi)
    confidence: float  # 0..1 estimate of how sure we are
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_iraqi(self) -> bool:
        return self.dialect == "iraqi"

    def __repr__(self) -> str:
        return (
            f"Detection(dialect={self.dialect!r}, "
            f"score={self.score:.3f}, confidence={self.confidence:.3f})"
        )


def _confidence(
    score: float,
    iraqi_hits: int,
    msa_hits: int,
    evidence_tokens: int,
    total: int,
) -> float:
    """A transparent confidence heuristic.

    High when the verdict is decisive, the evidence is dense, and there is
    enough text to judge. A balanced ``mixed`` result is honestly reported as
    less confident.
    """
    if iraqi_hits + msa_hits == 0:
        return round(0.25, 3)
    decisiveness = min(1.0, abs(score - 0.5) / 0.5)
    density = min(1.0, evidence_tokens / max(1, total))
    sample = min(1.0, total / 5.0)
    return round(min(1.0, 0.25 + 0.75 * (0.5 * decisiveness + 0.35 * density + 0.15 * sample)), 3)


def detect(text: str, *, threshold: float | None = None) -> Detection:
    """Detect whether ``text`` is Iraqi dialect, MSA, mixed or unknown.

    ``threshold`` overrides :data:`IRAQI_THRESHOLD` for the ``iraqi`` label
    (the default 0.75 means: Iraqi wins when it holds ≥ 75% of the evidence).
    """
    iraqi_threshold = IRAQI_THRESHOLD if threshold is None else threshold
    normalized = _canon(text)
    tokens = word_tokenize(normalized)

    total = len(tokens)
    if total == 0 or not any(ARABIC_RE.search(t) for t in tokens):
        return Detection(
            dialect="unknown",
            score=0.0,
            confidence=0.0,
            details={"total": total, "iraqi_hits": 0, "marker_letters": 0,
                     "strong_markers": 0, "iraqi_evidence": 0.0,
                     "msa_evidence": 0.0, "evidence": {}, "tokens": tokens},
        )

    iraqi_weight = 0.0
    msa_weight = 0.0
    iraqi_hits = 0
    msa_hits = 0
    strong_markers = 0
    marker_letters = 0
    evidence_tokens = 0
    evidence: dict[str, str] = {}

    for token in tokens:
        kind = _classify(token)
        evidence[token] = kind
        weight = _EVIDENCE_WEIGHTS.get(kind, 0.0)
        if weight == 0.0:
            continue
        evidence_tokens += 1
        if kind.endswith("iraqi"):
            iraqi_weight += weight
            iraqi_hits += 1
            if kind == "strong_iraqi":
                strong_markers += 1
                if any(ch in token for ch in IRAQI_LETTERS):
                    marker_letters += 1
        else:
            msa_weight += weight
            msa_hits += 1
            if any(ch in token for ch in IRAQI_LETTERS):
                marker_letters += 1

    total_weight = iraqi_weight + msa_weight
    if total_weight == 0:
        # Arabic text with no dialectal evidence at all: default to MSA.
        score = 0.5
        dialect = "msa"
    else:
        score = round(min(1.0, iraqi_weight / total_weight), 4)
        if msa_weight == 0:
            dialect = "iraqi"
        elif iraqi_weight == 0:
            dialect = "msa"
        elif score > iraqi_threshold:
            dialect = "iraqi"
        elif score < MIXED_THRESHOLD:
            dialect = "msa"
        else:
            dialect = "mixed"

    return Detection(
        dialect=dialect,
        score=score,
        confidence=_confidence(score, iraqi_hits, msa_hits, evidence_tokens, total),
        details={
            "total": total,
            "iraqi_hits": iraqi_hits,
            "marker_letters": marker_letters,
            "strong_markers": strong_markers,
            "iraqi_evidence": round(iraqi_weight, 3),
            "msa_evidence": round(msa_weight, 3),
            "evidence": evidence,
            "tokens": tokens,
        },
    )


def is_iraqi(text: str, *, threshold: float | None = None) -> bool:
    """Return ``True`` if ``text`` is clearly Iraqi dialect."""
    return detect(text, threshold=threshold).dialect == "iraqi"


def is_arabic(text: str) -> bool:
    """Return ``True`` if ``text`` contains any Arabic script characters."""
    return bool(ARABIC_RE.search(text))