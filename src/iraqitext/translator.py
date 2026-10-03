"""Translation between Iraqi Arabic and MSA, in four ordered tiers.

The 0.2 engine was a flat dictionary: longest key first, one non-cascading
pass.  That works for phrases and single words but it cannot conjugate.  This
module keeps that engine intact and adds two layers around it, so the priority
order the library promises is explicit and testable::

    1. phrase      a fixed Iraqi expression as a whole (``شكو ماكو``,
                   ``دير بالك على روحك``, ``للدوام``)
    2. structure   clause-level rules: negation, the future marker ``راح``,
                   the present marker ``دا``
    3. morphology  a known verb form, pronoun, marker or clitic
                   (``أگلك`` -> ``أخبرك``, ``ما نكدر نجي``)
    4. word        one dictionary entry, then the morphology token pass, then
                   the word is left exactly as it was

Rules:

* a rule only ever matches text that was **in the input** — every replacement
  is parked behind a ``\\x00<n>\\x00`` sentinel, so a translation is never
  re-translated by a shorter rule;
* punctuation is never touched, because the tiers 1-3 work on whole tokens
  and tier 4 uses the same word-boundary regexes as before;
* a **reviewed** entry (dictionary.json / lexicon.json) always beats a
  generated morphological form for the same surface string — the engine is
  handed a *shadow* set and stays out of the way, so no existing translation
  can change by accident;
* the same priority is used in the other direction, where only tier 4 exists
  (MSA -> Iraqi) so ``to_iraqi`` behaves exactly as it did in 0.1/0.2.

The public API is unchanged and only grew::

    from iraqitext import IraqiTranslator
    t = IraqiTranslator()
    t.to_fusha("شلونك")                       # كيف حالك
    t.to_iraqi("كيف حالك")                     # شلونك
    t.to_fusha("راح أگلك شصار", report=True)   # a TranslationReport
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from .morphology import (
    DIALECTS,
    MORPHOLOGY_CONFIDENCE,
    MorphologyEngine,
    canon,
    load_lexicon,
)
from .rules import flexible_pattern
from .tokenizer import ARABIC_LETTERS, tokenize

BASE_DIR = Path(__file__).parent

# Arabic proclitics handled by the (opt-in) ``clitics`` mode.
CLITIC_LETTERS = "وفبكل"

#: Tier order, highest priority first.  ``docs/coverage.md`` prints this.
TIERS = ("phrase", "structure", "morphology", "word")

#: Confidence per tier, used by :class:`iraqitext.coverage.TranslationReport`.
TIER_CONFIDENCE = {
    "phrase": 0.95,
    "structure": 0.85,
    "morphology": MORPHOLOGY_CONFIDENCE,
    "word": 0.9,
    "lexicon": 0.9,
    "unchanged": 1.0,
}


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


#: Marks that belong to the word they sit on but that ``\w`` does not cover.
#:
#: Python's ``\w`` is ``str.isalnum()`` plus ``_``, and the Arabic harakat,
#: the shadda, the sukun and the tatweel are Unicode *marks* — neither
#: alphanumerics nor underscores.  So ``(?<!\w)`` reads a shadda as a word
#: boundary, and «خدّام» matched the entry «ام» in the middle of the word and
#: came out as «خدوالدة».  Every boundary therefore also refuses a mark.
_ARABIC_MARKS = r"\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640"


def _boundaries(boundary: str) -> tuple[str, str]:
    if boundary == "word":
        return (
            r"(?<!\w)(?<![" + _ARABIC_MARKS + r"])",
            r"(?!\w)(?![" + _ARABIC_MARKS + r"])",
        )
    if boundary == "space":
        return r"(?<!\S)", r"(?!\S)"
    raise ValueError(f"unknown boundary: {boundary!r} (use 'word' or 'space')")


def _compile(pairs, boundary: str = "word"):
    pre, post = _boundaries(boundary)
    return [
        (re.compile(pre + flexible_pattern(pattern) + post), replacement)
        for pattern, replacement in pairs
    ]


def _compile_named(pairs, boundary: str = "word"):
    """Like :func:`_compile` but each rule also remembers its key.

    The coverage report needs the *key* a rule came from, not only its
    compiled pattern, so a translated word can be traced back to
    ``dictionary.json``.
    """
    pre, post = _boundaries(boundary)
    return [
        (pattern, re.compile(pre + flexible_pattern(pattern) + post), replacement)
        for pattern, replacement in pairs
    ]


# Replacements are parked behind a ``\x00<n>\x00`` sentinel while the remaining
# rules run, then restored in one go at the end of :func:`_apply`.
_PLACEHOLDER_RE = re.compile("\x00([0-9]+)\x00")


#: Marks a phrase entry may end with, and that the input may already carry.
#: The interjection entries in ``dictionary.json`` bake their own ``؟`` into the
#: value («مو كتلك» -> «ألم أحذرك؟»), which is right when the Iraqi writer left
#: the mark out and wrong when they typed it.  :func:`_seam` drops the value's
#: copy only in the second case, and only when the two marks are the same.
_SEAM_MARKS = "؟!،؛.:،"


def _seam(match: re.Match, value: str) -> str:
    """Drop a duplicated mark where a replacement meets the input.

    ``re.sub`` gives the callback the match but not the text after it, yet
    ``Match.string`` and ``Match.end()`` together are enough: if the character
    right after the match is the very mark the replacement ends with, the
    input already supplies it and the rule must not add a second one.
    """
    if not value:
        return value
    tail = match.string[match.end() : match.end() + 1]
    if tail and tail in _SEAM_MARKS and value[-1] == tail:
        return value[:-1]
    return value


def _substitute(text: str, regex: re.Pattern, value: str, replaced: list[str]) -> str:
    return regex.sub(
        lambda match, _value=value: _park(replaced, _seam(match, _value)), text
    )


def _apply(text: str, rules, replaced: list[str] | None = None) -> str:
    """Apply every rule in one **non-cascading** pass.

    ``rules`` is ordered longest-key-first and each rule still sees the whole
    text, but every replacement is hidden behind a sentinel while the
    remaining rules run. A rule can therefore only match characters that were
    present in the *input*; a translation is never translated again by a later
    rule.

    Without this, a short entry silently rewrites the output of a longer one:
    with ``غير`` -> ``سوى`` and ``ثقيل دم`` -> ``غير مرح`` in the dictionary,
    the naive cascade turns ``ثقيل دم`` into ``سوى مرح`` because the ``غير``
    produced by the first rule is re-matched by the second.

    ``replaced`` lets a caller share one parking list across several passes —
    the pipeline in :class:`IraqiTranslator` uses it so one tier can never try
    to restore a placeholder that another tier created.  Called with two
    arguments the behaviour is exactly as it was in 0.1/0.2.
    """
    own = replaced is None
    if own:
        if "\x00" in text:
            # Defensive: a literal NUL in the input would be indistinguishable
            # from a sentinel. Double it so the placeholder pattern cannot match.
            text = text.replace("\x00", "\x00\x00")
        replaced = []
        for rule in rules:
            text = _substitute(text, rule[-2], rule[-1], replaced)
        text = _PLACEHOLDER_RE.sub(lambda m: replaced[int(m.group(1))], text)
        return text.replace("\x00\x00", "\x00") if "\x00" in text else text

    for rule in rules:
        text = _substitute(text, rule[-2], rule[-1], replaced)
    return text


def _apply_recording(
    text: str, rules, replaced: list[str], on_hit
) -> str:
    """Like :func:`_apply` with a shared list, but it reports every firing.

    ``on_hit`` is called with the pattern that matched and its replacement, so
    the coverage report can say *which* dictionary key produced each output
    word instead of only reporting the structure tier.
    """
    for pattern, regex, replacement in rules:
        def _sub(match: re.Match, _p=pattern, _v=replacement):
            value = _seam(match, _v)
            on_hit(_p, value, match.group(0))
            return _park(replaced, value)

        text = regex.sub(_sub, text)
    return text


def _park(replaced: list[str], value: str) -> str:
    replaced.append(value)
    return "\x00%d\x00" % (len(replaced) - 1)


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


# --------------------------------------------------------------------------- #
# Tier 2: clause-level structure rules
# --------------------------------------------------------------------------- #
# A *clause* is a run of word tokens; these are the tokens that end one.
_CLAUSE_STOP_TOKENS = frozenset({"و", "ف", "عل", "لكن", "بس", "يعني", "means"})

#: Tokens that open a new clause and therefore end the previous one.
_CLAUSE_OPENERS = frozenset({"دا", "داك", "راح", "رح", "ما", "مو", "ماكو", "هو"})


_ARABIC_WORD_RE = re.compile("[" + ARABIC_LETTERS + r"]+")


def _is_word_token(token: str) -> bool:
    return bool(_ARABIC_WORD_RE.match(token))


class _Park:
    """The shared replacement list + text used by every tier.

    All tiers write into one list, so a placeholder produced by tier 1 and one
    produced by tier 3 can never be confused, and the final restore is a single
    pass over the text.
    """

    __slots__ = ("replaced", "text", "_had_nul")

    def __init__(self, text: str) -> None:
        self._had_nul = "\x00" in text
        if self._had_nul:
            text = text.replace("\x00", "\x00\x00")
        self.replaced: list[str] = []
        self.text = text

    def park(self, value: str) -> str:
        self.replaced.append(value)
        return "\x00%d\x00" % (len(self.replaced) - 1)

    def park_span(self, start: int, end: int, value: str) -> None:
        """Replace ``text[start:end]`` with a parked copy of ``value``."""
        self.text = self.text[:start] + self.park(value) + self.text[end:]

    def finish(self) -> str:
        text = _PLACEHOLDER_RE.sub(
            lambda m: self.replaced[int(m.group(1))], self.text
        )
        if self._had_nul:
            text = text.replace("\x00\x00", "\x00")
        return text


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
    dialect:
        Optional dialect tag (``baghdad``/``south``/``mosul``/``west``/
        ``general``). It only *adds* region-specific data: a word that is not
        tagged for the requested region is simply not offered, and a word
        tagged ``general`` always works.  ``None`` (the default) means "use
        everything".
    morphology:
        Set to ``False`` to run the 0.2 engine and nothing else: no generated
        verb forms, no clause templates and no ``lexicon.json`` entries.  Every
        stored ``dictionary.json`` value then comes back exactly as it is
        stored, which is what makes the flag useful for measuring what the
        new layers add and for bisecting a bad translation.
    """

    def __init__(
        self,
        *,
        strict_spaces: bool = False,
        clitics: bool = False,
        dialect: str | None = None,
        morphology: bool = True,
    ):
        if dialect is not None and dialect not in DIALECTS:
            raise ValueError(
                f"unknown dialect {dialect!r} (use one of {', '.join(DIALECTS)})"
            )
        self.boundary = "space" if strict_spaces else "word"
        self.clitics = bool(clitics)
        self.dialect = dialect
        self.morphology_enabled = bool(morphology)

        self._fusha_map = dict(_FUSHA_PAIRS)
        self._iraqi_map = dict(_IRAQI_PAIRS)

        phrase_pairs = [p for p in _FUSHA_PAIRS if " " in p[0]]
        word_pairs = [p for p in _FUSHA_PAIRS if " " not in p[0]]
        # Named rules keep the key next to the regex so the coverage report can
        # point at the exact dictionary line that produced a translation.
        self._phrase_rules = _compile_named(phrase_pairs, self.boundary)
        self._word_rules = _compile_named(word_pairs, self.boundary)
        self._to_iraqi_rules = _compile(_IRAQI_PAIRS, self.boundary)
        self._max_phrase_len = max((len(key) for key, _ in phrase_pairs), default=0)

        # --- structured lexicon (dialect aware) -------------------------- #
        # ``morphology=False`` means "the 0.2 engine and nothing else", so the
        # lexicon is left out as well: an ``override`` entry deliberately
        # replaces a stored value, and an escape hatch that quietly changed
        # stored values would not be an escape hatch.
        self.lexicon = (
            load_lexicon(dialect=dialect) if self.morphology_enabled else []
        )
        self._lexicon_map: dict[str, str] = {}
        for entry in self.lexicon:
            self._lexicon_map.setdefault(entry.iraqi, entry.fusha)

        # An entry marked ``override`` deliberately replaces a value that is
        # already in the dictionary, so it runs *before* the dictionary rules.
        # Every other entry is additive: it runs *after* them, which means it
        # can fill a gap but can never change a reviewed translation.
        self._lexicon_phrase_rules = self._compile_lexicon(multiword=True, override=True)
        self._lexicon_phrase_extra = self._compile_lexicon(multiword=True, override=False)
        self._lexicon_word_rules = self._compile_lexicon(multiword=False, override=True)
        self._lexicon_word_extra = self._compile_lexicon(multiword=False, override=False)

        # --- morphology ------------------------------------------------- #
        # The shadow is every surface string a reviewed entry already covers.
        # A generated form never overrules a human decision.  It is canonicalised
        # because that is how the engine looks words up.
        shadow = {canon(word) for word in self._fusha_map} | {
            canon(word) for word in self._lexicon_map
        }
        self.morphology = (
            MorphologyEngine(dialect=dialect, shadow=shadow)
            if self.morphology_enabled
            else None
        )
        self._compositions = self._load_compositions()

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _load_compositions(self) -> list[dict[str, Any]]:
        if self.morphology is None:
            return []
        return list(self.morphology.compositions)

    def _compile_lexicon(self, *, multiword: bool, override: bool) -> list:
        """Compile the lexicon rules for one group of entries.

        ``multiword`` splits phrases from single words; ``override`` splits the
        entries that are allowed to replace a dictionary value from the ones
        that are only allowed to fill a gap.
        """
        pairs = [
            (entry.iraqi, entry.fusha)
            for entry in self.lexicon
            if entry.is_multiword == multiword and entry.override == override
        ]
        pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
        return _compile_named(pairs, self.boundary)

    def _phrase_covers(self, text: str, start: int, end: int) -> bool:
        """True when a fixed phrase rule already knows ``text[start:end]``.

        The structure tier runs *before* the phrase replacements (it has to:
        ``راح`` alone is a dictionary word) but must never steal a span that a
        reviewed phrase already covers.  This is that guard.

        A rule counts when it *contains* the whole span, in either direction:

        * a longer phrase wins — ``ليش ما جيت`` is one entry, so the negation
          rule must not rewrite its tail into ``لماذا لا أتيت``;
        * a phrase that only starts inside the span does not — ``دا أحچي وياك``
          still becomes ``أتحدث معك الآن`` even though ``أحچي وياك`` and
          ``دا أحچي`` are both entries on their own.
        """
        # A phrase that contains the span may start up to one key-length before
        # it and end up to one key-length after it.
        floor = max(0, start - self._max_phrase_len)
        ceiling = min(len(text), end + self._max_phrase_len)
        for _pattern, regex, _value in self._phrase_rules + self._lexicon_phrase_rules:
            match = regex.search(text, floor, ceiling)
            if match is not None and match.start() <= start and match.end() >= end:
                return True
        return False

    # ------------------------------------------------------------------ #
    # tiers
    # ------------------------------------------------------------------ #
    def _tier_phrase(self, park: _Park, hits: list | None = None) -> None:
        """Tier 1: fixed phrases, longest first.

        Overriding lexicon phrases first, then every dictionary phrase, then
        the additive lexicon phrases — so a reviewed phrase always wins and a
        new lexicon entry can only fill a gap.
        """
        park.text = self._apply_tier(
            park, self._lexicon_phrase_rules, "lexicon", hits
        )
        park.text = self._apply_tier(park, self._phrase_rules, "phrase", hits)
        park.text = self._apply_tier(
            park, self._lexicon_phrase_extra, "lexicon", hits
        )

    def _apply_tier(
        self,
        park: _Park,
        rules,
        tier: str,
        hits: list | None,
    ) -> str:
        """One non-cascading pass that records what fired into ``hits``."""
        if hits is None:
            return _apply(park.text, rules, park.replaced)

        def _on_hit(pattern: str, value: str, matched: str) -> None:
            hits.append(
                {
                    "rule": pattern,
                    "tier": tier,
                    "iraqi": matched,
                    "fusha": value,
                }
            )

        return _apply_recording(park.text, rules, park.replaced, _on_hit)

    def _tier_structure(self, park: _Park) -> list[dict[str, Any]]:
        """Tier 2: negation / future / present clause restructuring.

        Returns one record per fired rule, for the report.
        """
        hits: list[dict[str, Any]] = []
        while True:
            tokens = tokenize(park.text)
            fired = None
            for index, token in enumerate(tokens):
                for comp in self._compositions:
                    if canon(token.text) not in {
                        canon(m) for m in comp.get("markers", ())
                    }:
                        continue
                    hit = self._apply_composition(park, comp, tokens, index)
                    if hit is not None:
                        fired = (token, hit)
                        break
                if fired:
                    break
            if fired is None:
                return hits
            token, hit = fired
            hits.append(hit)

    def _apply_composition(
        self, park: _Park, comp: dict[str, Any], tokens: list, index: int
    ) -> dict[str, Any] | None:
        """Try to fire one composition at ``tokens[index]``."""
        op = comp.get("op")
        if op == "future":
            return self._apply_future(park, comp, tokens, index)
        if op == "negation":
            return self._apply_negation(park, comp, tokens, index)
        if op == "present_marker":
            return self._apply_present_marker(park, comp, tokens, index)
        return None  # pragma: no cover - unknown op in the data

    # -- helpers shared by the compositions ----------------------------- #
    def _word_tokens(self, tokens: list, start: int) -> list[tuple[int, str]]:
        """Word tokens after ``start`` with their offsets in ``park.text``."""
        out: list[tuple[int, str]] = []
        index = start
        while index < len(tokens):
            token = tokens[index]
            if not token.is_word:
                break
            if index > start and canon(token.text) in _CLAUSE_STOP_TOKENS:
                break
            if index > start and canon(token.text) in _CLAUSE_OPENERS:
                break
            out.append((token.start, token.text))
            index += 1
        return out

    def _verb_msa(self, word: str) -> tuple[str, dict[str, Any]] | None:
        """MSA form of ``word`` as a *verb*, plus the match, or ``None``."""
        if self.morphology is None:
            return None
        match = self.morphology.verb_match(word)
        if match is None or match.msa == word:
            return None
        return match.msa, {
            "verb": match.verb,
            "tense": match.tense,
            "person": match.person,
        }

    def _apply_future(
        self, park: _Park, comp: dict[str, Any], tokens: list, index: int
    ) -> dict[str, Any] | None:
        """``راح أگلك`` -> ``سأخبرك`` (the marker is deleted, not translated)."""
        marker = tokens[index]
        rest = self._word_tokens(tokens, index + 1)
        if not rest:
            return None
        start, word = rest[0]
        verb = self._verb_msa(word)
        if verb is None:
            return None
        # راح + a *clitic-attached* verb: أخبرك is really راح + أگلك, but a
        # bare راح can also mean "will" on its own, so require a real verb.
        end = start + len(word)
        if comp.get("requires_verb") and verb[0] == word:
            return None
        if self._phrase_covers(park.text, marker.start, end):
            return None
        msa = comp.get("msa", "س") + verb[0]
        park.park_span(marker.start, end, msa)
        return {
            "rule": comp["id"],
            "tier": "structure",
            "iraqi": f"{marker.text} {word}",
            "fusha": msa,
            "span": (marker.start, end),
            "detail": verb[1],
        }

    def _apply_negation(
        self, park: _Park, comp: dict[str, Any], tokens: list, index: int
    ) -> dict[str, Any] | None:
        """``ما نكدر نجي`` -> ``لا نستطيع أن نأتي``."""
        marker = tokens[index]
        rest = self._word_tokens(tokens, index + 1)
        verbs: list[tuple[int, str, str, dict[str, Any]]] = []
        for start, word in rest:
            if comp.get("requires_verb", True) and not self.morphology.is_verb_like(word):
                continue
            verb = self._verb_msa(word)
            if verb is None:
                continue
            verbs.append((start, word, verb[0], verb[1]))
            if len(verbs) == 2:
                break
        if not verbs:
            return None
        first_past = verbs[0][3].get("tense") == "past"
        end = verbs[-1][0] + len(verbs[-1][1])
        if self._phrase_covers(park.text, marker.start, end):
            return None

        if len(verbs) == 1:
            if first_past:
                out = comp.get("msa_single_past", "لم") + " " + verbs[0][2]
            else:
                out = comp.get("msa_present", "لا") + " " + verbs[0][2]
        elif first_past:
            out = (
                comp.get("msa_past", "ما")
                + " "
                + verbs[0][2]
                + " "
                + comp.get("complement_past", "لـ")
                + verbs[1][2]
            )
        else:
            out = (
                comp.get("msa_present", "لا")
                + " "
                + verbs[0][2]
                + " "
                + comp.get("complement_present", "أن")
                + " "
                + verbs[1][2]
            )
        park.park_span(marker.start, end, out)
        return {
            "rule": comp["id"],
            "tier": "structure",
            "iraqi": f"{marker.text} " + " ".join(v[1] for v in verbs),
            "fusha": out,
            "span": (marker.start, end),
            "detail": {"verbs": [v[3] for v in verbs]},
        }

    def _apply_present_marker(
        self, park: _Park, comp: dict[str, Any], tokens: list, index: int
    ) -> dict[str, Any] | None:
        """``دا أحچي وياك`` -> ``أتحدث معك الآن`` (marker moves to the end)."""
        marker = tokens[index]
        rest = self._word_tokens(tokens, index + 1)
        if not rest:
            return None
        if comp.get("requires_verb", True):
            head = self._verb_msa(rest[0][1])
            if head is None:
                return None
        clause_start = rest[0][0]
        clause_end = rest[-1][0] + len(rest[-1][1])
        if self._phrase_covers(park.text, marker.start, clause_end):
            return None
        # The clause keeps its own translation; only the position of «الآن»
        # changes, so translate it with the tiers below (no structure tier,
        # which would recurse).
        inner = self._translate_fragment(park.text[clause_start:clause_end])
        out = f"{inner} {comp.get('msa', 'الآن')}"
        iraqi_clause = park.text[marker.start:clause_end]
        park.park_span(marker.start, clause_end, out)
        return {
            "rule": comp["id"],
            "tier": "structure",
            "iraqi": iraqi_clause,
            "fusha": out,
            "span": (marker.start, clause_end),
            "detail": {"clause": inner},
        }

    def _translate_fragment(self, text: str) -> str:
        """Translate a clause with tiers 1, 3, 4 (never tier 2)."""
        park = _Park(text)
        self._tier_phrase(park)
        self._tier_morphology(park)
        self._tier_word(park)
        return park.finish()

    def _tier_morphology(self, park: _Park, hits: list | None = None) -> None:
        """Tier 3: known verb forms, pronouns, markers and particles.

        A generated form carries the *engine's* confidence, not the tier
        default, because a conjugation is an inference while a dictionary
        entry is a decision.
        """
        if self.morphology is None:
            return
        pre, post = _boundaries(self.boundary)
        token_re = re.compile(pre + "[" + ARABIC_LETTERS + r"]+" + post)

        def _replacer(match: re.Match) -> str:
            word = match.group(0)
            result = self.morphology.translate_token(word)
            if result is None:
                return word
            if hits is not None:
                hits.append(
                    {
                        "rule": result.rule,
                        "tier": "morphology",
                        "iraqi": word,
                        "fusha": result.msa,
                        "kind": result.kind,
                        "confidence": result.confidence,
                    }
                )
            return park.park(result.msa)

        park.text = token_re.sub(_replacer, park.text)

    def _tier_word(self, park: _Park, hits: list | None = None) -> None:
        """Tier 4: one reviewed entry — or the word stays as it is.

        Overriding lexicon words, the dictionary, then the additive lexicon
        words.  Anything still untranslated is *left alone*: an unknown Iraqi
        word is a dictionary gap, not a mistake to guess at.
        """
        park.text = self._apply_tier(
            park, self._lexicon_word_rules, "lexicon", hits
        )
        park.text = self._apply_tier(park, self._word_rules, "word", hits)
        park.text = self._apply_tier(
            park, self._lexicon_word_extra, "lexicon", hits
        )

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def to_fusha(
        self,
        text: str,
        *,
        report: bool = False,
        dialect: str | None = None,
    ):
        """Translate Iraqi dialect text to Modern Standard Arabic.

        With ``report=True`` a :class:`iraqitext.coverage.TranslationReport` is
        returned instead of a plain string; it carries the output, the per-tier
        rules that fired, a confidence per token and the tokens that were left
        untouched.  ``dialect`` overrides the instance tag for one call.
        """
        if dialect is not None and dialect != self.dialect:
            return IraqiTranslator(
                strict_spaces=self.boundary == "space",
                clitics=self.clitics,
                dialect=dialect,
                morphology=self.morphology_enabled,
            )._to_fusha_full(text, report=report)

        return self._to_fusha_full(text, report=report)

    def _to_fusha_full(self, text: str, *, report: bool = False):
        if self.clitics:
            text = _clitic_pass(text, self._fusha_map)

        if not self.morphology_enabled:
            # Legacy path, byte for byte the 0.2 behaviour.
            park = _Park(text)
            legacy_hits: list[dict[str, Any]] = []
            self._tier_phrase(park, legacy_hits)
            self._tier_word(park, legacy_hits)
            staged = park.text
            out = park.finish()
            if report:
                from .coverage import build_report

                return build_report(text, out, legacy_hits, staged=staged)
            return out

        park = _Park(text)
        hits: list[dict[str, Any]] = []

        structure = self._tier_structure(park)
        hits.extend(structure)

        # A structure rule may have produced a whole clause, so the phrase tier
        # runs after it on what is still input text.
        self._tier_phrase(park, hits)
        self._tier_morphology(park, hits)
        self._tier_word(park, hits)

        # ``park.text`` still holds the sentinels here, which is what lets the
        # report tell "a rule replaced this" from "nothing knew this word".
        staged = park.text
        out = park.finish()
        if report:
            from .coverage import build_report

            return build_report(
                text,
                out,
                hits,
                staged=staged,
                engine=self.morphology,
                dialect=self.dialect,
            )
        return out

    def to_iraqi(self, text: str) -> str:
        """Translate Modern Standard Arabic text to Iraqi dialect.

        Unchanged from 0.1/0.2: one non-cascading pass over the reversed
        dictionary, because there is no morphological model for the MSA ->
        Iraqi direction yet.
        """
        if self.clitics:
            text = _clitic_pass(text, self._iraqi_map)
        return _apply(text, self._to_iraqi_rules)

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #
    def coverage_stats(self) -> dict[str, Any]:
        """Counts used by ``docs/coverage.md`` and ``CoverageLog``."""
        from .coverage import coverage_stats

        return coverage_stats(self)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        bits = [f"dialect={self.dialect!r}"]
        if self.clitics:
            bits.append("clitics=True")
        if self.morphology is None:
            bits.append("morphology=False")
        return f"IraqiTranslator({', '.join(bits)})"
