"""Rule-based Iraqi Arabic morphology: verbs, pronouns, clitics, negation, tense.

The dictionary engine in :mod:`iraqitext.translator` knows *words*; this module
knows *grammar*. It answers two questions about a single Iraqi word:

1. :meth:`MorphologyEngine.verb_match` — "is this a verb, and which
   person/number/tense/object is it?" Used by the structure pass to recognise
   ``ما نكدر نجي`` as ``NEG + verb(1pl) + verb(1pl)``.
2. :meth:`MorphologyEngine.translate_token` — "what is the MSA form?" Used by
   the morphology tier of the translation pipeline.

The data lives in ``morphology.json`` and is deliberately small: one Iraqi
*stem* plus an MSA stem (or an explicit paradigm) is expanded into the whole
paradigm, so a single entry covers dozens of surface forms
(``راح`` → ``أروح/تروح/يروح/نروح`` + ``راحنا/راحوا/راحتوا`` and the object
suffixes ``أروحي/أروحك/أروحچ/أروحل/أروحها/أروحلهم`` …).

Two design rules keep the engine honest:

* **never guess from a bare stem when the word is ambiguous.** A verb may set
  ``allow_bare`` to ``"3ms"``/``"1sg"``/``false``; ``سوى`` sets it to ``false``
  because ``على سوى`` is a preposition and not the verb ``فعل``.
* **a reviewed dictionary entry always wins over a generated form.** The
  translator hands the engine a *shadow* set of surface forms that
  ``dictionary.json`` / ``lexicon.json`` already cover; the engine then stays
  out of the way so no existing translation can change by accident.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .normalize import normalize

BASE_DIR = Path(__file__).parent

MORPHOLOGY_PATH = BASE_DIR / "morphology.json"
LEXICON_PATH = BASE_DIR / "lexicon.json"

#: Dialect tags accepted by :func:`load_lexicon` / :class:`MorphologyEngine`.
DIALECTS = ("baghdad", "south", "mosul", "west", "general")

#: Confidence attached to a morphological (generated) translation. A full
#: phrase outranks it, which is why the default sits below the dictionary tier.
MORPHOLOGY_CONFIDENCE = 0.75

#: Harakat / shadda. Iraqi uses the shadda *phonemically* to tell forms apart
#: (``خلي`` = أترك vs ``خلّي`` = ترك), and the data is written without tashkeel,
#: so a token that carries one is left to the dictionary instead of guessed.
TASHKEEL_RE = re.compile(r"[ً-ٰٟۖ-ۭـ]")


def canon(word: str) -> str:
    """Canonical form used for lookups (same folding as the tolerant matcher).

    Alef variants, hamza seats, Persian letters and ة/ه are unified, matching
    :func:`iraqitext.rules.flexible_pattern`. ``ى`` is *not* unified with ``ي``
    because the tolerant matcher keeps them apart too.
    """
    return normalize(
        word,
        unify_teh_marbuta=True,
        unify_alef_maqsura=False,
        collapse_repeats=False,
        punctuation=False,
        whitespace=True,
    )


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# Data objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VerbEntry:
    """One Iraqi verb: its stems, its MSA paradigm and its quirks."""

    key: str
    msa_root: str
    msa_stems: dict[str, str] = field(default_factory=dict)
    msa_prefixes: dict[str, str] = field(default_factory=dict)
    msa_gender_suffix: dict[str, str] = field(default_factory=dict)
    msa_number_suffix: dict[str, str] = field(default_factory=dict)
    msa_past_subject: dict[str, str] = field(default_factory=dict)
    paradigms: dict[str, dict[str, str]] = field(default_factory=dict)
    forms: dict[str, str] = field(default_factory=dict)
    form_tense: dict[str, str] = field(default_factory=dict)
    iraqi_stems: dict[str, str] = field(default_factory=dict)
    alt_stems: dict[str, tuple[str, ...]] = field(default_factory=dict)
    allow_bare: str | bool = "3ms"
    object_mode: str = "suffix"
    drop_alef_past: bool = False
    bare_tense: str = "past"
    region: tuple[str, ...] = ("general",)
    note: str = ""
    example: dict[str, str] | None = None

    def applies_to(self, dialect: str | None) -> bool:
        """True if the entry may be used for ``dialect`` (None = no filter)."""
        if dialect is None or "general" in self.region:
            return True
        return dialect in self.region

    def all_stems(self, tense: str) -> tuple[str, ...]:
        """Every Iraqi stem of ``tense``: the main one plus any alternates."""
        stems: list[str] = []
        main = self.iraqi_stems.get(tense)
        if main:
            stems.append(main)
        stems.extend(self.alt_stems.get(tense, ()))
        return tuple(stems)

    def shares_past_stem(self, tense: str) -> bool:
        """True if ``tense`` uses the same Iraqi stem as the past.

        For those verbs the perfective suffixes (``-نا``/``-وا``/``-توا``) are
        *subject* markers (``عرفنا`` = knew-we) rather than object clitics, so
        ``عرفنا`` must not come out as ``نعرفنا``.
        """
        present = self.iraqi_stems.get(tense)
        past = self.iraqi_stems.get("past")
        return bool(present) and present == past


@dataclass(frozen=True)
class VerbMatch:
    """The analysis of one Iraqi verb token."""

    token: str
    verb: str
    tense: str
    person: str
    msa: str
    object: str | None = None
    proclitics: tuple[str, ...] = ()
    stem: str = ""
    derived: bool = True

    @property
    def rule(self) -> str:
        return f"verb.{self.verb}.{self.tense}"


@dataclass(frozen=True)
class TokenTranslation:
    """A single token translated by the morphology layer."""

    token: str
    msa: str
    rule: str
    kind: str
    confidence: float = MORPHOLOGY_CONFIDENCE
    detail: dict[str, Any] = field(default_factory=dict)


class LexiconEntry:
    """One structured lexicon entry (``lexicon.json``)."""

    __slots__ = (
        "iraqi", "fusha", "type", "region", "tags", "example", "source",
        "confidence", "override", "id", "note",
    )

    def __init__(self, raw: dict[str, Any]):
        self.id = raw.get("id", "")
        self.iraqi = raw["iraqi"]
        self.fusha = raw["fusha"]
        self.type = raw.get("type", "word")
        self.region = tuple(raw.get("region", ("general",)))
        self.tags = tuple(raw.get("tags", ()))
        self.example = raw.get("example", "")
        self.source = raw.get("source", "manual")
        self.confidence = float(raw.get("confidence", 0.9))
        self.override = bool(raw.get("override", False))
        self.note = raw.get("note", "")

    @property
    def is_multiword(self) -> bool:
        return " " in self.iraqi.strip()

    def applies_to(self, dialect: str | None) -> bool:
        if dialect is None or "general" in self.region:
            return True
        return dialect in self.region

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LexiconEntry({self.iraqi!r} → {self.fusha!r}, {self.type})"


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class MorphologyEngine:
    """Analyzes and generates Iraqi verb forms, pronouns and markers."""

    def __init__(
        self,
        data: dict[str, Any] | None = None,
        *,
        dialect: str | None = None,
        shadow: Iterable[str] = (),
    ) -> None:
        self.data = data if data is not None else _load_json(MORPHOLOGY_PATH)
        self.dialect = dialect
        self.shadow = frozenset(shadow)

        self.msa_prefixes: dict[str, str] = self.data["msa_prefixes"]
        self.msa_gender_suffix: dict[str, str] = self.data["msa_gender_suffix"]
        self.msa_number_suffix: dict[str, str] = self.data["msa_number_suffix"]
        self.msa_past_subject: dict[str, str] = self.data["msa_past_subject"]
        self.person_prefixes: dict[str, dict] = self.data["person_prefixes"]
        self.plural_suffixes: dict[str, dict] = self.data["plural_suffixes"]
        self.subject_suffixes_past: dict[str, dict] = self.data["subject_suffixes_past"]
        self.subject_suffixes_present: dict[str, dict] = self.data["subject_suffixes_present"]
        self.proclitics: dict[str, dict] = self.data["proclitics"]
        self.object_suffixes: dict[str, dict] = self.data["object_suffixes"]
        self.pronouns: dict[str, dict] = self.data["pronouns"]
        self.negation: dict[str, dict] = self.data["negation"]
        self.markers: dict[str, dict[str, dict]] = self.data["markers"]
        self.compositions: list[dict[str, Any]] = list(self.data["compositions"])
        self.protected = frozenset(canon(w) for w in self.data["protected"])
        self.shared_msa = frozenset(canon(w) for w in self.data["shared_msa"])
        self.proper_nouns = frozenset(canon(w) for w in self.data["proper_nouns"])
        #: Same words, canonicalised, for callers that look them up themselves
        #: (the coverage report classifies every token it did not translate).
        self.proper_noun_canon = self.proper_nouns
        self.shared_msa_canon = self.shared_msa
        self.verb_like = frozenset(canon(w) for w in self.data["verb_like"])

        self.verbs: dict[str, VerbEntry] = {
            key: self._build_verb(key, raw)
            for key, raw in self.data["verbs"].items()
        }

        # stem -> [(verb key, tense)] and canonical surface -> explicit form
        self._stem_index: dict[str, list[tuple[str, str]]] = {}
        for key, verb in self.verbs.items():
            if not verb.applies_to(self.dialect):
                continue
            for tense in self._tenses(verb):
                for stem in verb.all_stems(tense):
                    self._stem_index.setdefault(canon(stem), []).append((key, tense))
        self._form_index: dict[str, tuple[str, str, str]] = {}
        for key, verb in self.verbs.items():
            if not verb.applies_to(self.dialect):
                continue
            for surface, msa in verb.forms.items():
                self._form_index.setdefault(
                    surface, (msa, key, verb.form_tense.get(surface, ""))
                )
        self._pronoun_index = {canon(k): v for k, v in self.pronouns.items()}
        self._marker_index = {
            canon(word): (kind, meta)
            for kind, group in self.markers.items()
            for word, meta in group.items()
        }
        # At most one proclitic.  Iraqi writes ``وفهمه`` and ``كفهمه`` but not
        # two in a row, and allowing a stack invents verbs: with two, ``كليجة``
        # parses as ك + ل + ``يج`` and becomes the verb جه.
        self._max_proclitics = 1
        self._max_suffix = max((len(s) for s in self.object_suffixes), default=0)
        self._max_subject = max(
            (
                len(s)
                for table in (self.subject_suffixes_past, self.subject_suffixes_present)
                for s in table
            ),
            default=0,
        )

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    def _build_verb(self, key: str, raw: dict[str, Any]) -> VerbEntry:
        forms: dict[str, str] = {}
        form_tense: dict[str, str] = {}
        for surface, value in raw.get("forms", {}).items():
            # a form is either "msa" or {"msa": ..., "tense": "past"}
            if isinstance(value, dict):
                forms[canon(surface)] = value["msa"]
                form_tense[canon(surface)] = value.get("tense", "")
            else:
                forms[canon(surface)] = value
        return VerbEntry(
            key=key,
            msa_root=raw.get("msa_root", key),
            msa_stems=dict(raw.get("msa_stems", {})),
            msa_prefixes=dict(raw.get("msa_prefixes", self.msa_prefixes)),
            msa_gender_suffix=dict(raw.get("msa_gender_suffix", self.msa_gender_suffix)),
            msa_number_suffix=dict(raw.get("msa_number_suffix", self.msa_number_suffix)),
            msa_past_subject=dict(raw.get("msa_past_subject", self.msa_past_subject)),
            paradigms={t: dict(p) for t, p in raw.get("paradigms", {}).items()},
            forms=forms,
            form_tense=form_tense,
            iraqi_stems={t: canon(s) for t, s in raw.get("iraqi_stems", {}).items()},
            alt_stems={
                t: tuple(canon(s) for s in v)
                for t, v in raw.get("alt_stems", {}).items()
            },
            allow_bare=raw.get("allow_bare", "3ms"),
            object_mode=raw.get("object_mode", "suffix"),
            drop_alef_past=bool(raw.get("drop_alef_past", False)),
            bare_tense=raw.get("bare_tense", "past"),
            region=tuple(raw.get("region", ("general",))),
            note=raw.get("note", ""),
            example=raw.get("example"),
        )

    # ------------------------------------------------------------------ #
    # paradigm
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tenses(verb: VerbEntry) -> tuple[str, ...]:
        """The tenses a verb declares, present first, without duplicates."""
        return tuple(
            dict.fromkeys(list(verb.iraqi_stems) + list(verb.alt_stems))
        )

    def person_forms(self, verb: VerbEntry, tense: str) -> dict[str, str]:
        """Return ``person -> msa form`` for ``tense``.

        Explicit ``paradigms`` win; otherwise the paradigm is derived from
        ``msa_stems[tense]`` plus the person prefix / suffix tables.
        """
        if tense in verb.paradigms:
            return verb.paradigms[tense]
        stem = verb.msa_stems.get(tense)
        if not stem:
            return {}
        forms: dict[str, str] = {}
        for person, prefix in verb.msa_prefixes.items():
            if tense == "past":
                if person == "1sg":
                    forms[person] = stem + verb.msa_past_subject.get("2ms", "ت")
                    continue
                suffix = verb.msa_past_subject.get(person)
                if suffix is None:
                    continue
                if (
                    verb.drop_alef_past
                    and suffix in ("نا", "توا", "تم")
                    and stem.endswith("ا")
                ):
                    forms[person] = stem[:-1] + suffix
                else:
                    forms[person] = stem + suffix
            else:
                form = prefix + stem
                if person in verb.msa_gender_suffix:
                    form += verb.msa_gender_suffix[person]
                if person in verb.msa_number_suffix:
                    form += verb.msa_number_suffix[person]
                forms[person] = form
        return forms

    def form_for(self, verb: VerbEntry, tense: str, person: str) -> str | None:
        return self.person_forms(verb, tense).get(person)

    def attach_object(self, verb: VerbEntry, form: str, obj: str, tense: str) -> str:
        """Attach an MSA object pronoun to ``form``."""
        if not obj:
            return form
        if verb.object_mode == "none":
            # عطى / أيّ in MSA do not take the enclitic pronouns
            return form
        if verb.object_mode == "raa" and tense == "present" and form.endswith("ى"):
            # رأى family: أرى + ك → أراك, ترى + ها → تراها
            return form[:-1] + "ا" + obj
        if obj.startswith("ل"):
            # the prepositional clitic is a separate word: أرسل لك
            return form + " " + obj
        return form + obj

    # ------------------------------------------------------------------ #
    # analysis
    # ------------------------------------------------------------------ #
    def _split_prefixes(self, token: str) -> tuple[tuple[tuple[str, ...], str | None, str, bool], ...]:
        """Peel proclitics and a person prefix off ``token``.

        Returns a tuple of *variants*, each ``(proclitics, person, body,
        has_person_prefix)``.  Several readings are produced on purpose:
        ``فهم`` starts with the conjunction ``ف`` and the stem is ``فهم``, while
        ``فوق`` only parses once ``ف`` is given back, and ``تبعه`` only parses
        once the leading ``ت`` is given back.  The caller keeps the reading
        with the longest stem.
        """
        peeled: list[str] = []
        heads: list[str] = [token]
        while len(peeled) < self._max_proclitics and heads[-1][:1] in self.proclitics:
            peeled.append(heads[-1][0])
            heads.append(heads[-1][1:])

        variants: list[tuple[tuple[str, ...], str | None, str, bool]] = []
        for used in range(len(peeled), -1, -1):
            word = heads[used]
            used_proclitics = tuple(peeled[:used])
            if not word:
                continue
            variants.append((used_proclitics, None, word, False))
            head = word[:1]
            if head in self.person_prefixes and len(word) > 1:
                person = self.person_prefixes[head]["person"]
                body = word[1:]
                if head == "ن" and body[:1] in ("ت", "ي") and len(body) > 1:
                    # ن + ت + stem: the emphatic 1pl keeps the imperfective
                    # marker (نتشرگ، نتفهم، نتهام) which MSA folds into ن-.
                    body = body[1:]
                variants.append((used_proclitics, person, body, True))
        return tuple(variants)

    def _parse_tail(
        self, tail: str, tense: str, prefix_person: str | None, verb: VerbEntry
    ) -> tuple[str, str | None, str] | None:
        """Classify what follows an Iraqi verb stem.

        ``tail`` may be empty (bare), a plural marker (``يروحون``), an object
        clitic (``أگلك``), a subject suffix of the past (``راحنا``) or a
        combination of a plural marker and an object (``يگدرنهم``).

        Returns ``(subject_person, msa_object_pronoun_or_None, kind)`` where
        ``kind`` is ``bare`` / ``plural`` / ``subject_past`` / ``subject_present``
        / ``object`` and is what the scorer uses to pick the tense, or ``None``
        when the tail cannot belong to this verb/tense.
        """
        if not tail:
            if prefix_person:
                return prefix_person, None, "bare"
            if verb.allow_bare:
                return str(verb.allow_bare), None, "bare"
            return None

        rest = tail
        table = self.subject_suffixes_past if tense == "past" else self.subject_suffixes_present
        shared = verb.shares_past_stem(tense)

        # 1. a subject suffix of this tense (ين/ي/ن for the imperfective,
        #    ت/نا/وا/توا for the perfective)
        if rest in table:
            person = table[rest]["person"]
            if (
                tense == "present"
                and person == "1pl"
                and prefix_person not in (None, "1pl")
            ):
                # the imperfective 1pl is marked by the ن- prefix (نحچين)، so a
                # ت- or أ- in front of -ن cannot be that form
                return None
            if tense == "present" and shared and rest in self.subject_suffixes_past:
                # عرفنا / عرفوا: one stem for both tenses, so the perfective
                # suffix table is the right one even on the imperfective side.
                return self.subject_suffixes_past[rest]["person"], None, "subject_past"
            return person, None, f"subject_{tense}"
        if tense == "present" and shared and rest in self.subject_suffixes_past:
            return self.subject_suffixes_past[rest]["person"], None, "subject_past"

        # 2. an object clitic (maybe with a ن- plural marker inside it)
        if rest in self.object_suffixes:
            meta = self.object_suffixes[rest]
            subject = meta.get("subject") or prefix_person
            if not subject:
                if not verb.allow_bare:
                    return None
                subject = str(verb.allow_bare)
            return subject, meta["msa"], "object"

        # 3. a plural marker (يروحون / يگدرين / يروحهه)
        for marker, meta in self.plural_suffixes.items():
            if rest.startswith(marker):
                return meta["person"], None, "plural"
        return None

    def _candidates(
        self, token: str
    ) -> list[tuple[int, VerbEntry, str, str, str, str | None, tuple[str, ...]]]:
        """All plausible analyses of ``token`` (best first).

        Each item is ``(score, verb, tense, person, stem, object, proclitics)``.
        Scoring prefers the longest stem, then an explicit person prefix, then
        a bare (suffix-less) reading, so the most specific parse wins. The
        "is that leading ت really a prefix?" question is settled by the same
        scoring: ``تبعه`` only parses as *bare stem تبع + ه* while ``تعرفه``
        parses as *prefix ت + عرف + ه*.
        """
        if not token or canon(token) in self.protected:
            return []
        if token.endswith("ة"):
            # A word ending in ta marbuta is a feminine noun, never an Iraqi
            # verb form: verbs end in a consonant or in a pronoun written with
            # heh (``أعرفه``).  ``ليجة`` is "a car", not ل + يجه, and ``كليجة``
            # is a sweet, not ك + ل + يجه.  Without this the spelling fold
            # ة -> ه lets a noun swallow a verb stem.
            return []
        variants = self._split_prefixes(token)
        found: list[tuple[int, VerbEntry, str, str, str, str | None, tuple[str, ...]]] = []
        for proclitics, person_prefix, body, has_prefix in variants:
            if not body:
                continue
            for stem_len in range(len(body), 0, -1):
                stem = body[:stem_len]
                entries = self._stem_index.get(canon(stem))
                if not entries:
                    continue
                tail = body[stem_len:]
                for verb_key, tense in entries:
                    verb = self.verbs[verb_key]
                    parsed = self._parse_tail(tail, tense, person_prefix, verb)
                    if parsed is None:
                        continue
                    subject, obj, kind = parsed
                    if self.form_for(verb, tense, subject) is None:
                        continue
                    score = (
                        stem_len * 10
                        + (5 if has_prefix else 0)
                        + (3 if not tail else 0)
                        - 4 * len(proclitics)
                    )
                    score += self._tense_bonus(verb, tense, kind, has_prefix)
                    found.append((score, verb, tense, subject, stem, obj, proclitics))
        found.sort(key=lambda item: (-item[0], item[1].key))
        return found

    @staticmethod
    def _tense_bonus(verb: VerbEntry, tense: str, kind: str, has_prefix: bool) -> int:
        """Prefer the tense that the shape of the word actually implies.

        Iraqi marks tense on the *prefix* (``تروح``) and the plural marker
        (``يروحون``), so a present reading is the right one there; the
        perfective suffixes (``راحنا``/``عرفنا``) only exist on past stems.
        Object clitics (``أعرفه``) hang off the imperfective as well, and a bare
        shared-stem verb defaults to the perfective.
        """
        if kind.startswith("subject_"):
            return 6 if kind == f"subject_{tense}" else 0
        if kind == "plural":
            return 6 if tense == "present" else 0
        if kind == "object":
            return 4 if tense == "present" else 0
        # bare
        if has_prefix:
            return 4 if tense == "present" else 0
        if verb.shares_past_stem(tense):
            return 6 if tense == verb.bare_tense else 0
        return 0

    def verb_match(self, token: str) -> VerbMatch | None:
        """Analyse ``token`` as a verb, or return ``None``.

        Explicit ``forms`` win over the generated paradigm (``چالت`` → ``قالت``
        rather than the 2ms reading of the same shape).
        """
        token = token.strip()
        if not token:
            return None
        if TASHKEEL_RE.search(token):
            # خُليّ (ترك) vs خَليّ (كان) — a shadda decides the meaning, and the
            # data is written without harakat, so the word goes to the dictionary.
            return None
        if canon(token) in self.protected:
            return None
        explicit = self._form_index.get(canon(token))
        if explicit is not None:
            msa, verb_key, listed_tense = explicit
            return VerbMatch(
                token=token, verb=verb_key, tense=listed_tense or "listed",
                person="", msa=msa, derived=False,
            )
        candidates = self._candidates(token)
        if not candidates:
            return None
        _score, verb, tense, person, stem, obj, proclitics = candidates[0]
        form = self.form_for(verb, tense, person)
        if form is None:  # pragma: no cover - guarded by _candidates
            return None
        msa = form
        if proclitics:
            msa = "".join(self.proclitics[c]["msa"] for c in proclitics) + msa
        msa = self.attach_object(verb, msa, obj or "", tense)
        return VerbMatch(
            token=token, verb=verb.key, tense=tense, person=person, msa=msa,
            object=obj, proclitics=proclitics, stem=stem, derived=True,
        )

    def is_verb_like(self, token: str) -> bool:
        """True when ``token`` can head a clause (verb or known Iraqi verb)."""
        if self.verb_match(token) is not None:
            return True
        return canon(token) in self.verb_like

    # ------------------------------------------------------------------ #
    # translation
    # ------------------------------------------------------------------ #
    def translate_token(self, token: str) -> TokenTranslation | None:
        """Translate one Iraqi word with morphology, or return ``None``.

        ``None`` means "I have nothing to say about this word" — the caller then
        tries the dictionary and finally keeps the word as it is.  A word that
        the reviewed dictionary already covers returns ``None`` as well
        (see the *shadow* rule in the module docstring).
        """
        token = token.strip()
        if not token:
            return None
        if TASHKEEL_RE.search(token):
            return None
        key = canon(token)
        if key in self.protected or key in self.shadow:
            return None
        match = self.verb_match(token)
        if match is not None:
            return TokenTranslation(
                token=token,
                msa=match.msa,
                rule=match.rule,
                kind="verb",
                confidence=MORPHOLOGY_CONFIDENCE,
                detail={
                    "verb": match.verb,
                    "tense": match.tense,
                    "person": match.person,
                    "object": match.object,
                    "proclitics": list(match.proclitics),
                },
            )
        pronoun = self._pronoun_index.get(key)
        if pronoun is not None:
            return TokenTranslation(
                token=token, msa=pronoun["msa"], rule=f"pronoun.{canon(token)}",
                kind="pronoun", confidence=0.9,
                detail={"person": pronoun.get("person", "")},
            )
        # A pronoun behind a conjunction: وهيه is one token to the tokenizer
        # but the pronoun is هيه with و in front.  Verb forms get the same
        # treatment above, and the MSA conjunction is re-attached in front.
        if token[:1] in self.proclitics and len(token) > 2:
            clitic, rest = token[0], token[1:]
            inner = self._pronoun_index.get(canon(rest))
            if inner is not None:
                return TokenTranslation(
                    token=token,
                    msa=self.proclitics[clitic]["msa"] + inner["msa"],
                    rule=f"pronoun.{canon(rest)}",
                    kind="pronoun",
                    confidence=0.85,
                    detail={
                        "person": inner.get("person", ""),
                        "proclitic": clitic,
                    },
                )
        marker = self._marker_index.get(key)
        if marker is not None and marker[1].get("placement") == "word":
            return TokenTranslation(
                token=token, msa=marker[1]["msa"], rule=f"marker.{marker[0]}",
                kind="participial", confidence=MORPHOLOGY_CONFIDENCE,
                detail={"marker": marker[0]},
            )
        return None

    # ------------------------------------------------------------------ #
    # introspection / docs
    # ------------------------------------------------------------------ #
    def generated_forms(self) -> dict[str, str]:
        """Every Iraqi surface form the engine can actually produce.

        Candidates are built by gluing the affix tables onto every stem, then
        each candidate is run back through :meth:`verb_match` and dropped unless
        the engine agrees. That keeps the inventory honest: ``أروحنهم`` is
        never generated because the engine would not parse it back, so the
        coverage report cannot overstate what is supported.
        """
        raw: dict[str, str] = {}
        for verb in self.verbs.values():
            if not verb.applies_to(self.dialect):
                continue
            for surface, msa in verb.forms.items():
                raw.setdefault(surface, msa)
        for verb in self.verbs.values():
            if not verb.applies_to(self.dialect):
                continue
            for tense in self._tenses(verb):
                for stem in verb.all_stems(tense):
                    for prefix, meta in self.person_prefixes.items():
                        for suffix, suffix_meta in self.object_suffixes.items():
                            msa = self._build_form(verb, tense, meta["person"], suffix_meta["msa"])
                            if msa:
                                raw.setdefault(f"{prefix}{stem}{suffix}", msa)
                        for marker, marker_meta in self.plural_suffixes.items():
                            msa = self._build_form(verb, tense, marker_meta["person"], "")
                            if msa:
                                raw.setdefault(f"{prefix}{stem}{marker}", msa)
                    for suffix, suffix_meta in self.subject_suffixes_past.items():
                        if tense != "past":
                            continue
                        msa = self._build_form(verb, tense, suffix_meta["person"], "")
                        if msa:
                            raw.setdefault(f"{stem}{suffix}", msa)
                    if verb.allow_bare:
                        msa = self._build_form(verb, tense, str(verb.allow_bare), "")
                        if msa:
                            raw.setdefault(stem, msa)
        for word, meta in self.pronouns.items():
            raw.setdefault(word, meta["msa"])

        out: dict[str, str] = {}
        for surface, msa in raw.items():
            match = self.verb_match(surface)
            if match is not None and match.msa == msa:
                out[surface] = msa
        return out

    def _build_form(self, verb: VerbEntry, tense: str, person: str, obj: str) -> str | None:
        form = self.form_for(verb, tense, person)
        if form is None:
            return None
        return self.attach_object(verb, form, obj, tense)

    def stats(self) -> dict[str, int]:
        """Counts used by ``docs/coverage.md``."""
        verbs = [v for v in self.verbs.values() if v.applies_to(self.dialect)]
        forms = self.generated_forms()
        return {
            "verbs": len(verbs),
            "verb_roots": len({v.msa_root for v in verbs}),
            "listed_verb_forms": sum(len(v.forms) for v in verbs),
            "generated_verb_forms": len(forms),
            "pronouns": len(self.pronouns),
            "object_suffixes": len(self.object_suffixes),
            "plural_suffixes": len(self.plural_suffixes),
            "proclitics": len(self.proclitics),
            "compositions": len(self.compositions),
            "protected_words": len(self.protected),
            "shared_msa_words": len(self.shared_msa),
            "proper_nouns": len(self.proper_nouns),
            "verb_like": len(self.verb_like),
        }


# --------------------------------------------------------------------------- #
# Lexicon (structured phrase / word entries)
# --------------------------------------------------------------------------- #
def load_lexicon(
    *, dialect: str | None = None, path: Path | None = None
) -> list[LexiconEntry]:
    """Load ``lexicon.json`` and return the entries that apply to ``dialect``."""
    data = _load_json(path or LEXICON_PATH)
    return [
        LexiconEntry(raw)
        for raw in data["entries"]
        if LexiconEntry(raw).applies_to(dialect)
    ]


def lexicon_stats(path: Path | None = None) -> dict[str, int]:
    """Counts per entry type / region, used by ``docs/coverage.md``."""
    data = _load_json(path or LEXICON_PATH)
    entries = [LexiconEntry(raw) for raw in data["entries"]]
    by_type: dict[str, int] = {}
    by_region: dict[str, int] = {}
    for entry in entries:
        by_type[entry.type] = by_type.get(entry.type, 0) + 1
        for region in entry.region:
            by_region[region] = by_region.get(region, 0) + 1
    return {
        "entries": len(entries),
        "multiword": sum(1 for e in entries if e.is_multiword),
        "by_type": by_type,
        "by_region": by_region,
    }
