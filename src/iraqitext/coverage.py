"""Coverage reporting: what the engine translated, and what it did not.

Two things live here:

* :class:`TranslationReport` — the per-call answer.  It records which tier
  produced every chunk, gives each chunk a confidence, and — most important
  for improving the data — lists the **untranslated tokens** so the next
  dictionary entry can be written for them.
* :class:`CoverageLog` — the accumulating answer.  Feed it every report (or
  just a text string) and it keeps a frequency count of the words that were
  never translated, so "what should I add to the dictionary next?" has an
  answer instead of a guess.

A token only counts as *untranslated* when nothing at all happened to it.
Names, cities, numbers, connectors, shared-MSA words and anything already in
MSA are **kept** and reported as ``kept``, never as errors — otherwise the
report would drown in false alarms.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
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
from .translator import TIERS, TIER_CONFIDENCE

BASE_DIR = Path(__file__).parent

#: Words that are *kept as they are* and are not mistakes: the script of the
#: output is Arabic either way, and they carry no dialect information.
KEEP_AS_IS = frozenset(
    """
    و ف ب ل ك أو ثم لكن إن إما أي إذا كما حتى بل بن بلى لم لما لن لو لا
    في على إلى عن من مع هذا هذه ذلك تلك الذي التي الذين كان كانت يكون تكون
    قد لقد حتى بين بعد قبل كل بعض جدا أيضا هنا هناك الآن اليوم غدا أمس
    ليلا صباحا مساءا سنة شهر يوم ساعة دقيقة رجل امرأة ولد بنت بيت سيارة
    مال عمل حياة دنيا آخرة نفس روح قلب عين يد رأس وجه صوت كلام أمر حال
    وقت مكان قسم جزء نصف كلمة جملة كتاب كتب مدرسة مدارس جامعة شركة دائرة
    مستشفى سيارة باب دار مكتب هاتف كمبيوتر انتر نت ماركت شمس قمر نهر بحر
    جبل مدينة أرض شجرة نار ماء هوا ثوب خبز لحم سمك
    """.split()
)

#: Pure numbers / latin / symbols.  Kept, never reported as a gap.
_NUMBER_RE = re.compile(r"^[\d٠-٩.,:/\-]+$")
_LATIN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9'’.\- ]*$")

#: Egyptian-only markers.  This project is Iraqi -> MSA; an Egyptian entry is
#: a data bug, so `iraqitext.validate` refuses it and a test pins the list.
#:
#: The list is deliberately short and every word in it is one an Iraqi reader
#: would *not* use.  Pan-Arabic colloquial words are left out on purpose:
#: ``مش``، ``يا``، ``عم``، ``ليه``، ``كمان``، ``كويس``، ``معاك``، ``خالص``،
#: ``بجد``، ``سيدي`` and ``وحش`` all occur in Iraqi speech, and banning them
#: would ban Iraqi.  What is listed instead is the set that has no Iraqi use at
#: all, mostly Cairo forms: ``عايز`` (Iraqi *ريد / أريد*), ``دلوقتي`` (Iraqi
#: *هسه*), ``علشان`` (Iraqi *بسبب / علشون*), ``بتاع`` (Iraqi *مال*), ``أوي``
#: (Iraqi *كتير*), ``إزيك`` (Iraqi *شلونك*).
EGYPTIAN_ONLY = frozenset(
    """
    عايز عايزة عايزين عوزين كده دلوقتي دلوقت دلوقتة
    إزيك ازيك بتاع بتاعي بتاعك بتوع بتوعنا
    مفيش خيصة باظ برضه بالظبط أوي كده
    """.split()
)


#: ``ARABIC_LETTERS`` is a regex character-class *body*, so it has to be
#: compiled before it can answer "does this token contain an Arabic letter?".
_HAS_ARABIC_RE = re.compile("[" + ARABIC_LETTERS + "]")


def _is_kept(token: str) -> bool:
    """True when a token is *supposed* to survive translation untouched."""
    text = token.strip()
    if not text:
        return True
    if _NUMBER_RE.match(text) or _LATIN_RE.match(text):
        return True
    if not _HAS_ARABIC_RE.search(text):
        return True  # punctuation and symbols
    # A one-letter token is a clitic or a particle (و، ف، ب، ل، ك، م، ه)،
    # never a word that is missing from the data.
    if len(text) == 1:
        return True
    return text in KEEP_AS_IS


def _is_proper_noun(token: str, engine: MorphologyEngine | None) -> bool:
    return engine is not None and canon(token) in engine.proper_noun_canon


def _is_already_msa(token: str, engine: MorphologyEngine | None) -> bool:
    """True when the word is already standard Arabic and needs no entry.

    Iraqi and MSA share most of their vocabulary, and the engine has an
    explicit list of the words that are the same in both (``كتابي``,
    ``مدرسته`` …).  Reporting one of those as a gap would fill the review list
    with noise, so a word on the list is *kept*, not *untranslated*.
    """
    return engine is not None and canon(token) in engine.shared_msa_canon


@dataclass
class TranslationReport:
    """What one :meth:`IraqiTranslator.to_fusha` call did.

    Attributes
    ----------
    source / output:
        The input and the translated text.
    hits:
        One record per rule that fired, in the order they were applied.  Each
        has ``rule``, ``tier``, ``iraqi``, ``fusha`` and sometimes ``detail``.
    untranslated_tokens:
        The words that came out exactly as they went in **and** were not
        supposed to — the review list.  ``["أبوية", "شغلة"]`` is the shape the
        caller should expect.
    kept_tokens:
        Words that came out unchanged on purpose (names, numbers, connectors,
        shared-MSA words, English).
    confidence:
        The lowest confidence among the rules that fired, i.e. how much of the
        sentence rests on generated forms.
    tokens:
        One :class:`TokenReport` per output token: the tier that produced it,
        the rule name and the confidence for that token alone.
    """

    source: str
    output: str
    hits: list[dict[str, Any]] = field(default_factory=list)
    untranslated_tokens: list[str] = field(default_factory=list)
    kept_tokens: list[str] = field(default_factory=list)
    confidence: float = 1.0
    dialect: str | None = None
    tokens: list["TokenReport"] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    @property
    def tiers_used(self) -> list[str]:
        """Distinct tiers that fired, in priority order."""
        used = {hit.get("tier") for hit in self.hits}
        return [t for t in TIERS if t in used] + sorted(
            t for t in used if t not in TIERS
        )

    @property
    def rules(self) -> list[str]:
        return [hit["rule"] for hit in self.hits if "rule" in hit]

    @property
    def is_fully_translated(self) -> bool:
        return not self.untranslated_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "output": self.output,
            "tiers_used": self.tiers_used,
            "rules": self.rules,
            "confidence": round(self.confidence, 3),
            "untranslated_tokens": list(self.untranslated_tokens),
            "kept_tokens": list(self.kept_tokens),
            "tokens": [token.to_dict() for token in self.tokens],
        }

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class TokenReport:
    """What happened to one output token.

    ``tier`` is ``phrase`` / ``structure`` / ``morphology`` / ``lexicon`` /
    ``word`` / ``unchanged``, and ``confidence`` is that tier's score — a
    reviewed phrase is the most trustworthy, a generated verb form the least.
    ``rule`` is the dictionary key or rule id, so a bad translation can be
    traced back to the data that caused it.
    """

    text: str
    tier: str
    confidence: float
    rule: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tier": self.tier,
            "confidence": round(self.confidence, 3),
            "rule": self.rule,
            "source": self.source,
        }


#: Matches the ``\x00<n>\x00`` sentinels the pipeline parks replacements behind.
_SENTINEL_RE = re.compile("\x00([0-9]+)\x00")

#: Only ``و`` is stripped from a reported word.
#:
#: The translator's clitic mode knows about و ف ب ك ل, but a *report* may not:
#: ف ب ك ل start too many real words (فاطمة، بس، كل، كتب) and stripping them
#: would report «فاطمة» as the gap «اطمة».  ``و`` is the one that is written
#: attached to a following word constantly and never starts one.
_WAWA = "و"


def _core(token: str) -> str:
    """Strip a leading written conjunction, keeping the spelling as written.

    ``وشيگلة`` is one token to the tokenizer but the *gap* is ``شيگلة``; the
    review list is more useful with the conjunction removed.  The spelling is
    left alone because the reader has to be able to grep the list back out of
    the corpus it came from — ``أبوية`` must not come back as ``ابويه``.
    """
    core = token
    while len(core) > 2 and core[0] == _WAWA:
        core = core[1:]
    return core


def residual_words(staged: str) -> set[str]:
    """Canonicalised words still standing in ``staged`` next to sentinels.

    ``staged`` is the text just before the pipeline restores its replacements:
    a translated chunk is a ``\\x00<n>\\x00`` sentinel and an untouched word is
    still there.  This is what makes the untranslated list exact instead of a
    guess based on comparing two finished strings.
    """
    keep: list[str] = []
    position = 0
    for match in _SENTINEL_RE.finditer(staged):
        keep.append(staged[position : match.start()])
        position = match.end()
    keep.append(staged[position:])
    words: set[str] = set()
    for chunk in keep:
        for token in tokenize(chunk):
            if token.is_word and token.text.strip():
                # Canonicalised: «أبوية» and «ابويه» are the same gap.
                words.add(canon(_core(token.text)))
    return words


def build_report(
    text: str,
    output: str,
    hits: Sequence[dict[str, Any]],
    *,
    staged: str | None = None,
    engine: MorphologyEngine | None = None,
    dialect: str | None = None,
) -> TranslationReport:
    """Assemble a :class:`TranslationReport` from the pieces ``to_fusha`` has.

    ``staged`` is the text just before the sentinels are restored (see
    :func:`residual_words`).  Without it the report falls back to comparing the
    two finished strings token by token, which is good enough for the legacy
    path but cannot tell "unchanged" from "reordered".
    """
    report = TranslationReport(
        source=text, output=output, hits=list(hits), dialect=dialect
    )
    confidence = 1.0
    for hit in hits:
        confidence = min(confidence, _tier_confidence(hit))
    report.confidence = confidence
    report.tokens = _token_reports(output, hits)

    if staged is None:
        report.untranslated_tokens, report.kept_tokens = _string_diff(text, output)
        return report

    remaining = residual_words(staged)
    for token in tokenize(text):
        if not token.is_word:
            continue
        word = token.text
        core = _core(word)
        if canon(core) not in remaining:
            continue  # a rule consumed it
        if (
            _is_kept(word)
            or _is_proper_noun(word, engine)
            or _is_already_msa(word, engine)
        ):
            report.kept_tokens.append(word)
        else:
            # Report the word without its clitic: the entry that is missing is
            # «شغلة», not «وشغلة».
            report.untranslated_tokens.append(core)
    return report


def _string_diff(source: str, output: str) -> tuple[list[str], list[str]]:
    """Fallback for the legacy (no-morphology) path: compare token by token.

    If a rule inserted or removed a word the two lists get out of step, so a
    length mismatch means "assume it changed" — the report then under-reports
    rather than blaming a word that was in fact translated.
    """
    src = [t.text for t in tokenize(source) if t.is_word]
    out = [t.text for t in tokenize(output) if t.is_word]
    unchanged: list[str] = []
    if len(src) == len(out):
        unchanged = [a for a, b in zip(src, out) if a == b]
    untranslated = [w for w in unchanged if not _is_kept(w)]
    kept = [w for w in unchanged if _is_kept(w)]
    return untranslated, kept


def _tier_confidence(hit: dict[str, Any]) -> float:
    if "confidence" in hit:
        return float(hit["confidence"])
    return TIER_CONFIDENCE.get(hit.get("tier", "word"), 0.8)


def _token_reports(output: str, hits: Sequence[dict[str, Any]]) -> list[TokenReport]:
    """Split the output into tokens and say who produced each one.

    A hit records the Iraqi text it replaced and the MSA text it produced, so
    the report can name the tier and the rule for each exact output word
    instead of only for the sentence.  Hits are consumed left to right: a
    replacement that produced three words (بما حدث) claims all three, and the
    scan continues after them, so repeated words are never double-counted.
    """
    words = [t for t in tokenize(output) if t.is_word]
    keys = [canon(t.text) for t in words]
    usable = [hit for hit in hits if hit.get("fusha") and words]
    reports: list[TokenReport] = []

    def _unchanged(token) -> TokenReport:
        return TokenReport(
            text=token.text,
            tier="unchanged",
            confidence=TIER_CONFIDENCE["unchanged"],
        )

    index = 0
    while index < len(words):
        owner = None
        width = 1
        for hit in usable:
            produced = [canon(w.text) for w in tokenize(hit["fusha"]) if w.is_word]
            if not produced:
                continue
            if keys[index : index + len(produced)] == produced:
                owner = hit
                width = len(produced)
                break
        if owner is None:
            reports.append(_unchanged(words[index]))
            index += 1
            continue
        confidence = _tier_confidence(owner)
        for token in words[index : index + width]:
            reports.append(
                TokenReport(
                    text=token.text,
                    tier=owner.get("tier", "word"),
                    confidence=confidence,
                    rule=owner.get("rule", ""),
                    source=owner.get("iraqi", ""),
                )
            )
        index += width
    return reports


# --------------------------------------------------------------------------- #
# The accumulating review list
# --------------------------------------------------------------------------- #
class CoverageLog:
    """Frequency log of everything the engine could not translate.

    Usage::

        log = CoverageLog()
        log.add_text("شنو رأيك نروح بكرة")      # a whole sentence
        log.add_report(translator.to_fusha(text, report=True))
        print(log.top(20))                      # what to add next
        log.save("coverage_log.json")            # next run merges into it

    ``minimum`` drops words seen fewer than N times, so the top of the list
    stays the words that actually block people.
    """

    def __init__(self, *, minimum: int = 1) -> None:
        self.counts: Counter[str] = Counter()
        self.samples: dict[str, list[str]] = {}
        self.minimum = minimum
        self.sentences = 0

    # -- recording ----------------------------------------------------- #
    def add_text(self, text: str, *, translator: Any = None) -> "CoverageLog":
        """Translate ``text`` and record whatever was left untranslated."""
        from .translator import IraqiTranslator

        translator = translator or IraqiTranslator()
        return self.add_report(translator.to_fusha(text, report=True))

    def add(self, source: str, output: str) -> "CoverageLog":
        """Record the gaps of a *finished* translation (no engine needed)."""
        untranslated, _kept = _string_diff(source, output)
        return self.add_words(untranslated, source)

    def add_words(self, words: Iterable[str], source: str = "") -> "CoverageLog":
        for word in words:
            self.counts[word] += 1
            self.sentences += 1
            self.samples.setdefault(word, [])
            if source and source not in self.samples[word] and len(self.samples[word]) < 3:
                self.samples[word].append(source)
        return self

    def add_report(self, report: TranslationReport) -> "CoverageLog":
        for word in report.untranslated_tokens:
            self.counts[word] += 1
            self.sentences += 1
            self.samples.setdefault(word, [])
            if report.source not in self.samples[word] and len(self.samples[word]) < 3:
                self.samples[word].append(report.source)
        return self

    # -- reading -------------------------------------------------------- #
    def top(self, limit: int = 20) -> list[tuple[str, int]]:
        """The most frequent untranslated words, most frequent first."""
        items = [(w, n) for w, n in self.counts.items() if n >= self.minimum]
        return sorted(items, key=lambda item: (-item[1], item[0]))[:limit]

    def review_list(self, limit: int = 20) -> list[dict[str, Any]]:
        """``top`` plus an example sentence — the list you work through."""
        return [
            {"word": word, "count": count, "examples": self.samples.get(word, [])}
            for word, count in self.top(limit)
        ]

    def __len__(self) -> int:
        return len(self.counts)

    def __contains__(self, word: str) -> bool:
        return word in self.counts

    def clear(self) -> None:
        self.counts.clear()
        self.samples.clear()
        self.sentences = 0

    # -- persistence ---------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "sentences": self.sentences,
            "untranslated": self.review_list(limit=len(self.counts)),
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return target

    def load(self, path: str | Path) -> "CoverageLog":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for item in data.get("untranslated", []):
            word = item["word"]
            self.counts[word] = int(item.get("count", 1))
            self.samples[word] = list(item.get("examples", []))
        self.sentences = int(data.get("sentences", self.sentences))
        return self


# --------------------------------------------------------------------------- #
# The numbers behind docs/coverage.md
# --------------------------------------------------------------------------- #
def coverage_stats(translator: Any = None) -> dict[str, Any]:
    """Everything ``docs/coverage.md`` prints, as plain data."""
    from .translator import IraqiTranslator

    t = translator or IraqiTranslator()
    engine: MorphologyEngine | None = t.morphology
    dictionary = t._fusha_map
    lexicon = load_lexicon(dialect=t.dialect)

    stats: dict[str, Any] = {
        "dialect": t.dialect or "all",
        "dictionary_entries": len(dictionary),
        "dictionary_phrases": sum(1 for k in dictionary if " " in k),
        "dictionary_words": sum(1 for k in dictionary if " " not in k),
        "lexicon_entries": len(lexicon),
        "lexicon_overrides": sum(1 for entry in lexicon if entry.override),
        "lexicon_region_specific": sum(
            1 for entry in lexicon if "general" not in entry.region
        ),
        "lexicon_with_example": sum(1 for entry in lexicon if entry.example),
    }
    if engine is None:
        stats["morphology_enabled"] = False
        return stats

    stats["morphology_enabled"] = True
    stats.update(engine.stats())
    stats["generated_forms_unique"] = len(engine.generated_forms())
    stats["shadowed_forms"] = sum(
        1 for surface in engine.generated_forms() if surface in set(dictionary)
    )
    stats["shadowed_surfaces"] = len(engine.shadow)
    return stats


def example_table(engine: MorphologyEngine, limit: int = 0) -> list[dict[str, str]]:
    """One worked example per verb, for the documentation."""
    rows: list[dict[str, str]] = []
    for key, verb in engine.verbs.items():
        example = verb.example or {}
        if not example:
            continue
        rows.append(
            {
                "iraqi_stem": verb.iraqi_stems.get("present", key),
                "in": example.get("in", ""),
                "out": example.get("out", ""),
                "note": verb.note,
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows


def composition_examples(engine: MorphologyEngine) -> list[dict[str, str]]:
    """One worked example per clause-level rule."""
    return [
        {
            "id": comp.get("id", ""),
            "in": (comp.get("example") or {}).get("in", ""),
            "out": (comp.get("example") or {}).get("out", ""),
            "note": comp.get("note", ""),
        }
        for comp in engine.compositions
    ]
