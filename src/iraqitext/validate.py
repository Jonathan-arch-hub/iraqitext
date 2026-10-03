"""Data validation for the JSON that ships with :mod:`iraqitext`.

The dictionary grew from a handful of lines to 1755 entries and then gained
two more files, so a typo is now a real risk: a malformed JSON file breaks the
import, a duplicate key silently keeps the last value, and two keys that differ
only in a spelling variant the matcher ignores can never both be honoured.
This module checks all of that and returns a list of problems instead of
raising, so it can drive a test *and* be run by hand::

    python -m iraqitext.validate                 # human readable
    python -m iraqitext.validate --json          # machine readable

Every check is a pure function of the data on disk, so a clean run means the
data is clean.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .morphology import DIALECTS, MorphologyEngine, canon, load_lexicon
from .rules import flexible_pattern

BASE_DIR = Path(__file__).parent
DATA_FILES = ("dictionary.json", "morphology.json", "lexicon.json")

#: Values that mean "nothing was translated" and are therefore data bugs.
_SUSPICIOUS_VALUES = frozenset(
    {"", "x", "X", "xx", "xxx", "t", "tbd", "TODO", "-", "?"}
)

#: Spellings the tolerant matcher cannot tell apart **on purpose**.
#:
#: ``flexible_pattern`` folds the alef variants, so ``أبد`` and ``إبد`` compile
#: to one regex.  They are different words (أبد "ever" / إبد "absolutely") and
#: both are worth keeping, so the collision is a documented, tolerated cost of
#: the tolerant matcher rather than a typo.  Only the first spelling in the
#: file is reachable at runtime; the other is what a future stricter matcher
#: would fix.  Each pair is listed here so ``validate`` stays clean and the
#: reason is written down in one place.
ACCEPTED_COLLISIONS: dict[frozenset[str], str] = {
    frozenset({"أبد", "إبد"}): "أبد «ever» vs إبد «absolutely» — different words",
    frozenset({"أنسى", "انسى"}): "أنسى «forget» — alef spelling variant",
    frozenset({"عدة", "عده"}): "عدة «several/tools» vs عنده «at his place»",
    frozenset({"قوطيه", "قوطية"}): "قوطية «a can» — heh/ta-marbuta spelling variant",
}


@dataclass(frozen=True)
class Problem:
    """One thing wrong with the data.

    ``check`` is the machine name, ``where`` points at the file (and the key
    when there is one), ``message`` says what is wrong and ``hint`` says what
    to do about it.
    """

    check: str
    where: str
    message: str
    hint: str = ""

    def __str__(self) -> str:
        text = f"[{self.check}] {self.where}: {self.message}"
        return f"{text}\n    -> {self.hint}" if self.hint else text


@dataclass
class Report:
    """The outcome of a full validation run."""

    problems: list[Problem] = field(default_factory=list)
    checked_files: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems

    def by_check(self, check: str) -> list[Problem]:
        return [p for p in self.problems if p.check == check]

    def add(self, problem: Problem) -> None:
        self.problems.append(problem)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checked_files": list(self.checked_files),
            "stats": self.stats,
            "problems": [
                {
                    "check": p.check,
                    "where": p.where,
                    "message": p.message,
                    "hint": p.hint,
                }
                for p in self.problems
            ],
        }

    def __str__(self) -> str:
        if self.ok:
            return "ok: " + ", ".join(
                f"{k}={v}" for k, v in self.stats.items()
            )
        body = "\n".join(str(p) for p in self.problems)
        return f"{len(self.problems)} problem(s):\n{body}"


# --------------------------------------------------------------------------- #
# JSON loading that can see a duplicate key
# --------------------------------------------------------------------------- #
def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate key: {key!r}")
        seen[key] = value
    return seen


def load_json_checked(path: Path) -> Any:
    """``json.load`` that also refuses duplicate keys."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_reject_duplicate_keys)


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def check_json_valid(report: Report) -> dict[str, Any]:
    """Every data file parses, and no file has a duplicate key."""
    data: dict[str, Any] = {}
    for name in DATA_FILES:
        path = BASE_DIR / name
        report.checked_files.append(name)
        try:
            data[name] = load_json_checked(path)
        except FileNotFoundError:
            report.add(
                Problem(
                    "json.missing",
                    name,
                    "file not found",
                    "the package data list in pyproject.toml must include it",
                )
            )
        except ValueError as exc:  # duplicate key
            report.add(
                Problem("json.duplicate_key", name, str(exc), "remove the earlier copy")
            )
        except json.JSONDecodeError as exc:
            report.add(
                Problem(
                    "json.syntax",
                    f"{name} line {exc.lineno}",
                    exc.msg,
                    "fix the JSON syntax",
                )
            )
    report.stats["json_files"] = len(data)
    return data


def check_dictionary(report: Report, dictionary: dict[str, Any]) -> None:
    """Shape of ``dictionary.json``: a flat ``iraqi -> msa`` map."""
    if not isinstance(dictionary, dict):
        report.add(Problem("dict.shape", "dictionary.json", "not a JSON object"))
        return
    report.stats["dictionary_entries"] = len(dictionary)
    for key, value in dictionary.items():
        where = f"dictionary.json[{key!r}]"
        if not isinstance(key, str) or not isinstance(value, str):
            report.add(Problem("dict.type", where, "key and value must be strings"))
            continue
        if not key.strip():
            report.add(Problem("dict.empty_key", where, "empty key"))
        if value.strip() in _SUSPICIOUS_VALUES:
            report.add(
                Problem(
                    "dict.suspicious_value",
                    where,
                    f"suspicious value {value!r}",
                    "this looks like a placeholder, not a translation",
                )
            )
        if not _ARABIC_RE.search(key):
            report.add(
                Problem(
                    "dict.non_arabic_key",
                    where,
                    f"key {key!r} has no Arabic letter",
                    "keys are Iraqi words or phrases",
                )
            )
        if re.search(r"[\x00-\x1f]", key + value):
            report.add(
                Problem("dict.control_char", where, "control character in the data")
            )
    check_collisions(
        report,
        {k: v for k, v in dictionary.items() if isinstance(v, str)},
        "dict.collision",
        "dictionary.json",
    )


_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿ]")


def check_collisions(
    report: Report,
    mapping: dict[str, str],
    check: str,
    where: str,
) -> list[tuple[str, str, str]]:
    """Keys that the tolerant matcher cannot tell apart.

    ``flexible_pattern`` folds alef and heh variants, so ``اطلع`` and ``أطلع``
    compile to the same regex and only one of the two values can ever be used.
    The pair is reported *and* returned, so the caller can decide which value
    survived.
    """
    buckets: dict[str, list[str]] = defaultdict(list)
    for key in mapping:
        buckets[flexible_pattern(key)].append(key)
    pairs: list[tuple[str, str, str]] = []
    for keys in buckets.values():
        if len(keys) < 2:
            continue
        values = {mapping[key] for key in keys}
        if len(values) < 2:
            continue  # same value: harmless duplicate
        group = frozenset(keys)
        if group in ACCEPTED_COLLISIONS:
            continue  # documented, see ACCEPTED_COLLISIONS
        for key in keys:
            others = [k for k in keys if k != key]
            pairs.append((key, mapping[key], ", ".join(others)))
        report.add(
            Problem(
                check,
                where,
                f"{len(keys)} keys share one pattern but differ in value: "
                + "; ".join(f"{k!r}->{mapping[k]!r}" for k in keys),
                "keep one key, or accept the collision in the test's known set",
            )
        )
    return pairs


def check_lexicon(report: Report, lexicon: Any) -> None:
    """Field-level validation of ``lexicon.json``."""
    if not isinstance(lexicon, dict) or "entries" not in lexicon:
        report.add(Problem("lexicon.shape", "lexicon.json", "no 'entries' list"))
        return
    entries = lexicon["entries"]
    if not isinstance(entries, list):
        report.add(Problem("lexicon.shape", "lexicon.json", "'entries' is not a list"))
        return
    report.stats["lexicon_entries"] = len(entries)

    allowed_types = {
        "word", "phrase", "clitic", "pronoun", "particle", "function", "verb",
        "noun", "adverb", "interjection", "expression",
    }
    required = ("iraqi", "fusha", "type", "region")
    seen_ids: dict[str, int] = {}
    mapping: dict[str, str] = {}
    for index, entry in enumerate(entries):
        where = f"lexicon.json entries[{index}]"
        if not isinstance(entry, dict):
            report.add(Problem("lexicon.type", where, "entry is not an object"))
            continue
        for key in required:
            if key not in entry:
                report.add(
                    Problem("lexicon.missing_field", where, f"missing {key!r}")
                )
        iraqi = entry.get("iraqi")
        fusha = entry.get("fusha")
        if not isinstance(iraqi, str) or not isinstance(fusha, str) or not fusha.strip():
            report.add(Problem("lexicon.type", where, "iraqi/fusha must be non-empty"))
            continue
        where = f"lexicon.json[{iraqi!r}]"
        if entry.get("type") not in allowed_types:
            report.add(
                Problem(
                    "lexicon.type",
                    where,
                    f"unknown type {entry.get('type')!r}",
                    f"use one of {', '.join(sorted(allowed_types))}",
                )
            )
        region = entry.get("region")
        regions = [region] if isinstance(region, str) else (region or [])
        if not regions:
            report.add(Problem("lexicon.region", where, "no region given"))
        for name in regions:
            if name not in DIALECTS:
                report.add(
                    Problem(
                        "lexicon.region",
                        where,
                        f"unknown region {name!r}",
                        f"use one of {', '.join(DIALECTS)}",
                    )
                )
        confidence = entry.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            report.add(
                Problem("lexicon.confidence", where, "confidence must be 0..1")
            )
        entry_id = entry.get("id")
        if entry_id in seen_ids:
            report.add(
                Problem("lexicon.duplicate_id", where, f"id {entry_id!r} reused")
            )
        if isinstance(entry_id, str):
            seen_ids[entry_id] = index
        if not entry.get("source"):
            report.add(
                Problem(
                    "lexicon.source",
                    where,
                    "no source",
                    "every entry says where the rendering came from",
                )
            )
        previous = mapping.get(iraqi)
        if previous is not None and previous != fusha:
            report.add(
                Problem("lexicon.duplicate", where, f"also defined as {previous!r}")
            )
        mapping[iraqi] = fusha

    check_collisions(report, mapping, "lexicon.collision", "lexicon.json")
    check_no_egyptian(report, entries)
    check_no_tashkeel(report, entries)


def check_no_tashkeel(report: Report, entries: list[Any]) -> None:
    """The Iraqi side of a lexicon entry is written without harakat.

    ``normalize()`` strips the marks before a rule ever sees a token, and a
    mark is not a word boundary, so an ``iraqi`` field or an ``example`` that
    carries one could only ever match text carrying the very same mark — a
    silent dead entry.  The ``fusha`` field is exempt: a shadda belongs to the
    MSA spelling itself (``عمّال``), and it is copied straight into the output.

    In ``dictionary.json`` the opposite rule holds and is not checked here:
    there the shadda separates two different words (``سدة`` a dam / ``سدّه`` he
    sealed it), so it has to stay.
    """
    marks = re.compile("[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for field_name in ("iraqi", "example"):
            text = entry.get(field_name)
            if not isinstance(text, str):
                continue
            match = marks.search(text)
            if match:
                report.add(
                    Problem(
                        "lexicon.tashkeel",
                        f"lexicon.json[{entry.get('iraqi')!r}].{field_name}",
                        f"carries the mark {match.group(0)!r}",
                        "write the field without tashkeel: «خدام» not «خدّام»",
                    )
                )


def check_no_egyptian(report: Report, entries: list[Any]) -> None:
    """No Egyptian data: this project is Iraqi -> MSA, nothing else.

    The guard is deliberately mechanical — a word-boundary match against a
    list of markers that are Egyptian and *not* Iraqi (``مش`` is pan-Arabic and
    is not on the list, ``بجد`` is Gulf-and-Iraqi and is on it only because
    Egyptian usage is the famous one).
    """
    from .coverage import EGYPTIAN_ONLY

    pattern = re.compile(
        r"(?<!\S)(" + "|".join(sorted(EGYPTIAN_ONLY)) + r")(?!\S)"
    )
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for field_name in ("iraqi", "fusha"):
            text = entry.get(field_name)
            if not isinstance(text, str):
                continue
            match = pattern.search(text)
            if match:
                report.add(
                    Problem(
                        "lexicon.egyptian",
                        f"lexicon.json[{entry.get('iraqi')!r}].{field_name}",
                        f"Egyptian marker {match.group(1)!r}",
                        "this library is Iraqi -> MSA only",
                    )
                )


def check_morphology(report: Report, morphology: Any) -> None:
    """The morphology data must be self-consistent.

    Every example in the file has to come back out of the engine, and every
    form the engine claims to generate has to be analysable again — otherwise
    ``docs/coverage.md`` would overstate what is supported.
    """
    if not isinstance(morphology, dict):
        report.add(Problem("morphology.shape", "morphology.json", "not an object"))
        return
    try:
        engine = MorphologyEngine()
    except Exception as exc:  # pragma: no cover - broken data
        report.add(Problem("morphology.load", "morphology.json", str(exc)))
        return

    stats = engine.stats()
    report.stats.update(
        {f"morphology_{key}": value for key, value in stats.items()}
    )

    for key, verb in engine.verbs.items():
        example = verb.example or {}
        source, expected = example.get("in"), example.get("out")
        if not source or not expected:
            report.add(
                Problem(
                    "morphology.example",
                    f"morphology.json verbs[{key!r}]",
                    "no worked example",
                    "every verb ships with one iraqi/msa pair",
                )
            )
            continue
        match = engine.verb_match(source)
        got = match.msa if match else None
        if got != expected:
            report.add(
                Problem(
                    "morphology.example",
                    f"morphology.json verbs[{key!r}]",
                    f"{source!r} -> {got!r}, expected {expected!r}",
                    "fix the verb data or the analysis",
                )
            )

    for surface in engine.generated_forms():
        if engine.verb_match(surface) is None:
            report.add(
                Problem(
                    "morphology.roundtrip",
                    "morphology.json",
                    f"generated form {surface!r} does not analyse",
                    "a generated form that cannot be parsed must not be counted",
                )
            )


def check_shadowing(report: Report) -> None:
    """A reviewed entry may never be changed by a generated form.

    The engine is handed a *shadow* set for exactly this, so this check is
    what proves the promise holds: every dictionary and lexicon surface string
    must come back from the morphology engine untouched.
    """
    from .translator import IraqiTranslator

    translator = IraqiTranslator()
    engine = translator.morphology
    assert engine is not None  # morphology is on by default
    shadowed = 0
    for surface in list(translator._fusha_map) + list(translator._lexicon_map):
        if canon(surface) not in engine.shadow:
            report.add(
                Problem(
                    "shadow.missing",
                    repr(surface),
                    "not in the morphology shadow set",
                    "a generated form could overrule this entry",
                )
            )
            continue
        result = engine.translate_token(surface)
        if result is not None and result.msa != surface:
            report.add(
                Problem(
                    "shadow.conflict",
                    repr(surface),
                    f"morphology would translate it to {result.msa!r}",
                    "add the word to protected/shared_msa, or fix the analysis",
                )
            )
        shadowed += 1
    report.stats["shadowed_surfaces"] = shadowed


def check_dialects(report: Report) -> None:
    """A region-tagged entry must only surface for the region it claims.

    This is the "do not present Mosul or southern words as general" rule: with
    a different ``dialect=`` the entry has to disappear, and with
    ``dialect=None`` it has to be reported as region-specific rather than as
    general.
    """
    # A regional entry is offered exactly for the regions it declares, and
    # only then.  This is what keeps Mosul's «كيف هسة» out of a Baghdad
    # translation while «للدوام» (general) works everywhere.
    for dialect in DIALECTS:
        entries = load_lexicon(dialect=dialect)
        bad = [
            (entry.iraqi, list(entry.region))
            for entry in entries
            if "general" not in entry.region and dialect not in entry.region
        ]
        if bad:
            report.add(
                Problem(
                    "dialect.filter",
                    f"dialect={dialect}",
                    f"entries offered that are not tagged for it: {bad}",
                    "load_lexicon must filter by region",
                )
            )
    # ... and an entry declared general must really be available everywhere.
    for entry in load_lexicon(dialect=None):
        if "general" not in entry.region:
            continue
        for dialect in DIALECTS:
            if not any(e.iraqi == entry.iraqi for e in load_lexicon(dialect=dialect)):
                report.add(
                    Problem(
                        "dialect.general",
                        f"lexicon.json[{entry.iraqi!r}]",
                        f"tagged general but missing for dialect={dialect}",
                        "a general entry must apply to every dialect",
                    )
                )
    everything = load_lexicon(dialect=None)
    regional = [e.iraqi for e in everything if "general" not in e.region]
    report.stats["lexicon_region_specific"] = len(regional)
    report.stats["lexicon_offered_baghdad"] = len(load_lexicon(dialect="baghdad"))
    report.stats["accepted_collisions"] = len(ACCEPTED_COLLISIONS)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def validate(*, check_translator: bool = True) -> Report:
    """Run every check and return a :class:`Report`."""
    report = Report()
    data = check_json_valid(report)
    if "dictionary.json" in data:
        check_dictionary(report, data["dictionary.json"])
    if "lexicon.json" in data:
        check_lexicon(report, data["lexicon.json"])
    if "morphology.json" in data:
        check_morphology(report, data["morphology.json"])
    if check_translator:
        check_shadowing(report)
    check_dialects(report)
    return report


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args(argv)
    report = validate()
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(report)
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
