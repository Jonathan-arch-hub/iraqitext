# IraqiText 🇮🇶

Lightweight NLP for the **Iraqi Arabic dialect** — zero mandatory dependencies,
runs on CPU-only machines (no GPU, no spaCy, no Transformers).

Combines the dictionary-based Iraqi ↔ MSA translation engine with
normalization, tokenization, dialect detection and keyword extraction.

## Features

- 🇮🇶 Iraqi ↔ Modern Standard Arabic translation, in four tiers
- 🧠 A morphology layer: **5167 verb forms generated from 30 roots** (pronouns, object suffixes, plurals, clitics, tense and aspect)
- 🧩 Clause templates: `راح أگلك شصار` → `سأخبرك بما حدث`, `ما نكدر نجي هسه` → `لا نستطيع أن نأتي الآن`, `دا أحچي وياك` → `أتحدث معك الآن`
- 📊 A coverage report that names every word it could **not** translate, plus a frequency log that turns those into a review list
- 🎚️ Per-sentence and per-token confidence, and the tier + rule behind every word
- 🗺️ Optional dialect tags: `baghdad` / `south` / `mosul` / `west` / `general`
- 🧾 A structured lexicon (`lexicon.json`) where an entry carries a type, a region, an example, a source and a confidence
- ✅ A data validator: JSON validity, duplicate keys, no junk, no Egyptian forms, every generated form parses back
- ✨ Arabic/Iraqi text normalization (tatweel & tashkeel removal, letter unification, punctuation cleanup)
- 🔤 Lightweight tokenization with offsets & token kinds
- 🕵️ Iraqi dialect detection (heuristics — no ML models)
- 🔑 Important-word (keyword) extraction with built-in stopwords
- 📎 Handles punctuation glued to words: `شلونك؟` → `كيف حالك؟`
- 🧩 Optional clitic handling: `وشخبارك` → `وما أخبارك`
- ⚡ Fast, pure-Python, no external dependencies

📖 **[What it actually covers →](docs/coverage.md)** — the counts, one example
per rule, and an honest list of what still needs extending.

## Installation

```bash
pip install iraqitext
```

## Quick start

### Unified API

```python
from iraqitext import IraqiText

it = IraqiText()

it.normalize("شــــلونك؟؟؟")      # 'شلونك؟'
it.tokenize("شلونك حبيبي؟")       # ['شلونك', 'حبيبي']
it.detect("شلونك شخبارك؟")        # Detection(dialect='iraqi', score=1.000, ...)
it.to_fusha("شلونك شخبارك؟")      # 'كيف حالك ما أخبارك؟'
it.to_iraqi("كيف حالك")           # 'إشلونك'
it.keywords("شلونك حبيبي شلونك عيني هسه أنا زين")
# ['شلونك', 'حبيبي', 'عيني', 'هسه', 'زين']
```

### Translation (backward-compatible API)

```python
from iraqitext import IraqiTranslator

translator = IraqiTranslator()

translator.to_fusha("شلونك")        # 'كيف حالك'
translator.to_iraqi("كيف حالك")     # 'إشلونك'
translator.to_fusha("هسه أكدر أجي") # 'الآن أستطيع آتي'
```

## Morphology, not just a dictionary

A flat word list cannot handle a conjugated verb, a clitic or a whole clause.
`iraqitext` reads those:

```python
from iraqitext import IraqiTranslator, MorphologyEngine

t = IraqiTranslator()
t.to_fusha("راح أگلك شصار")        # 'سأخبرك بما حدث'
t.to_fusha("ما نكدر نجي هسه")      # 'لا نستطيع أن نأتي الآن'
t.to_fusha("دا أحچي وياك")         # 'أتحدث معك الآن'
t.to_fusha("أبوية")                 # 'أبوية'  ← not guessed at

engine = MorphologyEngine()
engine.verb_match("أدزلك").msa      # 'أرسل لك'
engine.verb_match("تحچين").person    # '2fs'
engine.verb_match("راحتوا").msa      # 'ذهبتم'
len(engine.generated_forms())        # 5167 — all 30 roots × 8 persons × 2 tenses × …
```

Priority is fixed: **full phrase > known conjugation > single word > leave the
word exactly as it was.**  A generated form never overrules a value a person
wrote, and no rule is ever applied to another rule's output.

## Coverage: what the dictionary does *not* know

```python
report = IraqiTranslator().to_fusha("أبوية وشغلة", report=True)

report.output               # 'أبوية وشغلة'
report.untranslated_tokens  # ['أبوية', 'شغلة']   ← the actual gaps
report.confidence           # 1.0   (minimum over every tier that fired)
report.tiers_used           # ['word', ...]
report.tokens[0].tier       # 'word' | 'phrase' | 'structure' | 'morphology' | 'lexicon'
report.tokens[0].rule       # the exact dictionary key or rule id
```

Names, cities, numbers, Latin text, connectors and words already spelled in MSA
are **kept**, not counted as errors:

```python
IraqiTranslator().to_fusha("راحت بغداد 5 apples", report=True).kept_tokens
# ['بغداد', 'apples']
```

Turn a corpus into a review list:

```python
from iraqitext import CoverageLog, IraqiTranslator

log, t = CoverageLog(), IraqiTranslator()
for line in open("my-corpus.txt"):
    log.add_text(line, translator=t)

log.top(10)             # [('أبوية', 312), ('شخانة', 87), ...]
log.review_list()[0]    # {'word': 'أبوية', 'count': 312, 'examples': [...]}
log.save("coverage-log.json")
```

## Dialects

```python
IraqiTranslator(dialect="south").to_fusha("نشف")    # 'انزف'  ← to bleed
IraqiTranslator(dialect="baghdad").to_fusha("نشف")  # 'جف'    ← to dry up
```

A word tagged to a region is only offered to that region, and it always carries
a lower confidence than a general entry.  There is no `egyptian` dialect: this
is an Iraqi → MSA library, and `python -m iraqitext.validate` refuses an entry
that is one.

## Structured data

`dictionary.json` stays a flat `iraqi -> msa` map, because a flat map is the
right shape for 1752 reviewed words.  New material goes in `lexicon.json`, where
an entry is an object:

```json
{
  "iraqi": "نشف",
  "fusha": "انزف",
  "type": "word",
  "region": ["south"],
  "example": "راسه نشف",
  "source": "manual",
  "confidence": 0.7,
  "override": true,
  "note": "southern Iraqi «نشف» means to bleed"
}
```

An entry marked `override: true` deliberately replaces a stored `dictionary.json`
value and runs *before* it; every other entry runs *after*, so it can fill a gap
but can never change a reviewed translation.

Punctuation attached to words is handled by default:

```python
translator.to_fusha("شلونك؟")    # 'كيف حالك؟'
translator.to_fusha("(شلونك)")   # '(كيف حالك)'
translator.to_fusha("شلونك،")    # 'كيف حالك،'
```

## Normalization

```python
from iraqitext import normalize, light_normalize

normalize("شــــلونك؟؟؟")            # 'شلونك؟'      (tatweel + repeated marks)
normalize("كَيْفَ حَالُكْ")          # 'كيف حالك'     (tashkeel)
normalize("أحمد إبراهيم آمنة")      # 'احمد ابراهيم امنة'  (alef unification)
normalize("شلونك ؟")                # 'شلونك؟'       (space before punctuation)
```

All steps are also exposed individually (`remove_tatweel`, `remove_tashkeel`,
`unify_letters`, `normalize_punctuation`, ...) plus `light_normalize`, a
conservative variant that never rewrites letters.

## Detection

```python
from iraqitext import detect, is_iraqi

r = detect("شلونك شخبارك؟")
r.dialect      # 'iraqi'
r.score        # 0..1 Iraqi likelihood
r.confidence   # 0..1
r.details      # raw counts: iraqi_hits, marker_letters, strong_markers, tokens

is_iraqi("شلونك شخبارك؟")           # True
is_arabic("Hello")                  # False
```

## Tokenization

```python
from iraqitext import tokenize, word_tokenize

word_tokenize("شلونك حبيبي؟")       # ['شلونك', 'حبيبي']

for t in tokenize("شلونك؟"):
    print(t.text, t.kind, t.start, t.end)
# شلونك word 0 5
# ؟     punct 5 6
```

## Compatibility & tuning

- `IraqiTranslator(strict_spaces=True)` restores the old 0.1 behavior, where
  dictionary entries only matched when separated by whitespace.
- `IraqiTranslator(clitics=True)` additionally translates words attached to the
  proclitics `و / ف / ب / ك / ل` (experimental): `وشخبارك` → `وما أخبارك`.
- `IraqiTranslator(morphology=False)` runs the 0.2 engine and nothing else — no
  generated forms, no clause templates, no `lexicon.json`.  Every stored
  `dictionary.json` value then comes back exactly as stored, which is what makes
  the flag useful for bisecting a bad translation.
- `IraqiText(translator=IraqiTranslator(...))` lets you swap the engine.

Translation is **non-cascading**: a dictionary rule only ever matches text that
was in the input, so a short entry can never re-translate the output of a
longer one (`ع` used to turn `تُعَالَجُ` into `تُعلئَالَجُ`).

## Development

```bash
pip install -e ".[dev]"
pytest                                  # 136 tests, no network needed
python -m iraqitext.validate            # data checks, no test runner needed
```

## Releasing

`dist/` is a build artifact and is **not** tracked in git — build it fresh, or
you will ship whatever was committed there last:

```bash
rm -rf dist build
python -m build                        # needs the `dev` extra: pip install -e ".[dev]"
python -m twine upload dist/*
```

Bump `version` in `pyproject.toml` **and** `__version__` in
`src/iraqitext/__init__.py` first; PyPI refuses a re-upload of a version that
already exists.

Note that only `iraqitext/` is packaged. `packages.find` is pinned to
`include = ["iraqitext*"]` on purpose — without it, any stray directory under
`src/` (an accidental virtualenv, say) is discovered as a top-level package and
ends up in the exported metadata.

## Roadmap

- [x] Iraqi ↔ MSA translation
- [x] Morphology layer: 30 verb roots expanded into 5167 forms
- [x] Clause templates (future, negation, present marker)
- [x] Coverage report, per-token confidence, frequency log
- [x] Structured lexicon with region tags
- [x] Data validator (`python -m iraqitext.validate`)
- [x] Normalization (tatweel, tashkeel, letters, punctuation)
- [x] Tokenization
- [x] Iraqi dialect detection
- [x] Keyword extraction
- [ ] More than 20,000 Iraqi words
- [ ] More verb roots, and four-letter (umlaut) verbs
- [ ] More southern / Mosul / western entries
- [ ] Noun-phrase agreement (gender, number, case)
- [ ] Grammar rules beyond the three clause templates
- [ ] Iraqi spell checker

## License

MIT License