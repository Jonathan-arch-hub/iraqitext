# IraqiText 🇮🇶

Lightweight NLP for the **Iraqi Arabic dialect** — zero mandatory dependencies,
runs on CPU-only machines (no GPU, no spaCy, no Transformers).

Combines the dictionary-based Iraqi ↔ MSA translation engine with
normalization, tokenization, dialect detection and keyword extraction.

## Features

- 🇮🇶 Iraqi ↔ Modern Standard Arabic translation (dictionary + tolerant patterns)
- ✨ Arabic/Iraqi text normalization (tatweel & tashkeel removal, letter unification, punctuation cleanup)
- 🔤 Lightweight tokenization with offsets & token kinds
- 🕵️ Iraqi dialect detection (heuristics — no ML models)
- 🔑 Important-word (keyword) extraction with built-in stopwords
- 📎 Handles punctuation glued to words: `شلونك؟` → `كيف حالك؟`
- 🧩 Optional clitic handling: `وشخبارك` → `وما أخبارك`
- ⚡ Fast, pure-Python, no external dependencies

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
- `IraqiText(translator=IraqiTranslator(...))` lets you swap the engine.

Translation is **non-cascading**: a dictionary rule only ever matches text that
was in the input, so a short entry can never re-translate the output of a
longer one (`ع` used to turn `تُعَالَجُ` into `تُعلئَالَجُ`).

## Development

```bash
pip install -e ".[dev]"
pytest                                  # 70 tests, no network needed
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
- [x] Normalization (tatweel, tashkeel, letters, punctuation)
- [x] Tokenization
- [x] Iraqi dialect detection
- [x] Keyword extraction
- [ ] More than 20,000 Iraqi words
- [ ] Grammar rules
- [ ] Iraqi spell checker
- [ ] Southern / Mosul / Baghdadi dialect support

## License

MIT License