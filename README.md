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