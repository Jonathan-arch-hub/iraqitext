# What iraqitext actually covers

Every number and every example on this page is produced by the code, not typed
in by hand.  To regenerate it and to check the page against the shipped data:

```bash
python -m iraqitext.validate          # data checks, prints the same counts
python -c "import json, iraqitext; print(json.dumps(iraqitext.coverage_stats(), ensure_ascii=False, indent=2))"
```

If a number here disagrees with `coverage_stats()`, this file is stale.

## The numbers

| What | Count |
| --- | ---: |
| `dictionary.json` entries (reviewed, hand-written) | 1752 |
| …of which multiword phrases | 294 |
| …of which single words | 1458 |
| `lexicon.json` structured entries | 35 |
| …marked `override` (deliberately change a stored value) | 5 |
| …tagged to one region only | 4 |
| …carrying a worked example | 35 |
| Verb roots in `morphology.json` | 30 |
| Verb forms written out by hand | 36 |
| Verb forms the engine **generates** from the paradigms | 5167 |
| Distinct generated surface forms | 5167 |
| Generated forms already covered by a reviewed entry (shadowed) | 21 |
| Pronoun spellings | 13 |
| Object suffixes (`ـني، ـك، ـچ، ـه، ـها، ـهم، ـنا` …) | 27 |
| Plural / gender suffixes | 3 |
| Proclitics (`و، ف، ك، ل`) | 4 |
| Clause templates (compositions) | 3 |
| Words the engine refuses to touch (`protected`) | 80 |
| Words spelled the same in Iraqi and MSA (`shared_msa`) | 90 |
| Place names kept as written (`proper_nouns`) | 30 |
| Words allowed to head a clause (`verb_like`) | 22 |
| Surface strings a reviewed entry already decides (the shadow set) | 1759 |

The headline number is **5167 generated verb forms from 30 roots**.  Each root
declares its Iraqi stems and its MSA stems; the engine crosses them with eight
persons × two tenses × 27 object suffixes × 4 proclitics × 3 plural markers and
writes the result out.  `validate.py` parses every one of those forms back and
fails if a single one cannot be read, so the inventory cannot claim coverage
the engine does not have.

## The pipeline

Four tiers run in this order, and the first one that can read a span wins:

1. **structure** — a clause template (`راح أگلك`, `ما نكدر نجي`, `دا أحچي وياك`)
2. **phrase** — a multiword `dictionary.json` entry
3. **morphology** — a known conjugation, pronoun, marker or clitic
4. **word** — a single `dictionary.json` / `lexicon.json` entry

Three rules hold across all of them:

* **No cascade.** Every replacement is parked behind a sentinel while the
  remaining rules run, so a rule can only ever match characters that were in the
  input.  `غير` → `سوى` and `ثقيل دم` → `غير مرح` cannot turn into `سوى مرح`.
* **A generated form never overrules a person.** All 1759 reviewed surface
  strings are handed to the engine as a *shadow* set; a verb form that a human
  has already decided is left alone.
* **Punctuation survives.** A replacement only drops a trailing mark when the
  input already has the very same mark right there, so `وين رايح؟` gives
  `إلى أين تذهب؟` and not `إلى أين تذهب؟؟`.

## One example per rule

### Tier 1 — clause templates (`morphology.json` → `compositions`)

| Rule id | Iraqi | MSA |
| --- | --- | --- |
| `future.sara` | `راح أروح للدوام` | `سأذهب إلى العمل` |
| `negation.mahood` | `ما نكدر نجي هسه` | `لا نستطيع أن نأتي الآن` |
| `present.da` | `دا أحچي وياك` | `أتحدث معك الآن` |

A template only fires when the span really is the clause: `دا` alone stays the
dictionary word `يقوم بـ`, and `ليش ما جيت` is still the stored phrase
`لماذا لم تأت`.

### Tier 2 and 4 — reviewed entries

| Tier | Iraqi | MSA | Note |
| --- | --- | --- | --- |
| phrase | `مو كتلك` | `ألم أحذرك؟` | the entry carries its own `؟` |
| word | `شلونك` | `كيف حالك` | |
| lexicon (clitic) | `للدوام` | `إلى العمل` | `ل` + definite article handled as one entry |
| lexicon (override) | `تأخر` | `تتأخر` | corrects a stored value, on purpose |

### Tier 3 — morphology

| Rule | Iraqi | MSA |
| --- | --- | --- |
| pronoun | `إنتي` | `أنتِ` |
| pronoun behind `و` | `وهيه` | `وهي` |
| negation (existential) | `ماكو` | `لا يوجد` |
| present participle | `جاي` | `قادم` |
| object suffix | `أگلك` | `أخبرك` |
| object + proclitic | `أدزلك` | `أرسل لك` |
| feminine plural | `نگدرين` | `تستطيعين` |
| 2nd person feminine | `تحچين` | `تتحدثين` |
| past, 3rd person plural | `راحتوا` | `ذهبتم` |
| past, 3rd person feminine | `رحت` | `ذهبت` |
| nothing applies | `أبوية` | `أبوية` (left exactly as written) |

### Pronoun table

| Iraqi | MSA | Person |
| --- | --- | --- |
| `أني` / `اني` | `أنا` | 1sg |
| `إنت` | `أنت` | 2ms |
| `إنتي` | `أنتِ` | 2fs |
| `انتي` | `أنت` | 2ms (the Iraqi spelling is gender-neutral) |
| `إنتو` | `أنتم` | 2pl |
| `إنتن` | `أنتن` | 2pl |
| `إحنا` / `احنا` | `نحن` | 1pl |
| `هوه` / `هه` | `هو` | 3ms |
| `هيه` | `هي` | 3fs |
| `هم` | `هم` | 3pl |

### Object suffixes

27 forms are read, not only the seven the task named: `ـني، ـك، ـچ، ـه، ـها، ـهم،
ـنا` and, on top of those, the doubled and prepositional ones Iraqi actually
writes — `ـلni، ـلك، ـلچ، ـله، ـلها، ـلهم، ـنك، ـنها، ـنهم، ون` …  Each one
re-attaches in MSA position, so `أدزلك` is `أرسل لك`, not `أرسلَ لك` in the
wrong place.

## Dialect tags

`DIALECTS = ("baghdad", "south", "mosul", "west", "general")`.

There is **no `egyptian`**: this library is Iraqi → MSA, and a Cairo form is a
data bug.  `validate.py` refuses any entry in `EGYPTIAN_ONLY` and a test pins
the marker list so it cannot grow by accident.

The four region-tagged entries, and what happens when the wrong dialect asks for
them:

| Iraqi | MSA | Region | Confidence | Asking the wrong region |
| --- | --- | --- | ---: | --- |
| `نشف` (to bleed) | `انزف` | south | 0.7 | `baghdad` gives the stored `جف` |
| `كيف هسة` | `كيف الآن` | mosul | 0.7 | not offered outside Mosul |
| `خدام` (labourer) | `عمّال` | baghdad | 0.6 | not offered elsewhere |
| `كنور` (lamp) | `مصباح` | west, baghdad | 0.7 | not offered elsewhere |

Every regional entry carries a confidence **below** the general ones, so a
region-specific reading is never presented as certain.  `dialect=None` (the
default) offers everything.

## Coverage report and frequency log

`to_fusha(text, report=True)` returns a `TranslationReport`:

```python
>>> from iraqitext import IraqiTranslator
>>> report = IraqiTranslator().to_fusha("أبوية وشغلة", report=True)
>>> report.output
'أبوية وشغلة'
>>> report.untranslated_tokens
['أبوية', 'شغلة']
>>> report.confidence
1.0
```

Three buckets, and nothing else is ever counted as a failure:

* **translated** — a tier produced it; `report.tokens[i]` names the tier, the
  rule, the Iraqi source and a confidence
* **kept** — a place name, a number, Latin text, a connector, a word already
  spelled in MSA (`كتابي`), or a single letter
* **untranslated** — an Arabic word no rule could read

`CoverageLog` counts the third bucket over a corpus and turns it into a review
list:

```python
>>> from iraqitext import CoverageLog, IraqiTranslator
>>> log, t = CoverageLog(), IraqiTranslator()
>>> for text in ("أبوية وشخانة", "عندي شخانة وأبوية", "أبوية"):
...     log.add_text(text, translator=t)
>>> log.top(2)
[('أبوية', 3), ('شخانة', 2)]
>>> log.review_list()[0]["word"], log.review_list()[0]["count"]
('أبوية', 3)
>>> log.save("coverage-log.json")            # plain JSON, safe to commit
```

A written conjunction is stripped before counting, so `وشخانة` is logged as the
gap `شخانة`.  `ف` `ب` `ك` `ل` are **not** stripped: they start too many real
words, and stripping them would report `فاطمة` as the gap `اطمة`.

## Confidence

| Tier | Confidence | Why |
| --- | ---: | --- |
| `phrase` | 0.95 | a human wrote and reviewed this exact string |
| `word` / `lexicon` | 0.90 | a human wrote this word; the entry says so |
| `structure` | 0.85 | a template plus a generated verb form |
| `morphology` | 0.75 | fully generated from a paradigm |
| `unchanged` | 1.00 | correct only in the sense that nothing was claimed |

A sentence's confidence is the **minimum** over the tiers that fired, so one
generated form pulls the whole sentence down.  Regional entries sit below their
tier's default.

## What still needs extending

This is the honest list.

* **Coverage is 1752 reviewed words.** Iraqi has hundreds of thousands.  Every
  word the engine cannot read is reported, not guessed — that is the design,
  but it is also the limit.
* **30 verb roots.** Common Iraqi verbs are still missing, and for a missing
  root the engine cannot conjugate at all.  Adding one is three lines of JSON
  (`iraqi_stems`, `msa_stems`, `example`) and the paradigm is generated for
  free.
* **Dialect tags are thin.** 4 region-tagged entries out of 35.  South and Mosul
  are barely represented, and nothing is tagged for a region *and* verified —
  every regional entry is a judgement call, recorded with a lower confidence.
* **The negation template is one shape.** `ما` + verb is handled, `مو` is
  handled as a word, but `ما`, `مو` and `ماكو` inside longer clauses
  (`ما كنت أعرف`, `مو كلش`, `ما أريدك`) are still largely phrase-level.
* **No syntax.** Word order, gender agreement across a whole noun phrase, and
  case are not modelled.  `إنتي وين رايحة` is translated word by word, so the
  Iraqi order survives into the output even when MSA would not use it.
* **No Egyptian, deliberately.** If that ever changes, `EGYPTIAN_ONLY` and
  `DIALECTS` have to change together, and the guard in `validate.py` with them.
* **Four documented matcher collisions** (`أبد/إبد`, `أنسى/انسى`, `عدة/عده`,
  `قوطية/قوطيه`) are accepted, not fixed.  The tolerant matcher folds alef and
  `ه/ة`, so both spellings of a pair share one pattern.

## Checking the data

```bash
python -m iraqitext.validate          # or: python -m src.iraqitext.validate
python -m iraqitext.validate --json   # machine-readable
```

It checks that every JSON file parses and has no duplicate keys, that no
dictionary value is junk, that lexicon entries carry every documented field,
that no entry is Egyptian, that no Iraqi entry carries tashkeel, that every
verb example and all 5167 generated forms parse back to the MSA they claim, that
the shadow set is what it should be, and that every dialect tag is a known
one.  A failure prints the file, the key and what to do about it.
