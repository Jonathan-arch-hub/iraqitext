"""End-to-end tests for iraqitext 0.2.

Covers backward compatibility of the 0.1 API (IraqiTranslator) plus the new
NLP surface: normalize, tokenize, detect and the unified IraqiText API.
"""
import json
import re
from pathlib import Path

import pytest

from iraqitext import (
    DIALECTS,
    MORPHOLOGY_CONFIDENCE,
    CoverageLog,
    Detection,
    IraqiText,
    IraqiTranslator,
    MorphologyEngine,
    TranslationReport,
    coverage_stats,
    detect,
    is_arabic,
    is_iraqi,
    light_normalize,
    normalize,
    tokenize,
    word_tokenize,
)
from iraqitext.translator import _apply
from iraqitext.validate import ACCEPTED_COLLISIONS

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "src" / "iraqitext"
DICTIONARY_PATH = PACKAGE_DIR / "dictionary.json"


# --------------------------------------------------------------------------- #
# Backward compatibility: the 0.1 API must keep working
# --------------------------------------------------------------------------- #
class TestTranslatorCompat:
    def test_to_fusha_basic(self):
        assert IraqiTranslator().to_fusha("شلونك") == "كيف حالك"

    def test_to_iraqi_basic(self):
        # both شلونك and إشلونك map to "كيف حالك"; the longer entry wins
        assert IraqiTranslator().to_iraqi("كيف حالك") in ("شلونك", "إشلونك")

    def test_phrase_translation(self):
        assert IraqiTranslator().to_fusha("هسه أكدر أجي") == "الآن أستطيع آتي"

    def test_full_sentence(self):
        assert IraqiTranslator().to_fusha("شلونك شخبارك؟") == "كيف حالك ما أخبارك؟"

    def test_compound_iraqi_phrases_take_priority_over_single_words(self):
        """Long Iraqi expressions must be translated as complete phrases."""
        t = IraqiTranslator()
        assert t.to_fusha("مو كتلك") == "ألم أحذرك؟"
        assert t.to_fusha("ماكو فايدة") == "لا فائدة"
        assert t.to_fusha("دير بالك على روحك") == "اعتن بنفسك"
        assert t.to_fusha("خلي أروح") == "دعني أذهب"
        assert t.to_fusha("بلا لف ودوران") == "بلا مراوغة"

    def test_long_covered_sentence_has_no_untranslated_iraqi_terms(self):
        t = IraqiTranslator()
        assert t.to_fusha("هلا وغلا، شاسمك؟ إذا تريد شي، خلي أساعدك؛ ماكو مشكلة.") == (
            "أهلاً وسهلاً، ما اسمك؟ إن أردت شيئاً، دعني أساعدك؛ لا مشكلة."
        )

    def test_strict_spaces_keeps_old_behavior(self):
        old = IraqiTranslator(strict_spaces=True)
        assert old.to_fusha("شلونك") == "كيف حالك"
        # old boundaries do NOT match words glued to punctuation
        assert old.to_fusha("شلونك؟") == "شلونك؟"
        assert old.to_fusha("(شلونك)") == "(شلونك)"

    def test_new_boundaries_handle_punctuation(self):
        t = IraqiTranslator()
        assert t.to_fusha("شلونك؟") == "كيف حالك؟"
        assert t.to_fusha("شلونك!") == "كيف حالك!"
        assert t.to_fusha("(شلونك)") == "(كيف حالك)"
        assert t.to_fusha("شلونك،") == "كيف حالك،"
        assert t.to_fusha("شلونك وشخبارك!") == "كيف حالك وشخبارك!"

    def test_no_partial_word_matches(self):
        t = IraqiTranslator()
        assert t.to_fusha("شلونكما") == "شلونكما"
        assert t.to_fusha("شلونك2") == "شلونك2"

    def test_alif_ha_variants_match(self):
        t = IraqiTranslator()
        assert t.to_fusha("شلونگ") != "شلونك"  # گ is not a variant of ك
        assert t.to_fusha("هسه") == "الآن"
        assert t.to_fusha("هسة") == "الآن"  # ة treated as ه variant

    def test_dictionary_keys_identical_across_boundaries(self):
        """Lightning-rod compatibility check: on every dictionary key the
        default (word) boundaries must behave exactly like the old ones."""
        dictionary = json.loads(DICTIONARY_PATH.read_text(encoding="utf-8"))
        new_t = IraqiTranslator()
        old_t = IraqiTranslator(strict_spaces=True)
        for key in dictionary:
            assert new_t.to_fusha(key) == old_t.to_fusha(key), key

    def test_to_iraqi_reverse_keys_identical_across_boundaries(self):
        dictionary = json.loads(DICTIONARY_PATH.read_text(encoding="utf-8"))
        new_t = IraqiTranslator()
        old_t = IraqiTranslator(strict_spaces=True)
        for value in set(dictionary.values()):
            assert new_t.to_iraqi(value) == old_t.to_iraqi(value), value

    def test_translation_is_not_re_translated(self):
        """A rule must only match text that was in the *input*.

        Regression guard for the cascading-substitution bug: rules used to be
        replayed over the whole text, so a short entry rewrote the output of a
        longer one (``ع`` turned ``تُعَالَجُ`` into ``تُعلئَالَجُ``).
        """
        rules = [
            (re.compile(r"long"), "Replacement"),
            (re.compile(r"l"), "X"),  # must not fire inside "Replacement"
        ]
        assert _apply("long", rules) == "Replacement"
        # a rule matching the *input* is unaffected by substitution order
        assert _apply("l", rules) == "X"

    def test_dictionary_entries_are_not_rewritten_by_shorter_keys(self):
        """Every dictionary entry must translate to exactly its stored value.

        Guards against a short entry silently re-translating the output of a
        longer one, which is how ``تنحل`` used to come out as ``تُعلئَالَجُ``.
        """
        dictionary = json.loads(DICTIONARY_PATH.read_text(encoding="utf-8"))
        t = IraqiTranslator()
        mismatched = {
            key: (value, t.to_fusha(key))
            for key, value in dictionary.items()
            if t.to_fusha(key) != value
        }
        # Two documented reasons for a stored value not to come back, both
        # read from the data so they cannot drift away from the validator:
        #   * the tolerant matcher folds ه/ة and alef variants, so a pair of
        #     spellings shares one pattern and only one can be honoured;
        #   * lexicon.json holds a deliberate ``override`` — a reviewed
        #     correction of a stored value.
        known = {key for _pair in ACCEPTED_COLLISIONS for key in _pair}
        known |= {entry["iraqi"] for entry in _lexicon_entries() if entry.get("override")}
        assert set(mismatched) <= known, mismatched

    def test_clitics_opt_in(self):
        t = IraqiTranslator(clitics=True)
        assert t.to_fusha("وشخبارك") == "وما أخبارك"
        assert t.to_fusha("وبس") == "وفقط"
        assert t.to_fusha("وهسه") == "والآن"

    def test_clitics_off_by_default(self):
        assert IraqiTranslator().to_fusha("وشخبارك") == "وشخبارك"


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
class TestNormalize:
    def test_tatweel_and_repeated_marks(self):
        assert normalize("شــــلونك؟؟؟") == "شلونك؟"

    def test_tashkeel(self):
        assert normalize("كَيْفَ حَالُكْ") == "كيف حالك"

    def test_alef_unification(self):
        assert normalize("أحمد إبراهيم آمنة") == "احمد ابراهيم امنة"

    def test_punctuation_mapping_and_collapse(self):
        assert normalize("شلونك؟؟؟!!") == "شلونك؟!"
        assert normalize("شلونك , شخبارك") == "شلونك، شخبارك"
        assert normalize("شلونك ؟") == "شلونك؟"

    def test_control_chars_removed(self):
        assert normalize("شلونك\u200c\u200b") == "شلونك"

    def test_whitespace(self):
        assert normalize("   شلونك   شخبارك  ") == "شلونك شخبارك"

    def test_teh_marbuta_kept_by_default(self):
        assert normalize("مدرسة") == "مدرسة"
        assert normalize("مدرسة", unify_teh_marbuta=True) == "مدرسه"

    def test_alef_maqsura_kept_by_default(self):
        assert normalize("على") == "على"
        assert normalize("على", unify_alef_maqsura=True) == "علي"

    def test_collapse_repeats(self):
        assert normalize("ششششلونك", collapse_repeats=True) == "شلونك"
        assert normalize("حبييييبي", collapse_repeats=True) == "حبيبي"

    def test_collapse_repeated_letters_preserves_doubled(self):
        from iraqitext import collapse_repeated_letters

        assert collapse_repeated_letters("شدّ") == "شدّ"  # run-length 2 kept
        assert collapse_repeated_letters("شششلونك") == "شلونك"

    def test_persian_variants(self):
        assert normalize("کيف حالک یار") == "كيف حالك يار"
        assert normalize("۱ ۲ ۳") == "١ ٢ ٣"

    def test_light_normalize_keeps_letters(self):
        assert light_normalize("أحمد") == "أحمد"
        assert light_normalize("شــــلونك؟") == "شلونك؟"


# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #
class TestTokenizer:
    def test_word_tokenize(self):
        assert word_tokenize("شلونك حبيبي؟") == ["شلونك", "حبيبي"]

    def test_tokenize_with_positions_and_kinds(self):
        tokens = tokenize("شلونك؟")
        assert [(t.text, t.kind) for t in tokens] == [("شلونك", "word"), ("؟", "punct")]
        assert tokens[0].start == 0 and tokens[0].end == 5
        assert tokens[1].start == 5 and tokens[1].end == 6

    def test_latin_words_and_numbers(self):
        tokens = tokenize("Windows 10 شلونك")
        kinds = [(t.text, t.kind) for t in tokens]
        assert ("Windows", "word") in kinds
        assert ("10", "number") in kinds
        assert ("شلونك", "word") in kinds

    def test_keep_spaces(self):
        tokens = tokenize("أ ب", keep_spaces=True)
        assert [t.kind for t in tokens] == ["word", "space", "word"]


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
class TestDetect:
    def test_iraqi_sentence(self):
        result = detect("شلونك شخبارك؟")
        assert isinstance(result, Detection)
        assert result.dialect == "iraqi"
        assert result.score >= 0.9
        assert result.confidence > 0.5

    def test_msa_sentence(self):
        result = detect("أذهب إلى المدرسة كل يوم")
        assert result.dialect == "msa"

    def test_mixed(self):
        result = detect("شلونك اليوم أذهب إلى المدرسة")
        assert result.dialect in ("iraqi", "mixed")

    def test_unknown(self):
        result = detect("Hello world, 123")
        assert result.dialect == "unknown"
        assert result.score == 0.0

    def test_is_iraqi(self):
        assert is_iraqi("شلونك شخبارك؟")
        assert not is_iraqi("أذهب إلى المدرسة كل يوم")

    def test_is_arabic(self):
        assert is_arabic("شلونك؟")
        assert not is_arabic("Hello")


# --------------------------------------------------------------------------- #
# Weighted evidence detection (0.2.1)
# --------------------------------------------------------------------------- #
class TestDetectWeighted:
    """The 0.2.1 detector weights evidence by tier:

    strong Iraqi / strong MSA > medium > neutral, with a transparent
    ``score`` = share of Iraqi evidence and a mixed zone for balanced
    evidence. Shared words must NOT push the verdict toward mixed.
    """

    IRAQI = [
        "شنو تريد تسوي؟",
        "شخبارك ويا أهلك؟",
        "ليش ما جيت البارحة؟",
        "تره الجو حلو اليوم.",
        "شكد سعر هذا؟",
    ]
    MSA = [
        "كيف حالك اليوم؟",
        "ماذا تريد أن تفعل؟",
        "أريد أن أذهب إلى السوق.",
        "إن الجو جميل اليوم.",
        "كم سعر هذا؟",
    ]
    MIXED = [
        "شلونك اليوم، كيف حالك؟",
        "شنو تريد أن تفعل؟",
        "وين تذهب هسه؟",
        "أريد أروح إلى السوق.",
        "ليش لم تأتِ البارحة؟",
        "هسه سوف أرجع.",
        "چان يجب أن تخبرني.",
    ]

    @pytest.mark.parametrize("sentence", IRAQI)
    def test_iraqi_sentences(self, sentence):
        assert detect(sentence).dialect == "iraqi"

    @pytest.mark.parametrize("sentence", MSA)
    def test_msa_sentences(self, sentence):
        assert detect(sentence).dialect == "msa"

    @pytest.mark.parametrize("sentence", MIXED)
    def test_mixed_sentences(self, sentence):
        assert detect(sentence).dialect == "mixed"

    def test_single_strong_marker_beats_neutral_words(self):
        # One strong Iraqi marker surrounded by shared words is still Iraqi.
        result = detect("تره الجو حلو اليوم.")
        assert result.dialect == "iraqi"
        assert result.details["evidence"]["تره"] == "strong_iraqi"

    def test_shared_words_do_not_force_mixed(self):
        # Neutral/shared words alone carry no evidence at all.
        result = detect("أريد تريد اليوم السوق")
        assert result.dialect == "msa"
        assert result.details["iraqi_evidence"] == 0.0
        assert result.details["msa_evidence"] == 0.0
        assert result.confidence < 0.5

    def test_mixed_reports_both_sides(self):
        result = detect("أريد أروح إلى السوق.")
        assert result.dialect == "mixed"
        assert result.details["iraqi_evidence"] > 0.0
        assert result.details["msa_evidence"] > 0.0

    def test_msa_counterweight_beats_single_iraqi_word(self):
        # A single Iraqi word is not enough to win against real MSA evidence.
        result = detect("چان يجب أن تخبرني.")
        assert result.dialect == "mixed"
        assert result.details["msa_evidence"] > result.details["iraqi_evidence"]

    def test_strong_msa_word_balances_strong_iraqi(self):
        result = detect("هسه سوف أرجع.")
        assert result.dialect == "mixed"
        assert result.details["msa_evidence"] > 0.0

    def test_iraqi_letters_are_decisive(self):
        result = detect("گاللي چان مريض.")
        assert result.dialect == "iraqi"
        assert result.details["marker_letters"] > 0

    def test_clitic_prefix_fallback(self):
        # وهل -> هل : MSA evidence still counts when glued to a clitic.
        result = detect("شلونك وهل أنت بخير؟")
        assert result.dialect == "mixed"
        assert result.details["evidence"]["وهل"] == "strong_msa"
        assert detect("فماذا تريد؟").dialect == "msa"

    def test_score_is_evidence_share(self):
        iraqi = detect("شنو تريد تسوي؟")
        assert iraqi.score >= 0.9  # strong Iraqi, no MSA evidence
        balanced = detect("شلونك اليوم، كيف حالك؟")
        assert balanced.score == pytest.approx(0.5, abs=0.01)

    def test_deterministic(self):
        text = "وين تذهب هسه؟"
        a, b = detect(text), detect(text)
        assert a.dialect == b.dialect == "mixed"
        assert a.score == b.score and a.confidence == b.confidence

    def test_threshold_override(self):
        # Lowering the Iraqi threshold lets a 75% Iraqi share count as Iraqi.
        text = "وين تذهب هسه؟"  # score == 0.75 on the default boundary
        assert detect(text).dialect == "mixed"
        assert detect(text, threshold=0.7).dialect == "iraqi"

    def test_details_evidence_map_transparent(self):
        result = detect("تره الجو حلو اليوم.")
        assert set(result.details["evidence"]) == {"تره", "الجو", "حلو", "اليوم"}
        assert result.details["evidence"]["الجو"] == "neutral"


# --------------------------------------------------------------------------- #
# Unified IraqiText API
# --------------------------------------------------------------------------- #
class TestIraqiText:
    def test_example_api(self):
        it = IraqiText()
        assert it.normalize("شــــلونك؟؟؟") == "شلونك؟"
        assert it.tokenize("شلونك حبيبي؟") == ["شلونك", "حبيبي"]
        assert it.detect("شلونك شخبارك؟").dialect == "iraqi"
        assert it.to_fusha("شلونك شخبارك؟") == "كيف حالك ما أخبارك؟"
        assert it.to_iraqi("كيف حالك") in ("شلونك", "إشلونك")

    def test_tokenize_with_positions(self):
        it = IraqiText()
        tokens = it.tokenize("شلونك؟", with_positions=True)
        assert tokens[0].text == "شلونك" and tokens[1].text == "؟"

    def test_keywords(self):
        it = IraqiText()
        kw = it.keywords("شلونك حبيبي شلونك عيني شلونك هسه أنا زين")
        assert kw[0] == "شلونك"
        assert "أنا" not in kw  # stopword

    def test_keywords_empty(self):
        it = IraqiText()
        assert it.keywords("أنا أنت هو هي") == []

    def test_translator_kwargs_forwarding(self):
        it = IraqiText(clitics=True)
        assert it.to_fusha("وشخبارك") == "وما أخبارك"


# --------------------------------------------------------------------------- #
# The six required sentences
# --------------------------------------------------------------------------- #
class TestRequiredSentences:
    """The examples the translation layer was built for."""

    @pytest.mark.parametrize(
        "iraqi, fusha",
        [
            ("أروح للدوام باچر", "أذهب إلى العمل غداً"),
            ("ما نكدر نجي هسه", "لا نستطيع أن نأتي الآن"),
            ("راح أگلك شصار", "سأخبرك بما حدث"),
            ("دا أحچي وياك", "أتحدث معك الآن"),
            ("مو كتلك لا تتأخر", "ألم أحذرك؟ لا تتأخر"),
            (
                "هلا وغلا، شاسمك؟ إذا تريد شي، خلي أساعدك؛ ماكو مشكلة.",
                "أهلاً وسهلاً، ما اسمك؟ إن أردت شيئاً، دعني أساعدك؛ لا مشكلة.",
            ),
        ],
    )
    def test_sentence(self, iraqi, fusha):
        assert IraqiTranslator().to_fusha(iraqi) == fusha

    def test_punctuation_is_never_broken(self):
        """Nothing the reader typed as punctuation is lost or reordered.

        A rule may *add* punctuation of its own (the phrase entry ``مو كتلك``
        carries a ``؟``), so the test is that every mark of the input is still
        there and still in the order it was written.
        """
        t = IraqiTranslator()
        marks = "!?،؛.()«»"
        cases = [
            "ما نكدر نجي هسه!",
            "راح أگلك شصار؟",
            "دا أحچي وياك، شلونك؟",
            "مو كتلك لا تتأخر؛ هلا وغلا.",
            "(هلا وغلا)",
            "«شلونك»",
            "راح أگلك شصار، بعدين؟",
        ]
        for text in cases:
            out = t.to_fusha(text)
            position = -1
            for mark in text:
                if mark not in marks:
                    continue
                position = out.find(mark, position + 1)
                assert position >= 0, f"{mark!r} lost or moved: {text!r} -> {out!r}"

    def test_a_mark_is_never_a_word_boundary(self):
        """A shadda sits *inside* a word, so it must not split one.

        Python's ``\\w`` does not match a combining mark, which used to let the
        entry ``ام`` match the tail of «خدّام» and produce «خدوالدة».
        """
        t = IraqiTranslator()
        assert t.to_fusha("خدام عندنا") == "عمّال عندنا"
        # the marked spelling matches no entry, but it is still one word
        assert t.to_fusha("خدّام عندنا") == "خدّام عندنا"

    def test_trailing_and_leading_spaces_are_preserved(self):
        t = IraqiTranslator()
        assert t.to_fusha("  شلونك  ") == "  كيف حالك  "


# --------------------------------------------------------------------------- #
# The morphology layer
# --------------------------------------------------------------------------- #
class TestMorphology:
    def test_object_suffixes(self):
        """ـه، ـها، ـهم، ـكم attach to the imperfective."""
        t = IraqiTranslator()
        assert t.to_fusha("أعرفه") == "أعرفه"  # shared; unchanged
        m = MorphologyEngine()
        assert m.verb_match("أگلك").msa == "أخبرك"
        assert m.verb_match("أعرفه").object == "ه"
        assert m.verb_match("أعرفها").object == "ها"
        assert m.verb_match("أعرفهم").object == "هم"

    def test_pronouns(self):
        m = MorphologyEngine()
        cases = {
            "أني": "أنا",
            "إنت": "أنت",
            "إنتي": "أنتِ",
            "إنتو": "أنتم",
            "إحنا": "نحن",
            "هوه": "هو",
            "هيه": "هي",
        }
        for iraqi, fusha in cases.items():
            result = m.translate_token(iraqi)
            assert result is not None, iraqi
            assert result.msa == fusha, iraqi

    def test_negation_words(self):
        """مو / ما / ماكو are all in the table, and all reach the output."""
        m = MorphologyEngine()
        for word in ("ما", "ماكو", "ماكوش", "مو"):
            assert word in m.negation, word
        t = IraqiTranslator()
        assert t.to_fusha("ماكو مشكلة") == "لا مشكلة"
        assert t.to_fusha("ما أريد") == "لا أريد"
        # «ما» before an imperfective is «لا», not the past «ما»
        assert t.to_fusha("ما تحچي") == "لا تتحدث"

    def test_tense_markers(self):
        """راح / دا / جاي are markers, and they drive the clause templates."""
        m = MorphologyEngine()
        assert m.markers["future"]["راح"]["msa"] == "س"
        assert m.markers["present"]["دا"]["msa"] == "الآن"
        assert m.markers["participial"]["جاي"]["msa"] == "قادم"
        assert m.translate_token("جاي") is not None
        t = IraqiTranslator()
        assert t.to_fusha("راح تروح بكرة") == "ستذهب غداً"
        assert t.to_fusha("دا أحچي وياك") == "أتحدث معك الآن"

    def test_imperfective_2fs_keeps_the_ya_suffix(self):
        """تحچين is «you (f, pl) talk», not «we talk»."""
        m = MorphologyEngine()
        match = m.verb_match("تحچين")
        assert match is not None
        assert match.person == "2fs"
        assert match.msa == "تتحدثين"

    def test_past_tense_forms(self):
        m = MorphologyEngine()
        assert m.verb_match("راحنا").msa == "ذهبنا"
        assert m.verb_match("عرفوا").msa == "عرفوا"
        assert m.verb_match("راحت").msa == "ذهبت"

    def test_verbs_in_the_required_list(self):
        """Every verb the task named must conjugate, not just the bare stem."""
        m = MorphologyEngine()
        cases = {
            "راح": "ذهب",
            "إجه": "جاء",
            "كدر": "استطاع",
            "راد": "أراد",
            "گال": "قال",
            "شاف": "رأى",
            "أخذ": "آخذ",
            "انطى": "انطلق",
            "أدز": "أرسل",
            "طب": "دخل",
            "طلع": "خرج",
            "گعد": "جلس",
            "گام": "قام",
            "عرف": "عرف",
            "سمع": "سمع",
            "حچى": "تحدث",
            "سأل": "سأل",
            "اشتغل": "أشتغل",
        }
        for surface, msa in cases.items():
            match = m.verb_match(surface)
            assert match is not None, surface
            assert match.msa == msa, surface

    def test_bare_dز_is_left_to_the_dictionary(self):
        """دز alone also means «to pay», so the engine refuses the bare form."""
        m = MorphologyEngine()
        assert m.verb_match("دز") is None
        assert m.verb_match("أدز") is not None  # the conjugated reading works

    def test_tashkeel_makes_the_engine_stand_aside(self):
        """خُليّ (left) vs خَليّ (was) — a mark decides the meaning."""
        m = MorphologyEngine()
        assert m.verb_match("أَگُلُ") is None
        assert m.verb_match("أگول") is not None
        # a marked word is left exactly as written rather than guessed at
        assert IraqiTranslator().to_fusha("أَگُلُ") == "أَگُلُ"

    def test_shared_msa_words_are_never_touched(self):
        m = MorphologyEngine()
        for word in ("كتاب", "مدرسة", "أريد", "فوق", "وجه", "أروحة"):
            assert m.translate_token(word) is None, word

    def test_every_verb_example_round_trips(self):
        m = MorphologyEngine()
        for key, verb in m.verbs.items():
            example = verb.example or {}
            if not example.get("in"):
                continue
            match = m.verb_match(example["in"])
            assert match is not None, f"{key}: {example['in']}"
            assert match.msa == example["out"], key

    def test_generated_forms_all_parse_back(self):
        """The coverage inventory may not claim a form the engine cannot read."""
        m = MorphologyEngine()
        broken = [f for f in m.generated_forms() if m.verb_match(f) is None]
        assert broken == []

    def test_stats(self):
        stats = MorphologyEngine().stats()
        assert stats["verbs"] >= 30
        assert stats["generated_verb_forms"] > 1000
        assert stats["pronouns"] >= 7


# --------------------------------------------------------------------------- #
# Tier priority
# --------------------------------------------------------------------------- #
class TestTierPriority:
    def test_phrase_beats_morphology_and_word(self):
        t = IraqiTranslator()
        # both «راح أجي» (phrase) and «راح» (word) exist
        assert t.to_fusha("راح أجي") == "سآتي"
        assert t.to_fusha("راح") == "سوف"

    def test_structure_beats_a_bare_word(self):
        """راح + verb is a future clause even though راح is a dictionary word."""
        t = IraqiTranslator()
        assert t.to_fusha("راح أگلك") == "سأخبرك"

    def test_morphology_beats_leaving_the_word_alone(self):
        t = IraqiTranslator()
        # «أگلك» has no dictionary entry; the morphology layer resolves it
        assert t.to_fusha("راح أگلك") == "سأخبرك"
        assert "أگلك" not in _fusha_map_of(t)

    def test_unknown_word_is_left_exactly_as_it_was(self):
        t = IraqiTranslator()
        assert t.to_fusha("أبوية") == "أبوية"

    def test_no_cascade_into_rule_output(self):
        """A rule's output must never be re-translated by another rule."""
        t = IraqiTranslator()
        # «غير» -> «سوى» and «ثقيل دم» -> «غير مرح» must not turn into «سوى مرح»
        assert "سوى مرح" not in t.to_fusha("ثقيل دم")

    def test_only_documented_overrides_change_a_stored_value(self):
        """Reading the whole dictionary back is the regression test.

        Three reasons a stored value may not come back, all declared in the
        data: the four tolerated matcher collisions, and the two lexicon
        entries marked ``override``.
        """
        d = json.loads(DICTIONARY_PATH.read_text(encoding="utf-8"))
        known = {key for pair in ACCEPTED_COLLISIONS for key in pair}
        known |= {
            entry["iraqi"] for entry in _lexicon_entries() if entry.get("override")
        }
        differing = {key for key, value in d.items() if _fusha_of(key) != value}
        assert differing <= known, differing

    def test_legacy_flag_still_runs_the_old_path(self):
        """``morphology=False`` keeps the 0.2 engine byte for byte."""
        old = IraqiTranslator(morphology=False)
        d = json.loads(DICTIONARY_PATH.read_text(encoding="utf-8"))
        tolerated = {key for pair in ACCEPTED_COLLISIONS for key in pair}
        assert {
            key
            for key, value in d.items()
            if old.to_fusha(key) != value and key not in tolerated
        } == set()
        # the new engine does more with the same text
        assert IraqiTranslator().to_fusha("راح أگلك") == "سأخبرك"
        assert old.to_fusha("راح أگلك") == "سوف أگلك"


def _fusha_map_of(translator):
    return set(translator._fusha_map)


_TRANSLATOR = IraqiTranslator()


def _fusha_of(text):
    """Module-level helper so the big regressions build the engine once."""
    return _TRANSLATOR.to_fusha(text)


def _lexicon_entries():
    return json.loads((PACKAGE_DIR / "lexicon.json").read_text(encoding="utf-8"))[
        "entries"
    ]


def _example_dialect(entry):
    """The dialect an entry's own example is meant to be read in."""
    regional = [r for r in entry["region"] if r != "general"]
    return regional[0] if regional else None


# --------------------------------------------------------------------------- #
# Coverage report  (تقرير الكلمات غير المترجمة)
# --------------------------------------------------------------------------- #
class TestTranslationReport:
    def test_report_shape(self):
        report = IraqiTranslator().to_fusha("راح أگلك شصار", report=True)
        assert isinstance(report, TranslationReport)
        assert report.output == "سأخبرك بما حدث"
        assert report.source == "راح أگلك شصار"
        assert report.is_fully_translated

    def test_untranslated_tokens_are_listed(self):
        """The headline feature: what the dictionary does not cover."""
        report = IraqiTranslator().to_fusha("أبوية وشغلة", report=True)
        assert report.untranslated_tokens == ["أبوية", "شغلة"]

    def test_names_numbers_and_english_are_not_errors(self):
        report = IraqiTranslator().to_fusha(
            "راحت بغداد 5 apples", report=True
        )
        assert report.untranslated_tokens == []
        assert "بغداد" in report.kept_tokens
        assert "apples" in report.kept_tokens

    def test_shared_msa_words_are_kept_not_flagged(self):
        report = IraqiTranslator().to_fusha("كتابي بمدرسة", report=True)
        assert "كتابي" not in report.untranslated_tokens
        assert "مدرسة" not in report.untranslated_tokens

    def test_confidence_per_tier(self):
        """A reviewed phrase is more trustworthy than a generated form."""
        t = IraqiTranslator()
        phrase = t.to_fusha("شلونك", report=True)
        generated = t.to_fusha("ما نكدر نجي هسه", report=True)
        assert phrase.confidence > generated.confidence
        assert generated.confidence >= MORPHOLOGY_CONFIDENCE - 0.1

    def test_per_token_confidence_and_tier(self):
        report = IraqiTranslator().to_fusha("مو كتلك لا تتأخر", report=True)
        tiers = {token.text: token.tier for token in report.tokens}
        assert tiers["ألم"] == "phrase"
        assert tiers["لا"] == "word"
        for token in report.tokens:
            assert 0.0 <= token.confidence <= 1.0

    def test_tiers_used_lists_what_fired(self):
        report = IraqiTranslator().to_fusha("راح أگلك شصار", report=True)
        assert "structure" in report.tiers_used

    def test_rules_are_named(self):
        report = IraqiTranslator().to_fusha("مو كتلك", report=True)
        assert "مو كتلك" in report.rules

    def test_to_dict_is_json_serialisable(self):
        report = IraqiTranslator().to_fusha("أبوية", report=True)
        payload = report.to_dict()
        assert payload["untranslated_tokens"] == ["أبوية"]
        json.dumps(payload, ensure_ascii=False)  # must not raise

    def test_report_works_in_legacy_mode_too(self):
        report = IraqiTranslator(morphology=False).to_fusha("أبوية", report=True)
        assert report.untranslated_tokens == ["أبوية"]
        assert report.output == "أبوية"


class TestCoverageLog:
    def test_frequency_log_and_review_list(self):
        log = CoverageLog()
        t = IraqiTranslator()
        for text in (
            "أبوية وشخانة",
            "عندي شخانة وأبوية",
            "شلونك شخانة",
            "أبوية",
        ):
            log.add_text(text, translator=t)
        assert log.top(2) == [("أبوية", 3), ("شخانة", 3)]
        review = log.review_list()
        assert review[0]["count"] == 3
        assert "أبوية" in review[0]["examples"]

    def test_a_written_conjunction_is_stripped_from_the_gap(self):
        """وشخانة is one token to the tokenizer; the gap is شخانة."""
        log = CoverageLog()
        log.add_text("وشخانة")
        assert log.top(1) == [("شخانة", 1)]

    def test_a_translated_word_never_reaches_the_log(self):
        log = CoverageLog()
        log.add_text("شلونك راحت بغداد 5")
        assert log.top(10) == []

    def test_minimum_threshold(self):
        log = CoverageLog(minimum=2)
        t = IraqiTranslator()
        log.add_text("شخانة", translator=t)
        log.add_text("أبوية", translator=t)
        log.add_text("شخانة", translator=t)
        assert log.top(10) == [("شخانة", 2)]

    def test_save_and_load_round_trip(self, tmp_path):
        log = CoverageLog()
        log.add_text("أبوية")
        path = log.save(tmp_path / "log.json")
        assert path.exists()
        assert CoverageLog().load(path).to_dict() == log.to_dict()

    def test_membership_and_length(self):
        log = CoverageLog()
        log.add_text("أبوية")
        assert "أبوية" in log
        assert len(log) == 1
        log.clear()
        assert len(log) == 0


def test_coverage_stats():
    stats = coverage_stats()
    assert stats["dictionary_entries"] > 1000
    assert stats["dictionary_phrases"] > 100
    assert stats["lexicon_entries"] > 0
    assert stats["morphology_enabled"] is True
    assert stats["verbs"] >= 30


def test_translator_exposes_coverage_stats():
    stats = IraqiTranslator().coverage_stats()
    assert stats["generated_verb_forms"] > 1000


# --------------------------------------------------------------------------- #
# Dialect tags
# --------------------------------------------------------------------------- #
class TestDialects:
    def test_unknown_dialect_is_rejected(self):
        with pytest.raises(ValueError):
            IraqiTranslator(dialect="egyptian")

    def test_no_egyptian_dialect_exists(self):
        """The project is Iraqi -> MSA.  Egyptian is not a target, not a source."""
        assert "egyptian" not in DIALECTS
        with pytest.raises(ValueError):
            IraqiTranslator(dialect="egyptian")
        with pytest.raises(ValueError):
            IraqiTranslator().to_fusha("شلونك", dialect="egyptian")

    def test_mosul_phrase_is_not_offered_to_everyone(self):
        """«كيف هسة» is the Nineveh spelling of «كيف هسه»."""
        from iraqitext.morphology import load_lexicon

        mosul = {e.iraqi for e in load_lexicon(dialect="mosul")}
        baghdad = {e.iraqi for e in load_lexicon(dialect="baghdad")}
        assert "كيف هسة" in mosul
        assert "كيف هسة" not in baghdad
        assert IraqiTranslator(dialect="mosul").to_fusha("كيف هسة") == "كيف الآن"

    def test_southern_word_only_for_the_south(self):
        """نشف is «to bleed» in the south and «to dry up» in Baghdad."""
        assert IraqiTranslator(dialect="south").to_fusha("نشف") == "انزف"
        assert IraqiTranslator(dialect="baghdad").to_fusha("نشف") == "جف"
        assert IraqiTranslator(dialect="mosul").to_fusha("نشف") == "جف"

    def test_region_specific_entries_carry_a_lower_confidence(self):
        """A regional word is offered, but never as certain as a general one."""
        from iraqitext.morphology import load_lexicon

        general = {
            e.iraqi: e.confidence for e in load_lexicon() if "general" in e.region
        }
        regional = [
            e for e in load_lexicon() if "general" not in e.region and e.region
        ]
        assert regional, "no region-specific entry in the sample data"
        for entry in regional:
            assert entry.confidence <= min(general.values())

    def test_general_word_works_everywhere(self):
        for dialect in DIALECTS:
            t = IraqiTranslator(dialect=dialect)
            assert t.to_fusha("للدوام") == "إلى العمل"

    def test_per_call_override(self):
        t = IraqiTranslator()
        assert t.to_fusha("نشف", dialect="south") == "انزف"

    def test_dialect_defaults_to_everything(self):
        assert IraqiTranslator().to_fusha("كيف هسه") == "كيف الآن"


# --------------------------------------------------------------------------- #
# Structured lexicon
# --------------------------------------------------------------------------- #
class TestLexicon:
    def test_entries_carry_their_metadata(self):
        entries = _lexicon_entries()
        assert entries, "lexicon.json has no entries"
        for entry in entries:
            assert entry["iraqi"] and entry["fusha"]
            assert entry["type"]
            assert entry["region"]
            assert entry["source"]
            assert 0.0 <= entry["confidence"] <= 1.0
            for region in entry["region"]:
                assert region in DIALECTS

    def test_lexicon_is_extensible_beyond_flat_json(self):
        """New entries live as objects, not as a flat one-word-one-string map."""
        data = json.loads((PACKAGE_DIR / "lexicon.json").read_text(encoding="utf-8"))
        assert isinstance(data["entries"], list)
        assert all(isinstance(entry, dict) for entry in data["entries"])
        # the documented fields are all actually used
        keys = {k for entry in data["entries"] for k in entry}
        assert {"iraqi", "fusha", "type", "region", "source", "confidence"} <= keys

    def test_example_of_each_entry_translates(self):
        """Every shipped example has to actually exercise its own entry."""
        for entry in _lexicon_entries():
            example = entry.get("example")
            if not example:
                continue
            dialect = _example_dialect(entry)
            out = IraqiTranslator(dialect=dialect).to_fusha(example)
            assert out != example, f"{entry['iraqi']}: {example!r} did not translate"

    def test_lexicon_phrase_is_used(self):
        t = IraqiTranslator()
        assert t.to_fusha("أروح للدوام باچر") == "أذهب إلى العمل غداً"


# --------------------------------------------------------------------------- #
# Data validation
# --------------------------------------------------------------------------- #
class TestDataValidation:
    def test_shipped_data_is_valid(self):
        from iraqitext.validate import validate

        report = validate()
        assert report.ok, str(report)

    def test_no_egyptian_data(self):
        """This library is Iraqi -> MSA.  An Egyptian entry is a data bug.

        The check is a *word* match, not a substring: بتاع is a substring of
        ابتاع ("he bought"), which is ordinary MSA and has to stay.
        """
        from iraqitext.coverage import EGYPTIAN_ONLY

        assert "عايز" in EGYPTIAN_ONLY and "دلوقتي" in EGYPTIAN_ONLY
        pattern = re.compile(r"(?<!\S)(" + "|".join(EGYPTIAN_ONLY) + r")(?!\S)")
        for name in ("dictionary.json", "morphology.json", "lexicon.json"):
            raw = (PACKAGE_DIR / name).read_text(encoding="utf-8")
            match = pattern.search(raw)
            assert match is None, f"{match.group(0)!r} in {name}"

    def test_the_egyptian_marker_set_holds_no_iraqi_word(self):
        """A marker that Iraqi also uses would ban Iraqi, not Egyptian."""
        from iraqitext.coverage import EGYPTIAN_ONLY

        for word in ("يا", "مش", "عم", "ليه", "كمان", "كويس", "معاك", "سيدي"):
            assert word not in EGYPTIAN_ONLY, word
        assert len(EGYPTIAN_ONLY) >= 15

    def test_the_validator_agrees(self):
        from iraqitext.validate import validate

        assert validate().ok

    def test_no_junk_entries(self):
        dictionary = json.loads(DICTIONARY_PATH.read_text(encoding="utf-8"))
        assert "x" not in dictionary
        for key, value in dictionary.items():
            assert value.strip().lower() not in {"", "x", "todo", "tbd"}


# --------------------------------------------------------------------------- #
# Import surface
# --------------------------------------------------------------------------- #
def test_module_exports():
    import iraqitext

    assert hasattr(iraqitext, "IraqiTranslator")
    assert hasattr(iraqitext, "IraqiText")
    assert hasattr(iraqitext, "normalize")
    assert hasattr(iraqitext, "tokenize")
    assert hasattr(iraqitext, "detect")
    assert hasattr(iraqitext, "MorphologyEngine")
    assert hasattr(iraqitext, "TranslationReport")
    assert hasattr(iraqitext, "CoverageLog")
    assert iraqitext.__version__ == "0.3.0"
