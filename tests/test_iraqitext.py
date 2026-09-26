"""End-to-end tests for iraqitext 0.2.

Covers backward compatibility of the 0.1 API (IraqiTranslator) plus the new
NLP surface: normalize, tokenize, detect and the unified IraqiText API.
"""
import json
import re
from pathlib import Path

import pytest

from iraqitext import (
    Detection,
    IraqiText,
    IraqiTranslator,
    detect,
    is_arabic,
    is_iraqi,
    light_normalize,
    normalize,
    tokenize,
    word_tokenize,
)
from iraqitext.translator import _apply

DICTIONARY_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "iraqitext" / "dictionary.json"
)


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
        # Known, tolerated ambiguity: the tolerant matcher folds ه/ة and alef
        # variants, so these pairs share one pattern and cannot both be honoured.
        known = {"أبد", "إبد", "أنسى", "انسى", "عدة", "عده", "قوطيه", "قوطية"}
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
# Import surface
# --------------------------------------------------------------------------- #
def test_module_exports():
    import iraqitext

    assert hasattr(iraqitext, "IraqiTranslator")
    assert hasattr(iraqitext, "IraqiText")
    assert hasattr(iraqitext, "normalize")
    assert hasattr(iraqitext, "tokenize")
    assert hasattr(iraqitext, "detect")
    assert iraqitext.__version__ == "0.2.0"