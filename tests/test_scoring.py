import pytest

from scoring.normalize import normalize
from scoring.wer import compute_wer


def test_case_and_punctuation():
    assert normalize("Hello, World!") == "hello world"


def test_apostrophes_kept():
    assert normalize("Don't stop") == "don't stop"
    assert normalize("it’s") == "it's"


def test_hyphen_splits():
    assert normalize("well-known") == "well known"


def test_numbers():
    assert normalize("I have 3 cats") == "i have three cats"
    assert normalize("the 2nd one") == "the second one"
    assert normalize("1,000 people") == "one thousand people"


def test_percent_and_tags():
    assert normalize("50% [laughter] done") == "fifty percent done"


def test_british_american_spelling_unified():
    assert normalize("travellers honour colour") == normalize("travelers honor color")


def test_empty():
    assert normalize("") == ""
    assert normalize(None) == ""


def test_wer_zero_when_only_formatting_differs():
    assert compute_wer(["Hello, world."], ["hello world"]) == 0.0


def test_wer_value():
    assert compute_wer(["a b c d"], ["a b x d"]) == pytest.approx(0.25)


def test_wer_empty_hypothesis_is_one():
    assert compute_wer(["a b"], [""]) == pytest.approx(1.0)


def test_wer_all_empty_refs_raises():
    with pytest.raises(ValueError):
        compute_wer([""], ["x"])


# --- normalize() edge cases (issue #4) ---


def test_ampersand_expands_to_and():
    assert normalize("A&B") == "a and b"


def test_decimal_number_to_words():
    assert normalize("3.5") == "three point five"


def test_ordinal_number_to_words():
    assert normalize("21st") == "twenty first"


def test_stray_quote_marks_removed_but_inner_apostrophes_kept():
    assert normalize("'quoted'") == "quoted"
    assert normalize("'don't'") == "don't"


def test_xml_like_tags_dropped():
    assert normalize("<unk> hello") == "hello"


def test_underscores_split_words():
    assert normalize("snake_case") == "snake case"


def test_fullwidth_characters_fold_to_ascii():
    assert normalize("ＡＢＣ") == "abc"


# --- compute_wer() corpus behaviour (issue #5) ---


def test_wer_is_weighted_by_reference_length_not_averaged_per_pair():
    # 6 reference words total, 1 error -> 1/6, not the mean of 0.5 and 0.0.
    assert compute_wer(["a b", "c d e f"], ["a x", "c d e f"]) == pytest.approx(1 / 6)


def test_wer_drops_pairs_with_empty_normalized_reference():
    baseline = compute_wer(["a b", "c d e f"], ["a x", "c d e f"])
    # "" and "[noise]" both normalize to "", so those pairs carry no scoreable words.
    with_empty = compute_wer(
        ["a b", "", "[noise]", "c d e f"],
        ["a x", "ignored", "ignored", "c d e f"],
    )
    assert with_empty == pytest.approx(baseline)


def test_wer_unequal_lengths_truncate_to_the_shorter_list_like_zip():
    # zip() stops at the shortest list, so the second reference is never scored.
    assert compute_wer(["a b", "c d"], ["a x"]) == pytest.approx(0.5)
