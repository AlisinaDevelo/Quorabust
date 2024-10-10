import pytest

from quorabust.preprocess import clean_text, tokenize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Hello   World!  ", "hello world"),
        ("café", "café"),
        ("", ""),
        (None, ""),
    ],
)
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_tokenize_splits_words():
    assert tokenize("A B  a") == ["a", "b", "a"]
