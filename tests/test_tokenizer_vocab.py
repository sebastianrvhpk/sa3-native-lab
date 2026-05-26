from __future__ import annotations

from latent_audio_primitives.tokenizer_vocab import (
    extract_prompt_tokenizer,
    native_tokenizer_vocabulary,
    preview_native_tokenizer_vocabulary,
)


class FakeTokenizer:
    all_special_ids = [0]

    def __init__(self):
        self._vocab = {
            "<pad>": 0,
            "\u2581bright": 1,
            "fragment": 2,
            "\u2581dark": 3,
            "\u2581!!!": 4,
            "\u2581longwordthatiswaytoolongforthis": 5,
            "\u2581wide-stereo": 6,
            "\u2581s\u00ed": 7,
        }

    def get_vocab(self):
        return self._vocab

    def convert_tokens_to_string(self, tokens):
        return "".join(token.replace("\u2581", " ") for token in tokens)


class FakeConditioner:
    def __init__(self):
        self.tokenizer = FakeTokenizer()


class FakeMultiConditioner:
    def __init__(self):
        self.conditioners = {"prompt": FakeConditioner()}


class FakeCore:
    def __init__(self):
        self.conditioner = FakeMultiConditioner()


class FakeStableModel:
    def __init__(self):
        self.model = FakeCore()


def test_extract_prompt_tokenizer_from_sa3_like_model():
    tokenizer = extract_prompt_tokenizer(FakeStableModel())

    assert isinstance(tokenizer, FakeTokenizer)


def test_native_tokenizer_vocabulary_filters_word_pieces():
    vocab = native_tokenizer_vocabulary(
        FakeTokenizer(),
        max_candidates=10,
        max_chars=16,
        require_word_start=True,
        ascii_only=True,
    )

    assert vocab == ["bright", "dark", "wide-stereo"]


def test_native_tokenizer_vocabulary_can_include_fragments():
    vocab = native_tokenizer_vocabulary(
        FakeTokenizer(),
        max_candidates=10,
        require_word_start=False,
        ascii_only=True,
    )

    assert "fragment" in vocab


def test_preview_native_tokenizer_vocabulary_formats_rows():
    preview = preview_native_tokenizer_vocabulary(["bright", "dark", "wide"], columns=2, rows=2)

    assert "bright" in preview
    assert "wide" in preview
