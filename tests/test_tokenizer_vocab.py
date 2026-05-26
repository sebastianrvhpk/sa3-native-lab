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
            "\u2581the": 1,
            "\u2581bright": 2,
            "fragment": 3,
            "\u2581dark": 4,
            "\u2581!!!": 5,
            "\u2581longwordthatiswaytoolongforthis": 6,
            "\u2581wide-stereo": 7,
            "\u2581s\u00ed": 8,
            "\u2581bass": 9,
            "\u2581loop": 10,
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

    assert vocab == ["bass", "loop", "bright", "dark", "wide-stereo"]


def test_native_tokenizer_vocabulary_can_include_fragments():
    vocab = native_tokenizer_vocabulary(
        FakeTokenizer(),
        max_candidates=10,
        require_word_start=False,
        ascii_only=True,
    )

    assert "fragment" in vocab
    assert "the" not in vocab


def test_native_tokenizer_vocabulary_can_preserve_token_order():
    vocab = native_tokenizer_vocabulary(
        FakeTokenizer(),
        max_candidates=10,
        require_word_start=True,
        ascii_only=True,
        rank_by_audio_prior=False,
    )

    assert vocab[:3] == ["bright", "dark", "wide-stereo"]


def test_preview_native_tokenizer_vocabulary_formats_rows():
    preview = preview_native_tokenizer_vocabulary(["bright", "dark", "wide"], columns=2, rows=2)

    assert "bright" in preview
    assert "wide" in preview
