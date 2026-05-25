from latent_audio_primitives.prompt_optimization import (
    coordinate_prompt_search,
    default_modifier_axes,
    greedy_token_prompt_search,
    prompt_seed_from_audio_path,
)


def test_prompt_seed_from_audio_path_uses_folder_and_filename_tokens():
    prompt = prompt_seed_from_audio_path("/datasets/glass_swamp/bright_loop_01.wav", extra_tags=["ambient"])

    assert "glass" in prompt
    assert "swamp" in prompt
    assert "bright" in prompt
    assert "loop" in prompt
    assert "ambient" in prompt


def test_coordinate_prompt_search_uses_pluggable_scorer():
    def scorer(prompt: str) -> float:
        return float(("bright" in prompt) + ("wide stereo" in prompt))

    result = coordinate_prompt_search(
        "ambient texture",
        [["dark", "bright"], ["mono", "wide stereo"]],
        scorer,
        rounds=2,
    )

    assert "bright" in result.prompt
    assert "wide stereo" in result.prompt
    assert result.score == 2.0


def test_default_modifier_axes_are_non_empty():
    assert default_modifier_axes()


def test_greedy_token_prompt_search_builds_prompt_with_batch_scorer():
    vocab = ["dark", "bright", "mono", "wide"]

    def scorer(prompts):
        return [float(("bright" in prompt) + ("wide" in prompt)) for prompt in prompts]

    result = greedy_token_prompt_search(
        vocab,
        scorer,
        tokens_generated=2,
        runs=1,
        token_subset=None,
        seed=0,
    )

    assert "bright" in result.tokens
    assert "wide" in result.tokens
    assert result.score == 2.0
