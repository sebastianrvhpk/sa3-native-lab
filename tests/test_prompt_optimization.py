from latent_audio_primitives.prompt_optimization import (
    beam_token_prompt_search,
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


def test_greedy_token_prompt_search_batches_candidate_scoring():
    vocab = ["a", "b", "c", "d", "e"]
    batch_sizes = []

    def scorer(prompts):
        batch_sizes.append(len(prompts))
        return [float(prompt.endswith("e")) for prompt in prompts]

    result = greedy_token_prompt_search(
        vocab,
        scorer,
        tokens_generated=1,
        runs=1,
        token_subset=None,
        candidate_batch_size=2,
        seed=0,
    )

    assert result.tokens == ["e"]
    assert batch_sizes[:3] == [2, 2, 1]


def test_beam_token_prompt_search_keeps_multiple_partial_prompts():
    vocab = ["a", "bridge", "wide", "noise"]

    def scorer(prompts):
        scores = []
        for prompt in prompts:
            score = 0.0
            if "wide" in prompt:
                score += 1.0
            if "bridge wide" in prompt:
                score += 2.0
            if prompt.startswith("bridge"):
                score += 0.9
            scores.append(score)
        return scores

    result = beam_token_prompt_search(
        vocab,
        scorer,
        tokens_generated=2,
        beam_width=2,
        branch_factor=None,
        candidate_batch_size=3,
        seed=0,
    )

    assert result.prompt == "bridge wide"
    assert result.score == 3.9
    assert len(result.beams) == 2
