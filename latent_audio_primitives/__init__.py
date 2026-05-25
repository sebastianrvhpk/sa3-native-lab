"""Research primitives for latent audio memory and composition."""

from .composition import (
    TransitionWeights,
    best_path,
    bridge_cost,
    loop_cost,
    ranked_bridges,
    ranked_continuations,
    transition_cost,
)
from .index import LatentMemoryIndex, SearchResult
from .io import load_item, load_items, save_item, save_items
from .prompt_optimization import (
    GreedyPromptSearchResult,
    PromptOptimizationResult,
    coordinate_prompt_search,
    default_modifier_axes,
    greedy_token_prompt_search,
    prompt_seed_from_audio_path,
)
from .latent_math import (
    as_time_major,
    boundary_summary,
    cosine_similarity,
    euclidean_distance,
    latent_summary,
)
from .schema import LatentItem
from .audio_vectors import (
    AudioSetDirection,
    apply_frame_direction,
    frame_mean_direction,
    load_summary_direction,
    save_audio_frame_direction,
    save_summary_direction,
    summary_direction,
)
from .style import (
    LatentStyleDirection,
    LatentStyleProfile,
    apply_profile_attraction,
    apply_profile_to_item,
    apply_style_direction,
    fit_style_profile,
    load_style_direction,
    load_style_profile,
    save_style_direction,
    save_style_profile,
    style_direction,
)
from .experiments.soft_prompt import SoftPromptState

__all__ = [
    "LatentItem",
    "LatentMemoryIndex",
    "SearchResult",
    "TransitionWeights",
    "LatentStyleDirection",
    "LatentStyleProfile",
    "AudioSetDirection",
    "PromptOptimizationResult",
    "GreedyPromptSearchResult",
    "SoftPromptState",
    "apply_frame_direction",
    "apply_profile_attraction",
    "apply_profile_to_item",
    "apply_style_direction",
    "as_time_major",
    "best_path",
    "boundary_summary",
    "bridge_cost",
    "cosine_similarity",
    "coordinate_prompt_search",
    "default_modifier_axes",
    "euclidean_distance",
    "frame_mean_direction",
    "greedy_token_prompt_search",
    "fit_style_profile",
    "latent_summary",
    "load_item",
    "load_items",
    "load_style_direction",
    "load_style_profile",
    "load_summary_direction",
    "loop_cost",
    "prompt_seed_from_audio_path",
    "ranked_bridges",
    "ranked_continuations",
    "save_item",
    "save_items",
    "save_style_direction",
    "save_style_profile",
    "save_summary_direction",
    "save_audio_frame_direction",
    "style_direction",
    "summary_direction",
    "transition_cost",
]
