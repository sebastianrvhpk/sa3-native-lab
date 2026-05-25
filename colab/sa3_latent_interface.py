"""Gradio interface scaffold for SA3 Medium latent memory and steering.

Run this in Colab after installing Stable Audio 3 and this repo. It provides a
small interface for:

    - loading SA3 medium
    - generating audio and storing SAME latents
    - searching latent memory
    - extracting steering vectors from prompt pairs
    - running alpha sweeps

This is intentionally research-first. It keeps metadata and generated latents
on disk so you can inspect them later.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from latent_audio_primitives import LatentMemoryIndex, load_items, save_items
from latent_audio_primitives.adapters.audioscope_sa3 import SteeringVectors
from latent_audio_primitives.adapters.stable_audio3 import StableAudio3Adapter
from latent_audio_primitives.experiments.activation_vectors import SA3ActivationVectorExtractor
from latent_audio_primitives.experiments.prompt_pairs import DEFAULT_PROMPT_PAIRS
from latent_audio_primitives.experiments.sa3_sweeps import alpha_sweep
from latent_audio_primitives.style import apply_profile_attraction, load_style_profile


STATE = {
    "model": None,
    "sa3": None,
    "items": [],
    "index": None,
    "vectors": None,
    "profile": None,
}

ROOT = Path("/content/sa3_latent_lab")
MEMORY_DIR = ROOT / "memory"
VECTORS_DIR = ROOT / "vectors"
SWEEPS_DIR = ROOT / "sweeps"


def load_medium() -> str:
    from stable_audio_3 import StableAudioModel

    model = StableAudioModel.from_pretrained("medium", device="cuda", model_half=True)
    STATE["model"] = model
    STATE["sa3"] = StableAudio3Adapter(model=model, model_name="medium")
    return (
        "Loaded SA3 medium\n"
        f"sample_rate={model.model.sample_rate}\n"
        f"downsampling_ratio={model.model.pretransform.downsampling_ratio}"
    )


def generate_to_memory(prompt: str, duration: float, seed: int, steps: int, cfg_scale: float):
    sa3 = _sa3()
    prefix = f"gen_seed{seed}"
    items = sa3.generate_items(
        prompt=prompt,
        duration=duration,
        item_id_prefix=prefix,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
    )
    STATE["items"].extend(items)
    STATE["index"] = LatentMemoryIndex(STATE["items"])
    save_items(STATE["items"], MEMORY_DIR)

    latents = items[0].latent
    audio = sa3.model.generate(
        prompt=prompt,
        duration=duration,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
    )
    audio_path = ROOT / f"{items[0].item_id}.wav"
    _save_audio(audio_path, audio, sa3.sample_rate)
    return str(audio_path), _memory_summary()


def load_memory_from_disk():
    if not MEMORY_DIR.exists():
        return "No saved memory yet."
    items = load_items(MEMORY_DIR)
    STATE["items"] = items
    STATE["index"] = LatentMemoryIndex(items)
    return _memory_summary()


def search_memory(query_id: str, top_k: int):
    index = _index()
    query = index.get(query_id)
    results = index.query(query, top_k=top_k, exclude_id=query.item_id)
    lines = []
    for result in results:
        lines.append(f"{result.item_id} score={result.score:.4f} prompt={result.item.prompt}")
    return "\n".join(lines) if lines else "No results."


def extract_vectors(axis: str, num_pairs: int, duration: float, steps: int, cfg_scale: float, seed: int):
    model = _model()
    pairs = [pair for pair in DEFAULT_PROMPT_PAIRS if axis == "all" or pair.axis == axis]
    extractor = SA3ActivationVectorExtractor(model, cpu_offload=True)
    result = extractor.extract(
        pairs=pairs,
        num_pairs=num_pairs,
        duration=duration,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        probe=True,
    )
    out_dir = VECTORS_DIR / axis
    result.save(out_dir)
    STATE["vectors"] = result.vectors
    return (
        f"Saved vectors to {out_dir}\n"
        f"layers={sorted(result.vectors.vectors)}\n"
        f"best_layer={result.vectors.best_layer}\n"
        f"probe_accuracy={result.vectors.probe_accuracy}"
    )


def load_vectors(path: str):
    vectors = SteeringVectors.load(path)
    STATE["vectors"] = vectors
    return f"Loaded vectors. layers={sorted(vectors.vectors)} best_layer={vectors.best_layer}"


def run_alpha_sweep(prompt: str, alphas_csv: str, duration: float, steps: int, cfg_scale: float, seed: int, layer: int):
    sa3 = _sa3()
    vectors = _vectors()
    alphas = [float(value.strip()) for value in alphas_csv.split(",") if value.strip()]
    out_dir = SWEEPS_DIR / f"sweep_seed{seed}"
    outputs = alpha_sweep(
        sa3,
        prompt=prompt,
        vectors=vectors,
        alphas=alphas,
        output_dir=out_dir,
        duration=duration,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        layer=layer if layer >= 0 else None,
        top_k=1,
        save_audio=True,
    )
    paths = [output.audio_path for output in outputs if output.audio_path]
    return paths[0] if paths else None, "\n".join(paths)


def load_style_profile_file(path: str):
    profile = load_style_profile(path)
    STATE["profile"] = profile
    return f"Loaded profile {profile.name}: items={profile.item_count} dim={profile.dim}"


def generate_with_style_profile(prompt: str, duration: float, seed: int, steps: int, cfg_scale: float, alpha: float, match_std: bool):
    sa3 = _sa3()
    profile = _profile()
    latents = sa3.generate_latents(
        prompt=prompt,
        duration=duration,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
    )
    styled = _apply_profile_to_latent_batch(latents, profile, alpha=alpha, match_std=match_std)

    original_audio = sa3.decode_latents(latents)
    styled_audio = sa3.decode_latents(styled)
    original_path = ROOT / f"profile_seed{seed}_original.wav"
    styled_path = ROOT / f"profile_seed{seed}_styled.wav"
    _save_audio(original_path, original_audio, sa3.sample_rate)
    _save_audio(styled_path, styled_audio, sa3.sample_rate)
    return str(styled_path), f"original={original_path}\nstyled={styled_path}"


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="SA3 Latent Lab") as demo:
        gr.Markdown("# SA3 Latent Lab\nResearch interface for SA3 Medium, SAME latents, memory, and steering.")

        with gr.Tab("Setup"):
            load_button = gr.Button("Load SA3 Medium")
            setup_status = gr.Textbox(lines=4)
            load_button.click(load_medium, outputs=setup_status)

        with gr.Tab("Generate + Memory"):
            prompt = gr.Textbox(value="a sparse glassy ambient loop, slow evolving texture", label="Prompt")
            duration = gr.Slider(2, 30, value=8, step=1, label="Duration")
            seed = gr.Number(value=42, precision=0, label="Seed")
            steps = gr.Slider(1, 50, value=8, step=1, label="Steps")
            cfg = gr.Slider(0, 12, value=1.0, step=0.1, label="CFG")
            gen = gr.Button("Generate and Store Latent")
            audio = gr.Audio(label="Audio")
            memory_status = gr.Textbox(lines=8, label="Memory")
            gen.click(generate_to_memory, inputs=[prompt, duration, seed, steps, cfg], outputs=[audio, memory_status])

            load_mem = gr.Button("Load Memory From Disk")
            load_mem.click(load_memory_from_disk, outputs=memory_status)

        with gr.Tab("Search"):
            query_id = gr.Textbox(label="Query item_id")
            top_k = gr.Slider(1, 20, value=5, step=1, label="Top K")
            search = gr.Button("Search Neighbors")
            results = gr.Textbox(lines=12)
            search.click(search_memory, inputs=[query_id, top_k], outputs=results)

        with gr.Tab("Extract Vectors"):
            axis = gr.Dropdown(
                ["all", "valence", "brightness", "density", "tension", "grain", "stereo_width", "section_energy"],
                value="valence",
                label="Axis",
            )
            num_pairs = gr.Slider(1, len(DEFAULT_PROMPT_PAIRS), value=2, step=1, label="Num pairs")
            vec_duration = gr.Slider(2, 20, value=6, step=1, label="Duration")
            vec_steps = gr.Slider(1, 50, value=8, step=1, label="Steps")
            vec_cfg = gr.Slider(0, 12, value=1.0, step=0.1, label="CFG")
            vec_seed = gr.Number(value=100, precision=0, label="Seed")
            extract = gr.Button("Extract Steering Vectors")
            extract_status = gr.Textbox(lines=10)
            extract.click(
                extract_vectors,
                inputs=[axis, num_pairs, vec_duration, vec_steps, vec_cfg, vec_seed],
                outputs=extract_status,
            )

            vec_path = gr.Textbox(value=str(VECTORS_DIR / "valence" / "steering_vectors.pt"), label="Vector path")
            load_vec = gr.Button("Load Vectors")
            load_vec.click(load_vectors, inputs=vec_path, outputs=extract_status)

        with gr.Tab("Alpha Sweep"):
            sweep_prompt = gr.Textbox(value="a cinematic ambient piano phrase", label="Prompt")
            alphas = gr.Textbox(value="-8,-4,0,4,8", label="Alphas")
            sweep_duration = gr.Slider(2, 30, value=8, step=1, label="Duration")
            sweep_steps = gr.Slider(1, 50, value=8, step=1, label="Steps")
            sweep_cfg = gr.Slider(0, 12, value=1.0, step=0.1, label="CFG")
            sweep_seed = gr.Number(value=123, precision=0, label="Seed")
            layer = gr.Number(value=-1, precision=0, label="Layer (-1 uses best/top)")
            run_sweep = gr.Button("Run Alpha Sweep")
            first_audio = gr.Audio(label="First sweep audio")
            sweep_paths = gr.Textbox(lines=8, label="Saved audio paths")
            run_sweep.click(
                run_alpha_sweep,
                inputs=[sweep_prompt, alphas, sweep_duration, sweep_steps, sweep_cfg, sweep_seed, layer],
                outputs=[first_audio, sweep_paths],
            )

        with gr.Tab("Dataset Style Direction"):
            profile_path = gr.Textbox(
                value="/content/sa3_latent_lab/profiles/target_style.npz",
                label="Style profile .npz",
            )
            load_profile = gr.Button("Load Style Profile")
            profile_status = gr.Textbox(lines=3)
            load_profile.click(load_style_profile_file, inputs=profile_path, outputs=profile_status)

            style_prompt = gr.Textbox(value="a cinematic ambient loop with evolving texture", label="Prompt")
            style_duration = gr.Slider(2, 30, value=8, step=1, label="Duration")
            style_seed = gr.Number(value=321, precision=0, label="Seed")
            style_steps = gr.Slider(1, 50, value=8, step=1, label="Steps")
            style_cfg = gr.Slider(0, 12, value=1.0, step=0.1, label="CFG")
            style_alpha = gr.Slider(-2, 2, value=0.6, step=0.05, label="Style alpha")
            match_std = gr.Checkbox(value=True, label="Match dataset std")
            style_button = gr.Button("Generate With Dataset Style Direction")
            style_audio = gr.Audio(label="Styled audio")
            style_paths = gr.Textbox(lines=3, label="Saved paths")
            style_button.click(
                generate_with_style_profile,
                inputs=[style_prompt, style_duration, style_seed, style_steps, style_cfg, style_alpha, match_std],
                outputs=[style_audio, style_paths],
            )

    return demo


def _model():
    if STATE["model"] is None:
        raise gr.Error("Load SA3 Medium first.")
    return STATE["model"]


def _sa3() -> StableAudio3Adapter:
    if STATE["sa3"] is None:
        raise gr.Error("Load SA3 Medium first.")
    return STATE["sa3"]


def _index() -> LatentMemoryIndex:
    if STATE["index"] is None:
        raise gr.Error("Generate or load memory first.")
    return STATE["index"]


def _vectors() -> SteeringVectors:
    if STATE["vectors"] is None:
        raise gr.Error("Extract or load steering vectors first.")
    return STATE["vectors"]


def _profile():
    if STATE["profile"] is None:
        raise gr.Error("Load a style profile first.")
    return STATE["profile"]


def _memory_summary() -> str:
    items = STATE["items"]
    if not items:
        return "Memory is empty."
    lines = [f"{len(items)} items"]
    for item in items[-10:]:
        lines.append(f"{item.item_id}: frames={item.frames} prompt={item.prompt}")
    return "\n".join(lines)


def _save_audio(path: Path, audio, sample_rate: int) -> None:
    import torchaudio

    path.parent.mkdir(parents=True, exist_ok=True)
    audio = audio.float().clamp(-1, 1).cpu()
    torchaudio.save(str(path), audio[0], sample_rate)


def _apply_profile_to_latent_batch(latents, profile, *, alpha: float, match_std: bool):
    import numpy as np
    import torch

    arr = latents.detach().float().cpu().numpy()
    styled = []
    for latent in arr:
        styled_time_major = apply_profile_attraction(latent.T, profile, alpha=alpha, match_std=match_std)
        styled.append(styled_time_major.T)
    return torch.from_numpy(np.stack(styled).astype("float32")).to(device=latents.device, dtype=latents.dtype)


if __name__ == "__main__":
    build_demo().launch(share=True, debug=True)
