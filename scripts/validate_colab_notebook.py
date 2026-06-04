from __future__ import annotations

import argparse
import json
import os
import tempfile
import traceback
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute SA3 Native Lab notebook code cells in a local-safe configuration."
    )
    parser.add_argument(
        "--notebook",
        default="colab/sa3_same_native_experimental_modes.ipynb",
        help="Notebook to validate.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Local experiment work dir. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--with-models",
        action="store_true",
        help="Allow the model-loading cell to load HF weights.",
    )
    parser.add_argument(
        "--with-smoke",
        action="store_true",
        help="Run the SA3 smoke generation cell. Implies --with-models.",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip the setup/verification cell.",
    )
    args = parser.parse_args()

    notebook_path = Path(args.notebook)
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    work_dir_ctx = None
    work_dir = Path(args.work_dir) if args.work_dir else None
    if work_dir is None:
        work_dir_ctx = tempfile.TemporaryDirectory(prefix="sa3_native_notebook_")
        work_dir = Path(work_dir_ctx.name)
    work_dir.mkdir(parents=True, exist_ok=True)

    has_hf_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    env_defaults = {
        "SA3_PROJECT_DIR": str(Path.cwd()),
        "SA3_WORK_DIR": str(work_dir),
        "SA3_SETUP_MARKER": str(work_dir / ".sa3_native_lab_setup_complete"),
        "SA3_INSTALL_REPO": "0",
        "SA3_INSTALL_TORCH_CU126": "0",
        "SA3_INSTALL_FLASH_ATTN": "0",
        "SA3_PIN_NUMPY": "0",
        "SA3_REMOVE_SCIPY_SKLEARN": "0",
        "SA3_REMOVE_TORCHVISION": "0",
        "SA3_RESTART_AFTER_INSTALL": "0",
        "SA3_HF_LOGIN": "1" if (args.with_models or args.with_smoke) and has_hf_token else "0",
        "SA3_LOAD_MODELS": "1" if (args.with_models or args.with_smoke) else "0",
        "SA3_RUN_SMOKE_TEST": "1" if args.with_smoke else "0",
    }
    old_env = {key: os.environ.get(key) for key in env_defaults}
    os.environ.update(env_defaults)

    namespace: dict[str, object] = {
        "__name__": "__sa3_notebook_validation__",
        "__file__": str(notebook_path),
    }
    passed: list[int] = []
    failed: list[tuple[int, str, str]] = []

    try:
        for index, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            title = _cell_title(source, index)
            if args.skip_setup and _is_setup_cell(title):
                continue
            source = _local_validation_source(source, title, args=args, work_dir=work_dir)
            try:
                exec(compile(source, f"{notebook_path}:cell-{index}", "exec"), namespace)
            except Exception:
                failed.append((index, title, traceback.format_exc()))
                break
            passed.append(index)
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if work_dir_ctx is not None:
            work_dir_ctx.cleanup()

    print(f"validated_cells={len(passed)}")
    if failed:
        index, title, tb = failed[0]
        print(f"failed_cell={index} {title}")
        print(tb)
        raise SystemExit(1)
    print("notebook validation passed")


def _cell_title(source: str, index: int) -> str:
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("# @title"):
            return stripped
    return f"cell {index}"


def _is_setup_cell(title: str) -> bool:
    return "setup" in title.lower() or "install" in title.lower() or "auth" in title.lower()


def _local_validation_source(source: str, title: str, *, args: argparse.Namespace, work_dir: Path) -> str:
    lower_title = title.lower()
    source = source.replace(
        'WORK_DIR = Path("/content/sa3_same_native_experiments")',
        f"WORK_DIR = Path({str(work_dir)!r})",
    )
    if "hugging face login and model loading" in lower_title:
        load_models = bool(args.with_models or args.with_smoke)
        source = source.replace("LOAD_MODELS = True", f"LOAD_MODELS = {load_models!r}")
        source = source.replace("HF_LOGIN = True", f"HF_LOGIN = {load_models!r}")
    if "smoke test" in lower_title:
        source = source.replace("RUN_SMOKE_TEST = True", f"RUN_SMOKE_TEST = {bool(args.with_smoke)!r}")
    return source


if __name__ == "__main__":
    main()
