"""Generate ABC tunes for Irishman validation set using TunesFormer.

This script iterates over leadsheets in a JSON file, builds a conditional
prompt (control code + first N measures), and invokes the TunesFormer
`generate_abc` pipeline from `tunesformer/generate.py` to produce an
ABC tune for each item. Results are saved as individual `.abc` files.

Expected repository layout (relative to this script):
- tunesformer/
    - generate.py, utils.py, config.py, prompt.txt, weights.pth (auto-downloaded)
- data/
    - 02_preprocessed/irishman/validation_leadsheet.json
    - 04_generated/tunesformer/abc

Note: We reuse the original generation pipeline by temporarily writing
our constructed prompt into `tunesformer/prompt.txt` and calling
`generate.generate_abc(args)` with `num_tunes=1`. The function writes a
new file in `tunesformer/output_tunes/`; we read that file back and
store a per-sample copy under `output_dir`.
"""

from __future__ import annotations

import argparse
import json
import os
import re

# Import the tunesformer generation entrypoint
# We add tunesformer/ to sys.path so we can `import generate`.
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Tuple

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent  # go up from scripts/irishman_sripts/
TUNESFORMER_DIR = REPO_ROOT / "tunesformer"
OUTPUT_TUNES_DIR = TUNESFORMER_DIR / "output_tunes"

if str(TUNESFORMER_DIR) not in sys.path:
    sys.path.insert(0, str(TUNESFORMER_DIR))

import generate as tunes_generate  # type: ignore  # noqa: E402


@dataclass
class GenParams:
    max_patch: int = 128
    top_p: float = 0.8
    top_k: int = 200
    temperature: float = 0.8
    seed: int | None = None
    show_control_code: bool = False


BAR_SPLIT_RE = re.compile(r"(:\||::|\s\||\|\])")


def load_leadsheets(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prefixes_from_leadsheets(
    leadsheets: Iterable[dict], n_measures: int
) -> Generator[Tuple[str, str, str], None, None]:
    """Yield (id, control_code, prefix_abc) for each leadsheet."""
    for sheet in leadsheets:
        _id = str(sheet.get("id"))
        abc_notation: str = sheet.get("abc notation", "")
        control_code: str = sheet.get("control code", "")
        parts = BAR_SPLIT_RE.split(abc_notation)
        prefix = "".join(parts[: n_measures * 2])
        yield _id, control_code, prefix


def ensure_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _new_file_in_dir(before: set[str], directory: Path) -> Path | None:
    after = set(os.listdir(directory))
    new = after - before
    if not new:
        return None
    # choose the newest file among new additions
    newest = max((directory / n for n in new), key=lambda p: p.stat().st_mtime)
    return newest


def generate_with_prompt(prompt: str, p: GenParams) -> str:
    """Write prompt to tunesformer/prompt.txt, run a single generation, return text.

    Uses tunesformer/generate.py's `generate_abc` to perform the heavy lifting.
    """
    ensure_dirs(OUTPUT_TUNES_DIR)

    prompt_txt = TUNESFORMER_DIR / "prompt.txt"
    original_prompt = None
    if prompt_txt.exists():
        original_prompt = prompt_txt.read_text(encoding="utf-8")

    try:
        prompt_txt.write_text(prompt, encoding="utf-8")
        before = set(os.listdir(OUTPUT_TUNES_DIR))

        # Build args namespace matching generate.get_args expectations
        class _Args:
            pass

        args = _Args()
        args.num_tunes = 1
        args.max_patch = p.max_patch
        args.top_p = p.top_p
        args.top_k = p.top_k
        args.temperature = p.temperature
        args.seed = p.seed
        args.show_control_code = p.show_control_code

        tunes_generate.generate_abc(args)  # writes a file into output_tunes

        new_file = _new_file_in_dir(before, OUTPUT_TUNES_DIR)
        if new_file is None:
            raise RuntimeError("Generation produced no output file in output_tunes/")
        return new_file.read_text(encoding="utf-8")
    finally:
        # restore prompt.txt
        if original_prompt is not None:
            prompt_txt.write_text(original_prompt, encoding="utf-8")


def normalize_abc(text: str, tune_id: str) -> str:
    """Return ABC text with a leading `X:{tune_id}` header.

    If the generated text already contains an `X:` header, we strip it and
    replace with the requested id. We keep the remainder order intact.
    """
    # Split into lines, drop any leading blank lines
    lines = [ln for ln in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    # Remove an existing X: header if present
    if lines and lines[0].lstrip().startswith("X:"):
        lines.pop(0)
    body = "\n".join(lines).strip()
    return f"X:{tune_id}\n{body}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation_path",
        type=Path,
        default=REPO_ROOT / "data/02_preprocessed/irishman/validation_leadsheet.json",
        help="Path to validation_leadsheet.json",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "data/04_generated/tunesformer/abc",
        help="Directory to write per-sample ABC files",
    )
    parser.add_argument(
        "--n_conditional_measures",
        type=int,
        default=4,
        help="Number of measures from the ABC notation to prepend after the control code.",
    )
    parser.add_argument("--num_samples", type=int, default=1000)

    # Generation hyperparameters (mapped onto tunesformer generate.py)
    parser.add_argument("--max_patch", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--show_control_code",
        action="store_true",
        help="If set, keep S:/B:/E: control lines in outputs.",
    )

    args = parser.parse_args()

    ensure_dirs(args.output_dir)

    leadsheets = load_leadsheets(args.validation_path)
    prefix_iter = prefixes_from_leadsheets(leadsheets, args.n_conditional_measures)

    params = GenParams(
        max_patch=args.max_patch,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        seed=args.seed,
        show_control_code=args.show_control_code,
    )

    count = 0
    for sample_id, control_code, prefix in tqdm(prefix_iter, total=min(args.num_samples, len(leadsheets))):
        if count >= args.num_samples:
            break
        input_prompt = f"{control_code}{prefix}"
        abc_text = generate_with_prompt(input_prompt, params)
        norm_text = normalize_abc(abc_text, sample_id)
        out_path = args.output_dir / f"sample_{sample_id}.abc"
        out_path.write_text(norm_text, encoding="utf-8")
        count += 1


if __name__ == "__main__":
    main()
