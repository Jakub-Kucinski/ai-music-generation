#!/usr/bin/env python
"""
build_4part_dataset_no_resample.py
----------------------------------

Create *all possible* 4-part pieces from monophonic melody files without
re-using any melody more than once per time-signature.

See the header of the earlier `build_4part_dataset.py` for encoding
assumptions; they remain unchanged.

Usage
-----
    python build_4part_dataset_no_resample.py \
        --input_dir  /path/to/melody_files \
        --output_dir /path/to/synthetic_corpus \
        --seed 123
"""

from __future__ import annotations

import argparse
import pathlib
import random
import re
import sys
from collections import defaultdict
from typing import List, Tuple

###############################################################################
# Regular-expressions identical to the first script
###############################################################################
TIME_SIG_RE = re.compile(r"time_signature_(\d+/\d+)")
BAR_SPLIT_RE = re.compile(r"\|")
OFFSET_RE = re.compile(r"/\s*(o\d+)\s*$")
MELODY_RE = re.compile(r"/0(.*?)(?=/\d|\s/\s*o\d+\s*$)", re.S)


###############################################################################
# Parsing helper (unchanged)
###############################################################################
def parse_file(path: pathlib.Path) -> Tuple[str, str, List[str]]:
    txt = path.read_text(encoding="utf-8")

    tsig_m = TIME_SIG_RE.search(txt)
    if not tsig_m:
        raise ValueError(f"{path.name}: missing time-signature")
    tsig = tsig_m.group(1)

    bars = [b.strip() for b in BAR_SPLIT_RE.split(txt) if b.strip()]
    if not bars:
        raise ValueError(f"{path.name}: empty file?")

    off_m = OFFSET_RE.search(bars[0])
    if not off_m:
        raise ValueError(f"{path.name}: missing offset token")
    offset = off_m.group(1)

    melody_bars: List[str] = []
    for bar in bars:
        m = MELODY_RE.search(bar)
        if not m:
            raise ValueError(f"{path.name}: bar without /0")
        melody_bars.append(m.group(0).strip())

    return tsig, offset, melody_bars


###############################################################################
# Piece builder
###############################################################################
def write_piece(melodies: List[List[str]], out_path: pathlib.Path, tsig: str, offset: str) -> None:
    """
    `melodies` is a list of four *bar-lists* already sampled without reuse.
    """
    # keep bars aligned to the shortest voice
    n_bars = min(len(b) for b in melodies)

    parts = ("/0", "/1", "/2", "/3")
    lines: List[str] = [f"/0 time_signature_{tsig}\n"]

    for i in range(n_bars):
        bar_tokens = [melodies[v][i].replace("/0", parts[v], 1) for v in range(4)]
        bar_tokens.append(f"/ {offset} |")
        lines.append(" ".join(bar_tokens))

    out_path.write_text("\n".join(lines), encoding="utf-8")


###############################################################################
# Dataset creator
###############################################################################
def create_dataset(
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    seed: int = 0,
) -> None:
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    pools: dict[str, List[List[str]]] = defaultdict(list)  # tsig -> list of melodies
    offsets: dict[str, str] = {}  # tsig -> offset token

    for f in input_dir.glob("*"):
        if not f.is_file():
            continue
        try:
            tsig, offset, melody = parse_file(f)
        except ValueError as e:
            print(f"Skip {f.name}: {e}", file=sys.stderr)
            continue

        if tsig in offsets and offsets[tsig] != offset:
            print(f"Skip {f.name}: mismatched offset for {tsig}", file=sys.stderr)
            continue
        offsets[tsig] = offset
        pools[tsig].append(melody)

    if not pools:
        sys.exit("No valid melodies found.")

    # ---------------------------------------------------------------- synthesize
    for tsig, melodies in pools.items():
        random.shuffle(melodies)  # reproducible via seed
        full_groups = len(melodies) // 4
        if full_groups == 0:
            print(f"{tsig}: only {len(melodies)} melodies – skipped.")
            continue

        for idx in range(full_groups):
            group = melodies[idx * 4 : (idx + 1) * 4]
            fname = f"{tsig.replace('/', '-')}_mix_{idx:04d}.txt"
            write_piece(group, output_dir / fname, tsig=tsig, offset=offsets[tsig])
        print(f"{tsig}: wrote {full_groups} mixes " f"(used {full_groups*4}/{len(melodies)} melodies).")


###############################################################################
# CLI
###############################################################################
def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate maximal 4-part corpus without melody reuse.")
    p.add_argument("--input_dir", required=True, type=pathlib.Path)
    p.add_argument("--output_dir", required=True, type=pathlib.Path)
    p.add_argument("--seed", type=int, default=0, help="RNG seed.")
    args = p.parse_args()
    create_dataset(args.input_dir, args.output_dir, seed=args.seed)


if __name__ == "__main__":
    _cli()
