#!/usr/bin/env python
"""
build_4part_dataset.py
----------------------

Create a synthetic 4-part data-set from monophonic melody files that use the
Brand24 “text-MIDI” encoding.

Assumptions
-----------
* Each source file contains at least one `/0` (melody) part; we ignore /1 etc.
* A time-signature appears once near the top of the file as
      time_signature_M/N           (e.g. time_signature_6/8)
* Every measure ends with “… / oXX |” where “oXX” (e.g. o36) is an **offset
  token** that uniquely corresponds to that time-signature’s bar length.
* All files that have the same `time_signature_M/N` also share the same offset
  token; if not, the script skips the offending file and warns you.
* Synthetic pieces are written to `<output_dir>/<signature>_mix_###.txt`
  preserving the original tokens exactly (no re-quantisation).

Usage
-----
    python sample_synthetic_4_parts_from_irishman.py \
        --input_dir  /path/to/melody_files \
        --output_dir /path/to/synthetic_corpus \
        --pieces_per_sig 500 \
        --seed 42
"""

from __future__ import annotations

import argparse
import pathlib
import random
import re
import sys
from collections import defaultdict
from typing import List, Tuple

TIME_SIG_RE = re.compile(r"time_signature_(\d+/\d+)")
BAR_SPLIT_RE = re.compile(r"\|")  # split on barlines
OFFSET_RE = re.compile(r"/\s*(o\d+)\s*$")  # last “/ oXX” before “|”
MELODY_RE = re.compile(r"/0(.*?)(?=/\d|\s/\s*o\d+\s*$)", re.S)

###############################################################################
# Parsing helpers
###############################################################################


def parse_file(path: pathlib.Path) -> Tuple[str, str, List[str]]:
    """Return (time_signature, offset_token, [melody_bars])."""
    txt = path.read_text(encoding="utf-8")

    # 1. time signature -------------------------------------------------------
    m = TIME_SIG_RE.search(txt)
    if not m:
        raise ValueError(f"{path.name}: no time-signature found")
    tsig = m.group(1)  # e.g. '6/8'

    # 2. split into measures ---------------------------------------------------
    bars = [b.strip() for b in BAR_SPLIT_RE.split(txt) if b.strip()]
    if not bars:
        raise ValueError(f"{path.name}: empty file?")

    # 3. capture offset token (use first bar) ----------------------------------
    mo = OFFSET_RE.search(bars[0])
    if not mo:
        raise ValueError(f"{path.name}: no offset token found")
    offset = mo.group(1)  # e.g. 'o36'

    # 4. grab /0 melody from every bar ----------------------------------------
    melody_bars: List[str] = []
    for bar in bars:
        mm = MELODY_RE.search(bar)
        if not mm:
            raise ValueError(f"{path.name}: bar lacks /0 melody\n{bar}")
        melody_bars.append(mm.group(0).strip())  # keep the '/0 …' tokens

    return tsig, offset, melody_bars


###############################################################################
# Synthetic builder
###############################################################################


def build_one_piece(
    samples: List[Tuple[str, str, List[str]]], piece_idx: int, out_dir: pathlib.Path, tsig: str, offset: str
) -> None:
    """Write a single 4-part piece assembled from the given 4 melodies."""
    # ---------------------------------------- choose 4 distinct melodies
    m1, m2, m3, m4 = random.sample(samples, k=4)

    # ---------------------------------------- synchronise length
    n_bars = min(len(m1[2]), len(m2[2]), len(m3[2]), len(m4[2]))

    # ---------------------------------------- header (keep it minimal)
    header = f"/0 time_signature_{tsig}\n"

    # ---------------------------------------- build bar-by-bar
    full_piece: List[str] = [header]
    for i in range(n_bars):
        bar_tokens = [
            m1[2][i].replace("/0", "/0", 1),
            m2[2][i].replace("/0", "/1", 1),
            m3[2][i].replace("/0", "/2", 1),
            m4[2][i].replace("/0", "/3", 1),
            f"/ {offset} |",
        ]
        full_piece.append(" ".join(bar_tokens))

    # ---------------------------------------- write to disk
    out_path = out_dir / f"{tsig.replace('/', '-')}_mix_{piece_idx:04d}.txt"
    out_path.write_text("\n".join(full_piece), encoding="utf-8")


def create_dataset(
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    pieces_per_sig: int,
    seed: int = 0,
) -> None:

    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- collect
    pools: dict[str, List[Tuple[str, str, List[str]]]] = defaultdict(list)
    offsets: dict[str, str] = {}  # tsig -> offset token

    for f in input_dir.glob("*"):
        if not f.is_file():
            continue
        try:
            tsig, offset, melody = parse_file(f)
        except ValueError as e:
            print(f"Skip  {f.name}: {e}", file=sys.stderr)
            continue

        # ensure consistent offset per time-sig
        if tsig in offsets and offsets[tsig] != offset:
            print(f"Skip  {f.name}: mismatched offset for {tsig}", file=sys.stderr)
            continue
        offsets[tsig] = offset
        pools[tsig].append((tsig, offset, melody))

    if not pools:
        sys.exit("No valid files found.")

    # ---------------------------------------------------------------- synthesize
    for tsig, melodies in pools.items():
        if len(melodies) < 4:
            print(f"Time-sig {tsig}: only {len(melodies)} melodies – skipped.")
            continue

        for i in range(pieces_per_sig):
            build_one_piece(
                melodies,
                piece_idx=i,
                out_dir=output_dir,
                tsig=tsig,
                offset=offsets[tsig],
            )
        print(f"{tsig}: wrote {pieces_per_sig} mixes " f"(from {len(melodies)} source melodies).")


###############################################################################
# CLI
###############################################################################


def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic 4-part corpus.")
    p.add_argument("--input_dir", required=True, type=pathlib.Path, help="Folder with monophonic text-MIDI files.")
    p.add_argument("--output_dir", required=True, type=pathlib.Path, help="Destination folder for synthetic pieces.")
    p.add_argument(
        "--pieces_per_sig", type=int, default=500, help="How many 4-part mixes to create per time-signature."
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed.")
    args = p.parse_args()
    create_dataset(args.input_dir, args.output_dir, pieces_per_sig=args.pieces_per_sig, seed=args.seed)


if __name__ == "__main__":
    _cli()
