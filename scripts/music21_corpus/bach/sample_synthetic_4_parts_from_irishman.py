#!/usr/bin/env python
"""
build_4part_dataset.py
----------------------

Create a synthetic 4-part data-set from monophonic melody files that use the
Brand24 “text-MIDI” encoding.

Compared to the original:
- Emits clef/key/time_signature only once (in bar 0 per voice).
- Normalizes time-signature to the parsed M/N and injects it only in bar 0.
- Strips any meta tokens from subsequent bars.
- Writes each piece on a single line (bars separated by " / oXX |"), matching the target format.

Usage
-----
    python build_4part_dataset.py \
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
MELODY_RE = re.compile(r"/0(.*?)(?=\s+/\d|\s/\s*o\d+\s*$)", re.S)
FIRST_OFFSET = re.compile(r"\bo\d+\b")  # first offset within a /0 bar


def parse_file(path: pathlib.Path) -> Tuple[str, str, List[str]]:
    """Return (time_signature, offset_token, [raw_melody_bars_as_/0_chunks])."""
    txt = path.read_text(encoding="utf-8")

    # 1) time signature --------------------------------------------------------
    m = TIME_SIG_RE.search(txt)
    if not m:
        raise ValueError(f"{path.name}: no time-signature found (need M/N form)")
    tsig = m.group(1)  # e.g. '2/4'

    # 2) split into measures ---------------------------------------------------
    bars = [b.strip() for b in BAR_SPLIT_RE.split(txt) if b.strip()]
    if not bars:
        raise ValueError(f"{path.name}: empty file?")

    # 3) capture offset token (use first bar) ---------------------------------
    mo = OFFSET_RE.search(bars[0])
    if not mo:
        raise ValueError(f"{path.name}: no offset token found in bar 1")
    offset = mo.group(1)  # e.g. 'o24'

    # 4) grab /0 melody from every bar ----------------------------------------
    melody_bars: List[str] = []
    for bar in bars:
        mm = MELODY_RE.search(bar)
        if not mm:
            raise ValueError(f"{path.name}: bar lacks /0 melody\n{bar}")
        # Keep the raw '/0 …' substring; we will clean meta later.
        melody_bars.append(mm.group(0).strip())

    return tsig, offset, melody_bars


def _split_meta_and_content(bar_chunk: str) -> Tuple[str, str]:
    """
    Given a '/0 …' chunk, return (meta, content) where:
      - meta: only clef_* and key_signature_* tokens (time_signature_* removed)
      - content: starts from first 'oNN' (offset) token onward (no leading '/0')
    """
    assert bar_chunk.startswith("/0"), f"unexpected bar chunk start: {bar_chunk[:10]}"
    s = bar_chunk[2:].lstrip()  # drop '/0'
    mo = FIRST_OFFSET.search(s)
    if not mo:
        # If there's no 'oNN', treat whole thing as meta and content empty.
        meta = s.strip()
        content = ""
    else:
        meta = s[: mo.start()].strip()
        content = s[mo.start() :].strip()

    # Filter meta: keep only clef_* and key_signature_*; drop time_signature_*
    toks = meta.split()
    meta_filtered = " ".join(t for t in toks if t.startswith("clef_") or t.startswith("key_signature_"))
    return meta_filtered, content


def build_one_piece(
    samples: List[Tuple[str, str, List[str]]],
    piece_idx: int,
    out_dir: pathlib.Path,
    tsig: str,
    offset: str,
) -> None:
    """Write a single 4-part piece assembled from 4 distinct melodies."""
    # Choose 4 unique source melodies
    m1, m2, m3, m4 = random.sample(samples, k=4)
    parts = [m1, m2, m3, m4]

    # Synchronise length
    n_bars = min(len(p[2]) for p in parts)

    out_tokens: List[str] = []

    for i in range(n_bars):
        bar_voice_chunks: List[str] = []
        for voice_idx, (_, _, melody_bars) in enumerate(parts):
            meta, content = _split_meta_and_content(melody_bars[i])

            if i == 0:
                # Emit meta once (bar 0) + normalized time_signature_{tsig}
                if meta:
                    bar_voice_chunks.append(f"/{voice_idx} {meta} time_signature_{tsig} {content}".strip())
                else:
                    bar_voice_chunks.append(f"/{voice_idx} time_signature_{tsig} {content}".strip())
            else:
                # Subsequent bars: no meta, just content
                bar_voice_chunks.append(f"/{voice_idx} {content}".strip())

        # Bar delimiter with offset token
        bar_voice_chunks.append(f"/ {offset} |")
        out_tokens.append(" ".join(bar_voice_chunks))

    # Write as a single line (matching the provided proper sample)
    out_path = out_dir / f"{tsig.replace('/', '-')}_mix_{piece_idx:05d}.txt"
    out_path.write_text(" ".join(out_tokens), encoding="utf-8")


def create_dataset(
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    pieces_per_sig: int,
    seed: int = 0,
) -> None:
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect pools per time signature and enforce consistent offset per tsig
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

        if tsig in offsets and offsets[tsig] != offset:
            print(f"Skip  {f.name}: mismatched offset for {tsig}", file=sys.stderr)
            continue
        offsets[tsig] = offset
        pools[tsig].append((tsig, offset, melody))

    if not pools:
        sys.exit("No valid files found.")

    # Synthesize pieces
    for tsig, melodies in pools.items():
        if len(melodies) < 4:
            print(f"Time-sig {tsig}: only {len(melodies)} melodies - skipped.")
            continue

        for i in range(pieces_per_sig):
            build_one_piece(
                melodies,
                piece_idx=i,
                out_dir=output_dir,
                tsig=tsig,
                offset=offsets[tsig],
            )
        print(f"{tsig}: wrote {pieces_per_sig} mixes (from {len(melodies)} source melodies).")


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
