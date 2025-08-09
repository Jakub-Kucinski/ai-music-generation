#!/usr/bin/env python
"""
build_4part_dataset_noreuse.py
------------------------------

Create a synthetic 4-part data-set from monophonic melody files that use the
“text-MIDI” encoding, using each source melody **at most once**.
Produces as many 4-part mixes as possible from the provided tunes
(= floor(len(pool)/4) per time-signature pool).

Formatting rules (same as original):
- Emit clef_* / key_signature_* / normalized time_signature_M/N only once (bar 0 per voice).
- No meta tokens in subsequent bars.
- Write each piece on a single line; bars separated by " / oXX |".

Usage
-----
    python build_4part_dataset_noreuse.py \
        --input_dir  /path/to/melody_files \
        --output_dir /path/to/synthetic_corpus \
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

from tqdm import tqdm

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
            continue
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


def _write_piece(
    parts: List[Tuple[str, str, List[str]]],
    piece_idx: int,
    out_dir: pathlib.Path,
    tsig: str,
    offset: str,
) -> None:
    """
    Write a single 4-part piece assembled from exactly 4 distinct melodies.
    parts: list of 4 tuples (tsig, offset, melody_bars)
    """
    assert len(parts) == 4, "expected exactly 4 parts"
    # Synchronise by the shortest melody
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
                bar_voice_chunks.append(f"/{voice_idx} {content}".strip())

        # Bar delimiter with offset token
        bar_voice_chunks.append(f"/ {offset} |")
        out_tokens.append(" ".join(bar_voice_chunks))

    # Single-line piece; global index (no zero padding unless desired)
    out_path = out_dir / f"file_{piece_idx}.txt"
    out_path.write_text(" ".join(out_tokens), encoding="utf-8")


def create_dataset_noreuse(
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    seed: int = 0,
) -> None:
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect pools per time signature and enforce consistent offset per tsig
    pools: dict[str, List[Tuple[str, str, List[str]]]] = defaultdict(list)
    offsets: dict[str, str] = {}  # tsig -> offset token

    files = [f for f in input_dir.glob("*") if f.is_file()]
    if not files:
        sys.exit("No files found in input_dir.")

    for f in tqdm(files, desc="Parsing files", unit="file"):
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
        sys.exit("No valid files found after parsing.")

    # Precompute disjoint quartets per tsig and total pieces
    groups_per_tsig: dict[str, List[List[Tuple[str, str, List[str]]]]] = {}
    orig_counts: dict[str, int] = {}
    leftovers: dict[str, int] = {}

    total_pieces = 0
    for tsig, melodies in pools.items():
        orig_counts[tsig] = len(melodies)
        if len(melodies) < 4:
            print(f"Time-sig {tsig}: only {len(melodies)} melodies - skipped.")
            groups_per_tsig[tsig] = []
            leftovers[tsig] = len(melodies)
            continue

        melodies_shuffled = melodies[:]
        random.shuffle(melodies_shuffled)
        n_complete = len(melodies_shuffled) // 4
        groups = [melodies_shuffled[i * 4 : (i + 1) * 4] for i in range(n_complete)]
        leftover = len(melodies_shuffled) - n_complete * 4

        groups_per_tsig[tsig] = groups
        leftovers[tsig] = leftover
        total_pieces += n_complete

    if total_pieces == 0:
        sys.exit("No mixes written (insufficient melodies per time-signature).")

    # Write all pieces with a global progress bar
    piece_idx = 0
    with tqdm(total=total_pieces, desc="Writing pieces", unit="file") as pbar:
        for tsig, groups in groups_per_tsig.items():
            for quartet in groups:
                _write_piece(quartet, piece_idx=piece_idx, out_dir=output_dir, tsig=tsig, offset=offsets[tsig])
                piece_idx += 1
                pbar.update(1)

    # Final summary
    for tsig, groups in groups_per_tsig.items():
        print(
            f"{tsig}: wrote {len(groups)} mixes (from {orig_counts.get(tsig, 0)} source melodies; leftover {leftovers.get(tsig, 0)})."
        )
    print(f"Total written: {piece_idx} files.")


def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic 4-part corpus without reusing source melodies.")
    p.add_argument("--input_dir", required=True, type=pathlib.Path, help="Folder with monophonic text-MIDI files.")
    p.add_argument("--output_dir", required=True, type=pathlib.Path, help="Destination folder for synthetic pieces.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for deterministic grouping.")
    args = p.parse_args()
    create_dataset_noreuse(args.input_dir, args.output_dir, seed=args.seed)


if __name__ == "__main__":
    _cli()
