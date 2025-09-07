#!/usr/bin/env python
"""
build_4part_dataset_noreuse.py
------------------------------

Create a synthetic 4-part data-set from monophonic melody files that use the
“text-MIDI” encoding, using each source melody **at most once**.

**Now pools by (time signature, key signature)**, so every 4-part mix has
a consistent time signature and key signature.

Produces as many 4-part mixes as possible from the provided tunes
(= floor(len(pool)/4) per (M/N, key) pool).

Formatting rules (same as original, with explicit key normalization):
- Emit clef_* / key_signature_* / normalized time_signature_M/N only once (bar 0 per voice).
- No meta tokens in subsequent bars.
- Write each piece on a single line; bars separated by " / oXX |".

Usage
-----
Single dataset (no reuse within):
    python build_4part_dataset_noreuse.py \
        --input_dir  /path/to/melody_files \
        --output_dir /path/to/synthetic_corpus \
        --seed 42

Multiple independent datasets (no reuse within each; reuse allowed across):
    python build_4part_dataset_noreuse.py \
        --input_dir  /path/to/melody_files \
        --output_dir /path/to/corpora_root \
        --seed 42 \
        --num_datasets 5
"""

from __future__ import annotations

import argparse
import pathlib
import random
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

from tqdm import tqdm

TIME_SIG_RE = re.compile(r"time_signature_(\d+/\d+)")
KEY_SIG_RE = re.compile(r"key_signature_(-?\d+)")
BAR_SPLIT_RE = re.compile(r"\|")  # split on barlines
OFFSET_RE = re.compile(r"/\s*(o\d+)\s*$")  # last “/ oXX” before “|”
MELODY_RE = re.compile(r"/0(.*?)(?=\s+/\d|\s/\s*o\d+\s*$)", re.S)
FIRST_OFFSET = re.compile(r"\bo\d+\b")  # first offset within a /0 bar


def parse_file(path: pathlib.Path) -> Tuple[str, str, str, List[str]]:
    """Return (time_signature, key_signature, offset_token, [raw_melody_bars_as_/0_chunks])."""
    txt = path.read_text(encoding="utf-8")

    # 1) time signature --------------------------------------------------------
    m = TIME_SIG_RE.search(txt)
    if not m:
        raise ValueError(f"{path.name}: no time-signature found (need M/N form)")
    tsig = m.group(1)  # e.g. '6/8'

    # 2) key signature ---------------------------------------------------------
    k = KEY_SIG_RE.search(txt)
    if not k:
        raise ValueError(f"{path.name}: no key-signature found (need key_signature_K)")
    ksig = k.group(1)  # e.g. '3' or '-2'

    # 3) split into measures ---------------------------------------------------
    bars = [b.strip() for b in BAR_SPLIT_RE.split(txt) if b.strip()]
    if not bars:
        raise ValueError(f"{path.name}: empty file?")

    # 4) capture offset token (use first bar) ---------------------------------
    mo = OFFSET_RE.search(bars[0])
    if not mo:
        raise ValueError(f"{path.name}: no offset token found in bar 1")
    offset = mo.group(1)  # e.g. 'o24'

    # 5) grab /0 melody from every bar ----------------------------------------
    melody_bars: List[str] = []
    for bar in bars:
        mm = MELODY_RE.search(bar)
        if not mm:
            continue
        # Keep the raw '/0 …' substring; we will clean meta later.
        melody_bars.append(mm.group(0).strip())

    return tsig, ksig, offset, melody_bars


def _split_meta_and_content(bar_chunk: str) -> Tuple[str, str]:
    """
    Given a '/0 …' chunk, return (meta_clef_only, content) where:
      - meta_clef_only: ONLY clef_* tokens (time_signature_* and key_signature_* removed)
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

    # Filter meta: keep only clef_*; drop key_signature_* and time_signature_*
    toks = meta.split()
    meta_clef_only = " ".join(t for t in toks if t.startswith("clef_"))
    return meta_clef_only, content


def _write_piece(
    parts: List[Tuple[str, str, str, List[str]]],
    piece_idx: int,
    out_dir: pathlib.Path,
    tsig: str,
    ksig: str,
    offset: str,
) -> None:
    """
    Write a single 4-part piece assembled from exactly 4 distinct melodies.
    parts: list of 4 tuples (tsig, ksig, offset, melody_bars)
    """
    assert len(parts) == 4, "expected exactly 4 parts"
    # Synchronise by the shortest melody
    n_bars = min(len(p[3]) for p in parts)

    out_tokens: List[str] = []
    for i in range(n_bars):
        bar_voice_chunks: List[str] = []
        for voice_idx, (_, _, _, melody_bars) in enumerate(parts):
            meta_clef, content = _split_meta_and_content(melody_bars[i])
            if i == 0:
                # Emit clef_* (if present) + normalized key_signature_{ksig} + time_signature_{tsig}
                prefix = f"/{voice_idx}"
                if meta_clef:
                    bar_voice_chunks.append(
                        f"{prefix} {meta_clef} key_signature_{ksig} time_signature_{tsig} {content}".strip()
                    )
                else:
                    bar_voice_chunks.append(f"{prefix} key_signature_{ksig} time_signature_{tsig} {content}".strip())
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

    # Collect pools per (time signature, key signature) and enforce consistent offset per pair
    pools: Dict[Tuple[str, str], List[Tuple[str, str, str, List[str]]]] = defaultdict(list)
    offsets: Dict[Tuple[str, str], str] = {}  # (tsig, ksig) -> offset token

    files = [f for f in input_dir.glob("*") if f.is_file()]
    if not files:
        sys.exit("No files found in input_dir.")

    for f in tqdm(files, desc="Parsing files", unit="file"):
        try:
            tsig, ksig, offset, melody = parse_file(f)
        except ValueError as e:
            print(f"Skip  {f.name}: {e}", file=sys.stderr)
            continue

        key = (tsig, ksig)
        if key in offsets and offsets[key] != offset:
            print(f"Skip  {f.name}: mismatched offset for {tsig}, key {ksig}", file=sys.stderr)
            continue
        offsets[key] = offset
        pools[key].append((tsig, ksig, offset, melody))

    if not pools:
        sys.exit("No valid files found after parsing.")

    # Precompute disjoint quartets per (tsig, ksig) and total pieces
    groups_per_pair: Dict[Tuple[str, str], List[List[Tuple[str, str, str, List[str]]]]] = {}
    orig_counts: Dict[Tuple[str, str], int] = {}
    leftovers: Dict[Tuple[str, str], int] = {}

    total_pieces = 0
    for pair, melodies in pools.items():
        orig_counts[pair] = len(melodies)
        if len(melodies) < 4:
            tsig, ksig = pair
            print(f"Pool {tsig}, key {ksig}: only {len(melodies)} melodies - skipped.")
            groups_per_pair[pair] = []
            leftovers[pair] = len(melodies)
            continue

        melodies_shuffled = melodies[:]
        random.shuffle(melodies_shuffled)
        n_complete = len(melodies_shuffled) // 4
        groups = [melodies_shuffled[i * 4 : (i + 1) * 4] for i in range(n_complete)]
        leftover = len(melodies_shuffled) - n_complete * 4

        groups_per_pair[pair] = groups
        leftovers[pair] = leftover
        total_pieces += n_complete

    if total_pieces == 0:
        sys.exit("No mixes written (insufficient melodies per time-signature/key-signature).")

    # Write all pieces with a global progress bar
    piece_idx = 0
    with tqdm(total=total_pieces, desc="Writing pieces", unit="file") as pbar:
        for (tsig, ksig), groups in groups_per_pair.items():
            for quartet in groups:
                _write_piece(
                    quartet,
                    piece_idx=piece_idx,
                    out_dir=output_dir,
                    tsig=tsig,
                    ksig=ksig,
                    offset=offsets[(tsig, ksig)],
                )
                piece_idx += 1
                pbar.update(1)

    # Final summary
    for (tsig, ksig), groups in groups_per_pair.items():
        print(
            f"{tsig}, key {ksig}: wrote {len(groups)} mixes "
            f"(from {orig_counts.get((tsig, ksig), 0)} source melodies; "
            f"leftover {leftovers.get((tsig, ksig), 0)})."
        )
    print(f"Total written: {piece_idx} files.")


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Generate synthetic 4-part corpus without reusing source melodies (pooled by time & key signature)."
    )
    p.add_argument("--input_dir", required=True, type=pathlib.Path, help="Folder with monophonic text-MIDI files.")
    p.add_argument(
        "--output_dir",
        required=True,
        type=pathlib.Path,
        help="Destination folder. If --num_datasets>1, subfolders dataset_000, dataset_001, ... are created here.",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for deterministic grouping.")
    p.add_argument(
        "--num_datasets",
        type=int,
        default=1,
        help="How many independent datasets to write. Melodies can repeat across datasets, but not inside one.",
    )
    args = p.parse_args()
    if args.num_datasets <= 1:
        create_dataset_noreuse(args.input_dir, args.output_dir, seed=args.seed)
        return
    # Create multiple datasets: reuse is allowed across datasets but not within each.
    for i in tqdm(range(args.num_datasets), desc="Creating datasets", unit="dataset"):
        out_i = args.output_dir / f"dataset_{i:04d}"
        seed_i = (args.seed or 0) + i
        print(f"\n=== Dataset {i+1}/{args.num_datasets} → {out_i} (seed={seed_i}) ===")
        create_dataset_noreuse(args.input_dir, out_i, seed=seed_i)


if __name__ == "__main__":
    _cli()
