#!/usr/bin/env python3
"""Create a stressless CMU phone lexicon for alignment transfer experiments.

The Label-Prior reference implementation removes lexical stress before phone
modeling.  The stock icefall LibriSpeech lexicon instead treats AA0/AA1/AA2 as
different tokens.  This script keeps the word/pronunciation entries but strips
trailing stress digits, removes spoken-noise words, and retains ``<UNK> SPN``.
The unused ``!SIL SIL`` entry is kept because icefall's phone language compiler
requires a silence symbol; deterministic transcript expansion never inserts it.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Set, Tuple

from icefall.lexicon import read_lexicon, write_lexicon


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/lang_phone/lexicon.txt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/lang_phone_nostress"),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = get_parser().parse_args()
    destination = args.output_dir / "lexicon.txt"
    if destination.exists() and not args.force:
        raise FileExistsError(f"{destination} exists; pass --force to replace it")

    excluded_words = {"<SPOKEN_NOISE>"}
    output: List[Tuple[str, List[str]]] = []
    seen: Set[Tuple[str, Tuple[str, ...]]] = set()
    phone_set: Set[str] = set()

    for word, pronunciation in read_lexicon(args.source):
        if word in excluded_words:
            continue
        phones = [re.sub(r"\d+$", "", phone) for phone in pronunciation]
        if "SIL" in phones and word != "!SIL":
            continue
        key = (word, tuple(phones))
        if key in seen:
            continue
        seen.add(key)
        output.append((word, phones))
        phone_set.update(phones)

    if ("<UNK>", ("SPN",)) not in seen:
        output.insert(0, ("<UNK>", ["SPN"]))
        phone_set.add("SPN")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_lexicon(destination, output)
    print(
        f"Wrote {len(output)} pronunciations and {len(phone_set)} nonblank "
        f"phones to {destination}: {' '.join(sorted(phone_set))}"
    )


if __name__ == "__main__":
    main()
