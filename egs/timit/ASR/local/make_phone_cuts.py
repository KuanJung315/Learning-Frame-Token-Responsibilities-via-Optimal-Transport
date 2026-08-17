#!/usr/bin/env python3
"""
lhotse>=1.x stores TIMIT phones in supervision.alignment['phone'] and keeps
supervision.text as the orthographic sentence.  For phone-level CTC / VI-OT
training we need supervision.text to BE the phone sequence, while preserving
alignment['phone'] (start/duration) as the gold alignment for analysis.

This script, for each TIMIT cut set:
  * sets supervision.text = " ".join(phone symbols from alignment['phone'])
  * leaves alignment['phone'] (and 'word') intact for alignment evaluation
  * writes data/fbank/timit_cuts_{PART}_phone.jsonl.gz

It also writes data/lang_phone/lexicon.txt (identity phone lexicon) from the
TRAIN phone inventory, so prepare_lang.py can build tokens.txt / L.pt.
"""
import argparse
import logging
from pathlib import Path

from lhotse import CutSet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fbank-dir", type=Path, default=Path("data/fbank"))
    p.add_argument("--lang-dir", type=Path, default=Path("data/lang_phone"))
    p.add_argument("--parts", nargs="+", default=["TRAIN", "DEV", "TEST"])
    return p.parse_args()


def phones_of(sup) -> list:
    if not sup.alignment or "phone" not in sup.alignment:
        return []
    return [a.symbol for a in sup.alignment["phone"] if a.symbol.strip()]


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.lang_dir.mkdir(parents=True, exist_ok=True)

    phone_set = set()
    for part in args.parts:
        src = args.fbank_dir / f"timit_cuts_{part}.jsonl.gz"
        dst = args.fbank_dir / f"timit_cuts_{part}_phone.jsonl.gz"
        cuts = CutSet.from_file(src)

        def _relabel(cut):
            for sup in cut.supervisions:
                ph = phones_of(sup)
                sup.text = " ".join(ph)
            return cut

        cuts = cuts.map(_relabel)
        cuts.to_file(dst)

        # recompute phone inventory (from TRAIN only) + log stats
        n = 0
        for cut in CutSet.from_file(dst):
            for sup in cut.supervisions:
                toks = sup.text.split()
                n = max(n, len(toks))
                if part == "TRAIN":
                    phone_set.update(toks)
        logging.info(f"{part}: wrote {dst} (max phones/utt={n})")

    lexicon = args.lang_dir / "lexicon.txt"
    with open(lexicon, "w") as f:
        for ph in sorted(phone_set):
            f.write(f"{ph} {ph}\n")
        # OOV entry (harmless; TIMIT phone set is closed)
        f.write("<UNK> <UNK>\n")
    logging.info(f"Wrote {lexicon} with {len(phone_set)} phones (+<UNK>)")


if __name__ == "__main__":
    main()
