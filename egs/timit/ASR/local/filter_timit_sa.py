#!/usr/bin/env python3
"""Create canonical TIMIT evaluation manifests with SA sentences removed."""

from pathlib import Path

from lhotse import CutSet


def main() -> None:
    root = Path("data/fbank")
    expected = {"DEV": 400, "TEST": 192}
    for split, expected_count in expected.items():
        source = root / f"timit_cuts_{split}_phone.jsonl.gz"
        destination = root / f"timit_cuts_{split}_phone_nosa.jsonl.gz"
        cuts = CutSet.from_file(source)
        selected = CutSet.from_cuts(cut for cut in cuts if "-SA" not in cut.id)
        count = len(selected)
        if count != expected_count:
            raise ValueError(
                f"{split}: expected {expected_count} no-SA cuts, found {count}"
            )
        selected.to_file(destination)
        print(f"{split}: wrote {count} cuts to {destination}")


if __name__ == "__main__":
    main()
