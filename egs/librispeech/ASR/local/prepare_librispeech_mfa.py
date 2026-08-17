#!/usr/bin/env python3
"""Convert LibriSpeech MFA phone alignments into compact JSONL manifests.

The source parquet files contain phone intervals but no word labels.  This
script reconstructs word intervals only when the stress-stripped LibriSpeech
lexicon exactly explains the non-silence phone sequence.  Utterances that do
not have an exact pronunciation path are retained with phone intervals and an
empty ``words`` list so that they cannot silently contaminate word-boundary
evaluation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shlex
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import pyarrow.parquet as pq


SOURCE_DATASET = "anyspeech/librispeech_MFA_alignments"
SOURCE_URL = "https://huggingface.co/datasets/anyspeech/librispeech_MFA_alignments"
DEFAULT_SPLITS = (
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
)
SILENCE_PHONES = frozenset({"[SIL]", "SIL", "<SIL>", "SP", "SPN"})


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=Path("download/librispeech_MFA_alignments/parquet"),
    )
    parser.add_argument(
        "--librispeech-dir",
        type=Path,
        default=Path("download/LibriSpeech"),
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=Path("data/lang_phone/lexicon.txt"),
    )
    parser.add_argument(
        "--word-alignment-dir",
        type=Path,
        default=Path(
            "download/librispeech_MFA_alignments/word_alignments_raw/LibriSpeech"
        ),
        help=(
            "Root of the condensed CorentinJ/Loren Lugosch word alignments. "
            "When present, these are preferred over lexicon reconstruction."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/librispeech_mfa"),
    )
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Rebuild metadata.json from existing converted manifests.",
    )
    return parser


def strip_stress(phone: str) -> str:
    return re.sub(r"\d+$", "", phone.upper())


def read_lexicon(path: Path) -> Dict[str, Tuple[Tuple[str, ...], ...]]:
    pronunciations: Dict[str, List[Tuple[str, ...]]] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 2:
                raise ValueError(f"Malformed lexicon line {line_number}: {line!r}")
            word = fields[0].upper()
            pronunciation = tuple(strip_stress(phone) for phone in fields[1:])
            entries = pronunciations.setdefault(word, [])
            if pronunciation not in entries:
                entries.append(pronunciation)
    return {word: tuple(entries) for word, entries in pronunciations.items()}


def read_transcripts(split_dir: Path) -> Dict[str, str]:
    transcripts: Dict[str, str] = {}
    for path in sorted(split_dir.rglob("*.trans.txt")):
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    utterance_id, text = line.split(maxsplit=1)
                except ValueError as error:
                    raise ValueError(
                        f"Malformed transcript {path}:{line_number}: {line!r}"
                    ) from error
                transcripts[utterance_id] = text
    return transcripts


def read_condensed_word_alignments(
    split_dir: Path,
) -> Dict[str, List[Dict[str, object]]]:
    """Read the original condensed MFA word alignment format.

    Each line contains an utterance ID, a comma-separated interval label list,
    and the corresponding interval end times.  Empty labels are silences.
    """

    alignments: Dict[str, List[Dict[str, object]]] = {}
    if not split_dir.is_dir():
        return alignments
    for path in sorted(split_dir.rglob("*.alignment.txt")):
        # The public archive contains duplicated train-other files under one
        # incorrect speaker/book directory. Accept a file only when its name
        # agrees with its directory; rejected utterances can still use the
        # conservative exact-lexicon fallback below.
        expected_key = f"{path.parent.parent.name}-{path.parent.name}"
        file_key = path.name.removesuffix(".alignment.txt")
        if file_key != expected_key:
            continue
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    fields = shlex.split(line)
                except ValueError as error:
                    raise ValueError(
                        f"Cannot parse alignment {path}:{line_number}: {line!r}"
                    ) from error
                if len(fields) != 3:
                    raise ValueError(
                        f"Malformed alignment {path}:{line_number}: {line!r}"
                    )
                utterance_id, label_field, end_field = fields
                labels = label_field.split(",")
                interval_ends = [float(value) for value in end_field.split(",")]
                if len(labels) != len(interval_ends):
                    raise ValueError(
                        f"Mismatched alignment arrays {path}:{line_number}"
                    )
                intervals: List[Dict[str, object]] = []
                interval_start = 0.0
                for label, interval_end in zip(labels, interval_ends):
                    if interval_end < interval_start:
                        raise ValueError(
                            f"Non-monotonic alignment {path}:{line_number}"
                        )
                    if label:
                        intervals.append(_interval(label, interval_start, interval_end))
                    interval_start = interval_end
                alignments[utterance_id] = intervals
    return alignments


def find_parquet(parquet_dir: Path, split: str) -> Path:
    parquet_prefix = split.replace("-", ".")
    matches = sorted(parquet_dir.glob(f"{parquet_prefix}-*.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one parquet for {split} in {parquet_dir}; "
            f"found {matches}"
        )
    return matches[0]


def exact_word_phone_ends(
    words: Sequence[str],
    phones: Sequence[str],
    lexicon: Mapping[str, Sequence[Tuple[str, ...]]],
) -> Tuple[int, ...] | None:
    """Return the exclusive phone end for every word, or ``None``.

    Backtracking is needed for words with multiple pronunciations.  The cache
    keeps the cost linear for the overwhelmingly common deterministic case.
    """

    normalized_words = tuple(word.upper() for word in words)
    normalized_phones = tuple(strip_stress(phone) for phone in phones)

    @lru_cache(maxsize=None)
    def visit(word_index: int, phone_index: int) -> Tuple[int, ...] | None:
        if word_index == len(normalized_words):
            return () if phone_index == len(normalized_phones) else None
        word = normalized_words[word_index]
        for pronunciation in lexicon.get(word, ()):
            next_phone = phone_index + len(pronunciation)
            if normalized_phones[phone_index:next_phone] != tuple(pronunciation):
                continue
            suffix = visit(word_index + 1, next_phone)
            if suffix is not None:
                return (next_phone,) + suffix
        return None

    return visit(0, 0)


def _interval(symbol: str, start: float, end: float) -> Dict[str, object]:
    return {
        "symbol": symbol,
        "start": round(float(start), 6),
        "duration": round(float(end) - float(start), 6),
    }


def convert_row(
    row: Mapping[str, object],
    split: str,
    transcript: str | None,
    lexicon: Mapping[str, Sequence[Tuple[str, ...]]],
    condensed_words: Sequence[Mapping[str, object]] | None = None,
) -> Tuple[Dict[str, object], str]:
    utterance_id = str(row["identifier"])
    duration = float(row["duration"])
    phones = [str(phone) for phone in row["phones"]]  # type: ignore[index]
    starts = [float(value) for value in row["start"]]  # type: ignore[index]
    ends = [float(value) for value in row["end"]]  # type: ignore[index]
    if not (len(phones) == len(starts) == len(ends)):
        raise ValueError(f"Mismatched phone arrays for {utterance_id}")
    if any(end < start for start, end in zip(starts, ends)):
        raise ValueError(f"Negative phone duration for {utterance_id}")
    if any(starts[i] < starts[i - 1] for i in range(1, len(starts))):
        raise ValueError(f"Non-monotonic phone alignment for {utterance_id}")

    phone_intervals = [
        _interval(phone, start, end)
        for phone, start, end in zip(phones, starts, ends)
    ]
    result: Dict[str, object] = {
        "id": utterance_id,
        "split": split,
        "duration": duration,
        "text": transcript,
        "phones": phone_intervals,
        "words": [],
    }
    if transcript is None:
        result["word_alignment_status"] = "missing_transcript"
        return result, "missing_transcript"

    words = transcript.split()
    if condensed_words is not None:
        condensed_symbols = [str(item["symbol"]) for item in condensed_words]
        if condensed_symbols == words:
            result["words"] = list(condensed_words)
            result["word_alignment_status"] = "condensed_exact"
            return result, "condensed_exact"
        result["word_alignment_status"] = "condensed_text_mismatch"
        return result, "condensed_text_mismatch"

    if any(word.upper() not in lexicon for word in words):
        result["word_alignment_status"] = "lexicon_oov"
        return result, "lexicon_oov"

    non_silence_indices = [
        index
        for index, phone in enumerate(phones)
        if phone.upper() not in SILENCE_PHONES
    ]
    non_silence_phones = [phones[index] for index in non_silence_indices]
    word_ends = exact_word_phone_ends(words, non_silence_phones, lexicon)
    if word_ends is None:
        result["word_alignment_status"] = "pronunciation_mismatch"
        return result, "pronunciation_mismatch"

    word_intervals: List[Dict[str, object]] = []
    phone_begin = 0
    for word, phone_end in zip(words, word_ends):
        if phone_end <= phone_begin:
            raise ValueError(f"Empty pronunciation for {utterance_id}: {word}")
        first_original = non_silence_indices[phone_begin]
        last_original = non_silence_indices[phone_end - 1]
        interval = _interval(word, starts[first_original], ends[last_original])
        interval["phone_begin"] = phone_begin
        interval["phone_end"] = phone_end
        word_intervals.append(interval)
        phone_begin = phone_end

    result["words"] = word_intervals
    result["word_alignment_status"] = "lexicon_exact_fallback"
    return result, "lexicon_exact_fallback"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_parquet_rows(path: Path, batch_size: int) -> Iterable[Mapping[str, object]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def convert_split(
    split: str,
    parquet_path: Path,
    transcript_dir: Path,
    word_alignment_dir: Path,
    output_path: Path,
    lexicon: Mapping[str, Sequence[Tuple[str, ...]]],
    batch_size: int,
) -> Dict[str, object]:
    transcripts = read_transcripts(transcript_dir)
    condensed = read_condensed_word_alignments(word_alignment_dir)
    counts = {
        "total": 0,
        "condensed_exact": 0,
        "lexicon_exact_fallback": 0,
        "missing_transcript": 0,
        "condensed_text_mismatch": 0,
        "lexicon_oov": 0,
        "pronunciation_mismatch": 0,
        "phones": 0,
        "words": 0,
    }
    with gzip.open(output_path, "wt", encoding="utf-8") as sink:
        for row in iter_parquet_rows(parquet_path, batch_size):
            utterance_id = str(row["identifier"])
            converted, status = convert_row(
                row=row,
                split=split,
                transcript=transcripts.get(utterance_id),
                lexicon=lexicon,
                condensed_words=condensed.get(utterance_id),
            )
            sink.write(json.dumps(converted, ensure_ascii=False) + "\n")
            counts["total"] += 1
            counts[status] += 1
            counts["phones"] += len(converted["phones"])  # type: ignore[arg-type]
            counts["words"] += len(converted["words"])  # type: ignore[arg-type]

    counts["word_aligned"] = (
        counts["condensed_exact"] + counts["lexicon_exact_fallback"]
    )
    counts["word_coverage"] = (
        counts["word_aligned"] / counts["total"] if counts["total"] else 0.0
    )
    return {
        "split": split,
        "source_parquet": str(parquet_path),
        "source_sha256": sha256(parquet_path),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        **counts,
    }


def summarize_existing_split(
    split: str,
    parquet_path: Path,
    output_path: Path,
) -> Dict[str, object]:
    if not output_path.is_file():
        raise FileNotFoundError(output_path)
    counts: Dict[str, int | float] = {
        "total": 0,
        "condensed_exact": 0,
        "lexicon_exact_fallback": 0,
        "missing_transcript": 0,
        "condensed_text_mismatch": 0,
        "lexicon_oov": 0,
        "pronunciation_mismatch": 0,
        "phones": 0,
        "words": 0,
    }
    with gzip.open(output_path, "rt", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            status = str(record["word_alignment_status"])
            counts["total"] += 1
            counts[status] = int(counts.get(status, 0)) + 1
            counts["phones"] += len(record["phones"])
            counts["words"] += len(record["words"])
    counts["word_aligned"] = int(counts["condensed_exact"]) + int(
        counts["lexicon_exact_fallback"]
    )
    counts["word_coverage"] = (
        float(counts["word_aligned"]) / float(counts["total"])
        if counts["total"]
        else 0.0
    )
    return {
        "split": split,
        "source_parquet": str(parquet_path),
        "source_sha256": sha256(parquet_path),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        **counts,
    }


def main() -> None:
    args = get_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lexicon = read_lexicon(args.lexicon)

    split_stats = []
    for split in args.splits:
        if split not in DEFAULT_SPLITS:
            raise ValueError(f"Unsupported split: {split}")
        parquet_path = find_parquet(args.parquet_dir, split)
        transcript_dir = args.librispeech_dir / split
        output_path = args.output_dir / f"librispeech_mfa_{split}.jsonl.gz"
        if args.summarize_existing:
            print(f"Summarizing {split}: {output_path}", flush=True)
            stats = summarize_existing_split(split, parquet_path, output_path)
        else:
            print(f"Converting {split}: {parquet_path} -> {output_path}", flush=True)
            stats = convert_split(
                split=split,
                parquet_path=parquet_path,
                transcript_dir=transcript_dir,
                word_alignment_dir=args.word_alignment_dir / split,
                output_path=output_path,
                lexicon=lexicon,
                batch_size=args.batch_size,
            )
        split_stats.append(stats)
        print(
            f"  word coverage: {stats['word_aligned']}/{stats['total']} "
            f"({100.0 * float(stats['word_coverage']):.3f}%)",
            flush=True,
        )

    metadata = {
        "format_version": 1,
        "source_dataset": SOURCE_DATASET,
        "source_url": SOURCE_URL,
        "source_annotation": "Montreal Forced Aligner phone timestamps",
        "condensed_word_source": (
            "CorentinJ/librispeech-alignments (Loren Lugosch MFA alignments)"
        ),
        "condensed_word_archive_sha256": (
            "80c0b0bc8190ef3fd565e2bf490f9a1c088656ba025a717880a101e71055d921"
        ),
        "reference_warning": (
            "MFA timestamps are automatic pseudo-references, not manual ground truth."
        ),
        "word_reconstruction": (
            "Prefer original condensed MFA word intervals; when a condensed entry "
            "is missing, use exact stress-stripped pronunciation matching against "
            "the supplied lexicon. No approximate matches are admitted."
        ),
        "lexicon": str(args.lexicon),
        "lexicon_sha256": sha256(args.lexicon),
        "splits": split_stats,
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
