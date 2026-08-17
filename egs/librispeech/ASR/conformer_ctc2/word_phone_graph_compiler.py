"""Deterministic word-to-phone targets and linear CTC graphs.

The same compiler is intended to be used for LibriSpeech phone training and
for transcript-conditioned zero-shot alignment on TIMIT.  In particular, the
OT/FGW columns and the CTC graph are built from the exact same phone sequence;
we do not compose a pronunciation lattice for one loss while giving a single
pronunciation to the other.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import k2
import torch

from icefall.lexicon import Lexicon, read_lexicon


_WORD_RE = re.compile(r"[A-Z0-9]+(?:'[A-Z0-9]+)*")


@dataclass(frozen=True)
class PhoneTranscript:
    """A normalized word transcript and its deterministic phone expansion."""

    words: List[str]
    phones: List[str]
    phone_ids: List[int]
    # Half-open phone index span [begin, end) for each word.
    word_phone_spans: List[Tuple[int, int]]
    oov_words: List[str]


class WordPhoneCtcTrainingGraphCompiler:
    """Expand word transcripts with ``lexicon.txt`` and build linear CTC FSAs.

    Multiple pronunciations are resolved by taking the first pronunciation in
    the lexicon file.  This rule is deliberately simple and stable so training
    and cross-corpus evaluation cannot silently choose different phone targets.
    Unknown words map to the phone pronunciation of ``<UNK>`` (normally SPN).
    """

    def __init__(
        self,
        lang_dir: Path,
        lexicon: Lexicon,
        device: torch.device,
        oov_word: str = "<UNK>",
    ) -> None:
        self.lang_dir = Path(lang_dir)
        self.token_table = lexicon.token_table
        self.device = device
        self.sos_id = None
        self.eos_id = None

        pronunciations: Dict[str, Tuple[str, ...]] = {}
        for word, phones in read_lexicon(self.lang_dir / "lexicon.txt"):
            # setdefault implements the documented first-pronunciation rule.
            pronunciations.setdefault(word.upper(), tuple(phones))

        oov_key = oov_word.upper()
        if oov_key not in pronunciations:
            if "SPN" not in self.token_table.symbols:
                raise ValueError(
                    f"{oov_word} is absent from the lexicon and SPN is not a token"
                )
            pronunciations[oov_key] = ("SPN",)

        for word, phones in pronunciations.items():
            missing = [p for p in phones if p not in self.token_table.symbols]
            if missing:
                raise ValueError(
                    f"Pronunciation for {word} contains unknown phones: {missing}"
                )

        self.pronunciations = pronunciations
        self.oov_word = oov_key

    @staticmethod
    def normalize_words(text: str) -> List[str]:
        """Normalize Libri/TIMIT sentence text without consulting gold phones."""
        normalized = (
            text.replace("\u2018", "'")
            .replace("\u2019", "'")
            .replace("\u02bc", "'")
            .upper()
        )
        return _WORD_RE.findall(normalized)

    def expand_text(self, text: str) -> PhoneTranscript:
        words = self.normalize_words(text)
        phones: List[str] = []
        spans: List[Tuple[int, int]] = []
        oov_words: List[str] = []

        for word in words:
            begin = len(phones)
            pronunciation = self.pronunciations.get(word)
            if pronunciation is None:
                pronunciation = self.pronunciations[self.oov_word]
                oov_words.append(word)
            phones.extend(pronunciation)
            spans.append((begin, len(phones)))

        phone_ids = [self.token_table[p] for p in phones]
        return PhoneTranscript(
            words=words,
            phones=phones,
            phone_ids=phone_ids,
            word_phone_spans=spans,
            oov_words=oov_words,
        )

    def texts_to_ids(self, texts: Sequence[str], sep: str = " ") -> List[List[int]]:
        # ``sep`` is retained for API compatibility with icefall graph compilers.
        del sep
        return [self.expand_text(text).phone_ids for text in texts]

    def compile(self, token_ids: List[List[int]], modified: bool = False) -> k2.Fsa:
        return k2.ctc_graph(token_ids, modified=modified, device=self.device)

