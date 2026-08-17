# Phone CTC training graph compiler for TIMIT.
#
# Mirrors icefall's CharCtcTrainingGraphCompiler API (texts_to_ids -> token ids,
# compile(token_ids) -> k2 ctc graph) which the VarCTC-v2 / VI-OT training code
# is written against, but splits the transcript on whitespace so each phone is
# one token (TIMIT supervision.text is a space-separated phone sequence).

from typing import List

import k2
import torch

from icefall.lexicon import Lexicon


class PhoneCtcTrainingGraphCompiler(object):
    def __init__(
        self,
        lexicon: Lexicon,
        device: torch.device,
        sos_token: str = "<sos/eos>",
        eos_token: str = "<sos/eos>",
        oov: str = "<UNK>",
    ):
        """
        Args:
          lexicon: built from data/lang_phone (tokens.txt holds <eps>=0=blank
            and the phone inventory).
          device: device for compiling transcripts to FSAs.
          oov: out-of-vocabulary token; phones not in the table map to it. The
            TIMIT phone set is closed, so this is rarely used.
        """
        self.token_table = lexicon.token_table
        self.device = device

        self.oov_id = (
            self.token_table[oov] if oov in self.token_table.symbols else None
        )
        # SOS/EOS are only needed for the attention decoder (att_rate > 0).
        self.sos_id = (
            self.token_table[sos_token]
            if sos_token in self.token_table.symbols
            else None
        )
        self.eos_id = (
            self.token_table[eos_token]
            if eos_token in self.token_table.symbols
            else None
        )

    def texts_to_ids(self, texts: List[str], sep: str = " ") -> List[List[int]]:
        """Convert space-separated phone transcripts to lists of token IDs."""
        ids: List[List[int]] = []
        for text in texts:
            sub_ids: List[int] = []
            for tok in text.split():
                if tok in self.token_table.symbols:
                    sub_ids.append(self.token_table[tok])
                elif self.oov_id is not None:
                    sub_ids.append(self.oov_id)
            ids.append(sub_ids)
        return ids

    def compile(self, token_ids: List[List[int]], modified: bool = False) -> k2.Fsa:
        """Build a CTC graph (FsaVec) from a list-of-list of token IDs."""
        return k2.ctc_graph(token_ids, modified=modified, device=self.device)
