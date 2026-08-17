#!/usr/bin/env python3
"""One real LibriSpeech batch through TDNN phone CTC, including backward."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from lhotse import CutSet
from lhotse.dataset import K2SpeechRecognitionDataset, PrecomputedFeatures
from lhotse.dataset.sampling import SimpleCutSampler
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
RECIPE_DIR = SCRIPT_DIR.parent
ASR_DIR = RECIPE_DIR.parent
for path in (RECIPE_DIR, ASR_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import label_prior_ctc.train as label_prior_train  # noqa: E402
from icefall.lexicon import Lexicon  # noqa: E402
from icefall.utils import AttributeDict  # noqa: E402
from tdnn_alignment.model import TdnnLabelPriorCTC  # noqa: E402
from tdnn_alignment.train import ProtocolPhoneGraphCompiler  # noqa: E402


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lang_dir = Path("data/lang_phone_nostress")
    ProtocolPhoneGraphCompiler.lang_dir = lang_dir
    label_prior_train.CtcTrainingGraphCompiler = ProtocolPhoneGraphCompiler

    cuts = CutSet.from_file(
        "data/fbank/librispeech_cuts_train-clean-100.jsonl.gz"
    ).subset(first=2)
    dataset = K2SpeechRecognitionDataset(
        input_strategy=PrecomputedFeatures(), return_cuts=True
    )
    sampler = SimpleCutSampler(cuts, max_duration=40.0, shuffle=False)
    batch = next(
        iter(
            DataLoader(
                dataset,
                batch_size=None,
                sampler=sampler,
                num_workers=0,
            )
        )
    )

    lexicon = Lexicon(lang_dir)
    num_classes = max(lexicon.tokens) + 1
    compiler = ProtocolPhoneGraphCompiler(lexicon=lexicon, device=device)
    model = TdnnLabelPriorCTC(
        num_features=80,
        num_classes=num_classes,
        subsampling_factor=2,
        d_model=640,
        num_decoder_layers=0,
    ).to(device)
    params = AttributeDict(
        {
            "subsampling_factor": 2,
            "beam_size": 10,
            "reduction": "sum",
            "use_double_scores": True,
            "att_rate": 0.0,
            "label_prior_alpha": 0.3,
            "label_prior_floor": 6.14421235332821e-6,
            "label_prior_start_epoch": 2,
            "label_prior_update_until_epoch": 0,
            "cur_epoch": 1,
        }
    )
    loss, info = label_prior_train.compute_loss(
        params=params,
        model=model,
        batch=batch,
        graph_compiler=compiler,
        is_training=True,
        warmup=1.0,
    )
    loss.backward()
    grad_norm = torch.sqrt(
        sum(
            parameter.grad.float().square().sum()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )
    print(
        f"TDNN_SMOKE device={device} utterances={int(info['utterances'])} "
        f"frames={int(info['frames'])} loss={float(loss):.6f} "
        f"grad_norm={float(grad_norm):.6f}"
    )


if __name__ == "__main__":
    main()
