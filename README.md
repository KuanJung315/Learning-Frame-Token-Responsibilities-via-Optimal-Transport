# FTA

This repository contains the implementation and experiments for FTA,
built on top of the [icefall](https://github.com/k2-fsa/icefall) speech
recognition toolkit.

> This project is derived from icefall and includes modifications for our
> FTA training and alignment experiments. The original icefall project is
> licensed under the Apache License 2.0.

## Overview

FTA is a frame-token assignment method for CTC-based automatic
speech recognition. It introduces a learned blank gate and an optimal
transport alignment between acoustic frames and transcript tokens. The OT
coupling provides global mass conservation, positional preference, and
optional Fused Gromov-Wasserstein structural matching, while CTC preserves
valid monotonic alignments.

The main experiments include:

- LibriSpeech 100-hour and 960-hour ASR
- AISHELL-1 Mandarin ASR
- TIMIT phone recognition and alignment
- CTC forced-alignment evaluation
- Optimal Transport and Fused Gromov-Wasserstein alignment
- Blank-prior and alignment-loss ablation studies

## Main Code

The primary implementation is located in:

- `egs/librispeech/ASR/conformer_ctc2/`
- `egs/aishell/ASR/conformer_ctc2/`
- `egs/timit/ASR/conformer_ctc2/`

Important files:

- `train_vi_ot_v2.py`: FTA training
- `varctc_v2_utils.py`: FTA utilities
- `evaluate_mfa_alignment.py`: alignment evaluation
- `train.py`: training entry point

## Installation

This project follows the original icefall installation procedure:

https://k2-fsa.github.io/icefall/installation/index.html

## Usage

Prepare the LibriSpeech data and language resources following the original
icefall recipe, then run training from the LibriSpeech ASR directory:

```bash
cd egs/librispeech/ASR

python3 conformer_ctc2/train_vi_ot_v2.py \
  --world-size 2 \
  --num-epochs 40 \
  --start-epoch 1 \
  --exp-dir conformer_ctc2/exp/vfta \
  --full-libri 0 \
  --att-rate 0.7 \
  --num-decoder-layers 6 \
  --max-duration 500 \
  --use-fp16 1 \
  --lambda-ot 0.1 \
  --ot-eps 0.3 \
  --ot-iters 30 \
  --ot-beta-pos 1.0 \
  --col-marginal-type acoustic \
  --ot-token-prior-sigma 0.15 \
  --ot-token-prior-score-temp 1.0 \
  --ot-token-prior-floor 0.05 \
  --lambda-kl-blank 0.03 \
  --alpha-smooth-mix 0.1 \
  --label-embed-dim 256 \
  --init-blank-prob 0.35 \
  --gate-warmup-start 1000 \
  --gate-warmup-steps 3000 \
  --train-prior-mix 1.0 \
  --lambda-alpha-mean 0.005 \
  --alpha-mean-source prior \
  --alpha-mean-mode floor \
  --alpha-mean-target 0.35
