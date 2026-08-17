# FTA

This repository contains the implementation and experiments for FTA,
built on top of the [icefall](https://github.com/k2-fsa/icefall) speech
recognition toolkit.

> This project is derived from icefall and includes modifications for our
> FTA training and alignment experiments. The original icefall project is
> licensed under the Apache License 2.0.

## Overview

FTA is a method for speech recognition and alignment based on
[請在這裡簡短描述你的方法].

The main experiments include:

- LibriSpeech
- AISHELL-1
- TIMIT
- CTC and alignment evaluation
- [其他主要實驗]

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

Training example:

```bash
python3 egs/librispeech/ASR/conformer_ctc2/train_vi_ot_v2.py \
  [請填入實際參數]
