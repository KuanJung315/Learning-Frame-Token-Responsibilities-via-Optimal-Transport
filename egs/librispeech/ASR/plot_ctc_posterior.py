import os
import sys
import types
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RECIPE_DIR = SCRIPT_DIR / "conformer_ctc2"
PROJECT_ROOT = SCRIPT_DIR.parents[2]

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

if str(RECIPE_DIR) not in sys.path:
    sys.path.insert(0, str(RECIPE_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(SCRIPT_DIR)

import torch
import sentencepiece as spm
import matplotlib.pyplot as plt
from lhotse import CutSet

# ==== 你要改的設定 ====
CKPT_PATH = "/home/icefall/egs/librispeech/ASR/conformer_ctc2/exp_baseline/epoch-40.pt"   # 你的checkpoint
LANG_DIR  = SCRIPT_DIR / "data/lang_bpe_500"
CUTS_PATH = SCRIPT_DIR / "data/fbank/librispeech_cuts_test-clean.jsonl.gz"
OUT_PNG   = SCRIPT_DIR / "ctc_posterior_peaky_base40_1.png"
X_AXIS_UNIT = "frame"  # "frame", "input_frame", or "time"

# 選一個 cut（第0筆）
CUT_INDEX = 3

# 想畫「哪一串 token」：
# 1) 用 reference 文本（最方便）
USE_REF_TEXT = True

# 2) 或自己指定一段文字（例如中文也可，只要 tokenizer 支援）
CUSTOM_TEXT = "IN EVERY WAY THEY SOUGHT TO UNDERMINE THE AUTHORITY OF SAINT PAUL"

# 對 VarCTC / OTCTC checkpoint 有效:
# - "prior": 真正 inference 會用的 blank predictor
# - "posterior": 用目標文字輔助的 debug blank predictor
# - "mix": 訓練時的混合 blank，需同時給 mix_alpha
VARCTC_BLANK_SOURCE = "prior"
VARCTC_MIX_ALPHA = 0.5
# ======================


if "icefall.utils" not in sys.modules:
    icefall_stub = types.ModuleType("icefall")
    utils_stub = types.ModuleType("icefall.utils")

    def is_jit_tracing():
        return torch.jit.is_tracing()

    utils_stub.is_jit_tracing = is_jit_tracing
    icefall_stub.utils = utils_stub
    sys.modules["icefall"] = icefall_stub
    sys.modules["icefall.utils"] = utils_stub


from conformer import Conformer


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def ctc_collapse(ids, blank_id=0):
    out = []
    prev = None
    for i in ids:
        if i == blank_id:
            prev = i
            continue
        if i != prev:
            out.append(i)
        prev = i
    return out


def ctc_forced_align_viterbi(logp, targets, blank_id=0):
    """CTC forced alignment via Viterbi.

    Args:
      logp:
        Tensor of shape (T, V), log-probs for each frame.
      targets:
        List/Tensor of target token ids (non-blank), length U.
      blank_id:
        Blank token id.

    Returns:
      path_labels:
        Tensor of shape (T,), aligned labels (including blanks).
      segments:
        List of (u, start, end) for each target token index u.
        `start` is inclusive, `end` is exclusive.
    """
    if not torch.is_tensor(logp):
        logp = torch.tensor(logp)
    logp = logp.cpu()

    if not torch.is_tensor(targets):
        targets = torch.tensor(targets, dtype=torch.long)
    else:
        targets = targets.to(dtype=torch.long)

    T, V = logp.shape
    U = targets.numel()
    if T < 2 * U + 1:
        raise ValueError(f"T={T} too short for targets U={U}")

    # Build extended sequence with blanks inserted: [blank, t0, blank, t1, ... , tU-1, blank]
    ext = torch.full((2 * U + 1,), blank_id, dtype=torch.long)
    if U > 0:
        ext[1::2] = targets
    S = ext.numel()

    neg_inf = -1e9
    dp = torch.full((T, S), neg_inf)
    bp = torch.full((T, S), -1, dtype=torch.long)

    dp[0, 0] = logp[0, ext[0]]
    bp[0, 0] = 0
    if S > 1:
        dp[0, 1] = logp[0, ext[1]]
        bp[0, 1] = 1

    for t in range(1, T):
        for s in range(S):
            best_prev = s
            best_score = dp[t - 1, s]

            if s - 1 >= 0 and dp[t - 1, s - 1] > best_score:
                best_score = dp[t - 1, s - 1]
                best_prev = s - 1

            if s - 2 >= 0 and ext[s] != blank_id and ext[s] != ext[s - 2]:
                if dp[t - 1, s - 2] > best_score:
                    best_score = dp[t - 1, s - 2]
                    best_prev = s - 2

            dp[t, s] = best_score + logp[t, ext[s]]
            bp[t, s] = best_prev

    if S == 1:
        last_state = 0
    else:
        last_state = S - 1 if dp[T - 1, S - 1] > dp[T - 1, S - 2] else S - 2

    path_states = []
    s = last_state
    for t in range(T - 1, -1, -1):
        path_states.append(s)
        s = bp[t, s]
    path_states.reverse()

    path_states = torch.tensor(path_states, dtype=torch.long)
    path_labels = ext[path_states]

    segments = []
    for u in range(U):
        sidx = 2 * u + 1
        positions = (path_states == sidx).nonzero(as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        st = int(positions[0].item())
        ed = int(positions[-1].item()) + 1
        segments.append((u, st, ed))

    return path_labels, segments


def get_encoder_model(params: AttributeDict):
    return Conformer(
        num_features=getattr(params, "feature_dim", 80),
        subsampling_factor=getattr(params, "subsampling_factor", 4),
        d_model=getattr(params, "encoder_dim", 512),
        nhead=getattr(params, "nhead", 8),
        dim_feedforward=getattr(params, "dim_feedforward", 2048),
        num_encoder_layers=getattr(params, "num_encoder_layers", 12),
        num_decoder_layers=getattr(params, "num_decoder_layers", 0),
        num_classes=params.num_classes,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load spm
    sp = spm.SentencePieceProcessor()
    sp.load(str(Path(LANG_DIR) / "bpe.model"))
    blank_id = sp.piece_to_id("<blk>")

    # load cut
    cuts = CutSet.from_file(CUTS_PATH)
    cut = list(cuts)[CUT_INDEX]
    ref_text = cut.supervisions[0].text

    text = ref_text if USE_REF_TEXT else CUSTOM_TEXT

    # token ids (你想畫的 token 序列)
    tok_ids = sp.encode(text, out_type=int)
    tok_pieces = sp.encode(text, out_type=str)
    targets = torch.tensor(tok_ids, dtype=torch.long, device=device).unsqueeze(0)
    target_lengths = torch.tensor([len(tok_ids)], dtype=torch.long, device=device)

    # load ckpt and build model using ckpt params
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    exclude = {
        "model","optimizer","scheduler","grad_scaler","sampler","model_avg",
        "env_info","tensorboard","train_loss"
    }
    params = AttributeDict({k: v for k, v in ckpt.items() if k not in exclude})
    state_dict = ckpt["model"]
    params.num_classes = state_dict["encoder_output_layer.1.weight"].shape[0]

    is_varctc_ckpt = any(k.startswith("var_ctc.") for k in state_dict)
    if is_varctc_ckpt:
        raise RuntimeError(
            "This script is currently wired for conformer_ctc2 checkpoints. "
            "Your checkpoint looks like a VarCTC checkpoint."
        )

    model = get_encoder_model(params).to(device).eval()
    load_info = model.load_state_dict(state_dict, strict=False)
    if load_info.missing_keys or load_info.unexpected_keys:
        print("Checkpoint load diagnostics:")
        print("  missing_keys:", load_info.missing_keys[:10])
        print("  unexpected_keys:", load_info.unexpected_keys[:10])
        raise RuntimeError(
            "conformer_ctc2 checkpoint did not load cleanly. "
            "Please check model arguments and checkpoint compatibility."
        )

    print("Model type: conformer_ctc2")

    # load features and forward
    feats = cut.load_features()  # (T, F)
    x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logp_batched, _, _ = model(x, supervision=None, warmup=1.0)  # (1, T', V)
        Tprime = logp_batched.shape[1]
        logp = logp_batched[0, :Tprime].cpu()

        post = logp.exp()  # (T', V)

    frame_shift = 0.01  # fbank 是 10ms
    subsampling = int(getattr(params, "subsampling_factor", 4))  # ckpt 裡通常有
    t_sec = [i * frame_shift * subsampling for i in range(Tprime)]
    t_frame = list(range(Tprime))
    t_input_frame = [i * subsampling for i in range(Tprime)]
    if X_AXIS_UNIT == "frame":
        x_values = t_frame
        x_label = "Output frame index"
    elif X_AXIS_UNIT == "input_frame":
        x_values = t_input_frame
        x_label = "Input frame index (10ms)"
    elif X_AXIS_UNIT == "time":
        x_values = t_sec
        x_label = "Time (sec)"
    else:
        raise ValueError(f"Unsupported X_AXIS_UNIT: {X_AXIS_UNIT}")


    # (可選) 看看 greedy collapse 的文字，確認模型有輸出
    pred = torch.argmax(post, dim=-1).tolist()
    hyp_ids = ctc_collapse(pred, blank_id=blank_id)
    hyp_text = sp.decode(hyp_ids)
    max_prob = post.max(dim=-1).values
    entropy = -(post * logp).sum(dim=-1)
    print("REF:", ref_text)
    print("HYP:", hyp_text)
    print("Plot text:", text)
    print(
        "Posterior stats:",
        {
            "max_prob_min": round(float(max_prob.min().item()), 6),
            "max_prob_mean": round(float(max_prob.mean().item()), 6),
            "max_prob_max": round(float(max_prob.max().item()), 6),
            "num_frames_max_prob_gt_0.999": int((max_prob > 0.999).sum().item()),
            "blank_prob_mean": round(float(post[:, blank_id].mean().item()), 6),
            "entropy_mean": round(float(entropy.mean().item()), 6),
        },
    )

    # forced alignment for token positions on x-axis
    _, segments = ctc_forced_align_viterbi(logp, tok_ids, blank_id)

    # plot
    plt.figure(figsize=(12, 3))
    t = x_values

    # 先畫 blank token，看看是否長時間接近 1
    plt.plot(t, post[:, blank_id].numpy(), label="<blk>", linestyle="--")

    for i, tid in enumerate(tok_ids):
        if tid == blank_id:
            continue
        y = post[:, tid].numpy()
        plt.plot(t, y, label=tok_pieces[i])

    plt.xlabel(x_label)
    plt.ylabel("Probability")
    plt.title(f"CTC token posteriors ({X_AXIS_UNIT})")
    # token 太多 legend 會爆；你可以只畫前N個或把 legend 關掉
    if len(tok_ids) <= 20:
        plt.legend(loc="upper right", fontsize=8, ncol=2)

    # 在 x 軸底部標 token（放在每個 token 的對齊區間中點）
    y_text = -0.08  # 文字放在圖下方一點點（視你的 ylim 調整）
    for u, st, ed in segments:
        if X_AXIS_UNIT == "frame":
            mid = (st + ed) / 2
        elif X_AXIS_UNIT == "input_frame":
            mid = (st + ed) / 2 * subsampling
        else:
            mid = (st + ed) / 2 * frame_shift * subsampling
        label = tok_pieces[u].replace("▁", " ")
        plt.text(mid, y_text, label, ha="center", va="top", fontsize=10, rotation=0)

    plt.ylim(bottom=-0.18)  # 給下方文字留空間
    plt.subplots_adjust(bottom=0.28)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    print("Saved:", OUT_PNG)


if __name__ == "__main__":
    main()
