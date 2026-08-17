#!/usr/bin/env python3
"""Aggregate the attention-decoder sweep TSVs into a dev-selected 3-seed table.

Reads the per-seed TSVs written by the run_li100h_*_attention_decode.sbatch
launchers (columns: model seed bias set best_key wer) plus the 960h
attention-rescore TSVs, then for each model picks the single decode bias that
minimises mean dev WER (dev-clean+dev-other averaged over seeds) and reports
the matched test-clean/test-other mean +/- std across seeds.

Usage:
    python aggregate_attention_decode.py logs/li100_*_att_*_seed*.tsv \
        logs/cf_vi960_floor030_attention_rescore_*.tsv \
        logs/cf_baseline960_ctc_bias_attention_*.tsv
"""
import sys
import glob
from collections import defaultdict
from statistics import mean, pstdev


def load(paths):
    # rows[(model, bias)][set] -> {seed: wer}
    rows = defaultdict(lambda: defaultdict(dict))
    for p in paths:
        with open(p) as fh:
            header = fh.readline().split("\t")
            idx = {h.strip(): i for i, h in enumerate(header)}
            for line in fh:
                c = line.rstrip("\n").split("\t")
                if len(c) < len(header):
                    continue
                try:
                    model = c[idx["model"]]
                    seed = c[idx["seed"]]
                    bias = c[idx["bias"]]
                    s = c[idx["set"]]
                    wer = float(c[idx["wer"]])
                except (KeyError, ValueError):
                    continue
                rows[(model, bias)][s][seed] = wer
    return rows


def avg_over_seeds(d):
    return mean(d.values()) if d else None


def main():
    paths = []
    for a in sys.argv[1:]:
        paths.extend(glob.glob(a))
    if not paths:
        print("no TSVs matched", file=sys.stderr)
        sys.exit(1)
    rows = load(paths)

    models = sorted({m for (m, _b) in rows})
    print(f"{'model':10} {'bias':>5} {'tc(mean±std)':>16} {'to(mean±std)':>16} "
          f"{'dev sel':>8} seeds")
    for model in models:
        biases = [b for (m, b) in rows if m == model]
        # dev-select: minimise mean(dev-clean, dev-other) averaged over seeds.
        best = None
        for b in biases:
            sets = rows[(model, b)]
            dc = avg_over_seeds(sets.get("dev-clean", {}))
            do = avg_over_seeds(sets.get("dev-other", {}))
            if dc is None or do is None:
                continue
            devscore = (dc + do) / 2
            if best is None or devscore < best[0]:
                best = (devscore, b)
        if best is None:
            # no dev data (e.g. ran test-only); fall back to listing each bias
            for b in sorted(biases):
                sets = rows[(model, b)]
                tc = sets.get("test-clean", {})
                to = sets.get("test-other", {})
                tcv = list(tc.values()); tov = list(to.values())
                print(f"{model:10} {b:>5} "
                      f"{(mean(tcv) if tcv else float('nan')):>8.3f}±"
                      f"{(pstdev(tcv) if len(tcv)>1 else 0):.3f}  "
                      f"{(mean(tov) if tov else float('nan')):>8.3f}±"
                      f"{(pstdev(tov) if len(tov)>1 else 0):.3f}  {'(test)':>8} "
                      f"{sorted(set(list(tc)+list(to)))}")
            continue
        b = best[1]
        sets = rows[(model, b)]
        tc = sets.get("test-clean", {}); to = sets.get("test-other", {})
        tcv = [tc[s] for s in sorted(tc)]; tov = [to[s] for s in sorted(to)]
        seeds = sorted(set(list(tc) + list(to)))
        tcm = mean(tcv) if tcv else float("nan")
        tom = mean(tov) if tov else float("nan")
        tcs = pstdev(tcv) if len(tcv) > 1 else 0.0
        tos = pstdev(tov) if len(tov) > 1 else 0.0
        print(f"{model:10} {b:>5} {tcm:>8.3f}±{tcs:.3f}  {tom:>8.3f}±{tos:.3f}  "
              f"{best[0]:>8.3f} {seeds}")


if __name__ == "__main__":
    main()
