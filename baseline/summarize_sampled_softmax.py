#%%
"""
Collect the per-run json files written by cf_ss.py / seq_rec_ss.py /
seq_rec_tisasrec_ss.py and print the numbers behind Table 1 / Table 2 / Fig. 2
for the "vanilla backbone + sampled-softmax" control (mean +/- std over seeds).

  Table 1 : overall test Recall / NDCG @ K               (test_{metric}_{K})
  Table 2 : head/tail x {overall, recent_3d, recent_7d}  (test_{split}_{metric}_{K})
  Fig. 2  : same split-level numbers as Table 2 (long-format csv has every
            dataset x backbone x split x metric x K row to re-plot from)

Run from the repo root:
    python baseline/summarize_sampled_softmax.py
    python baseline/summarize_sampled_softmax.py --result-dir ./results/sampled_softmax --metrics recall ndcg --topks 10 20
"""
import os
import re
import csv
import glob
import json
import argparse
import numpy as np
from collections import defaultdict

SPLITS = ["head_overall", "head_recent_3d", "head_recent_7d",
          "tail_overall", "tail_recent_3d", "tail_recent_7d"]
DATASET_ORDER = ["micro_video", "ml-1m", "kuairand"]
BACKBONE_ORDER = ["mf", "grurec", "sasrec", "tisasrec", "fearec", "bsarec"]


def load(result_dir):
    runs = defaultdict(list)  # (dataset, model) -> [metrics dict per seed]
    for path in sorted(glob.glob(os.path.join(result_dir, "*.json"))):
        with open(path) as f:
            r = json.load(f)
        runs[(r["dataset"], r["model"])].append(r["metrics"])
    return runs


def mean_std(runs, key):
    vals = [m[key] for m in runs if key in m]
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def fmt(ms):
    if ms is None:
        return "   -   "
    return f"{ms[0]:.4f}±{ms[1]:.4f}"


def sort_keys(keys):
    d = {x: i for i, x in enumerate(DATASET_ORDER)}
    b = {x: i for i, x in enumerate(BACKBONE_ORDER)}
    return sorted(keys, key=lambda k: (d.get(k[0], 99), b.get(k[1], 99)))


def print_table(title, runs, columns):
    """columns: list of (header, metric key)."""
    print(f"\n### {title}")
    header = ["dataset", "backbone", "#seeds"] + [h for h, _ in columns]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for (ds, model) in sort_keys(runs.keys()):
        n = len(runs[(ds, model)])
        cells = [fmt(mean_std(runs[(ds, model)], key)) for _, key in columns]
        print("| " + " | ".join([ds, model, str(n)] + cells) + " |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="./results/sampled_softmax")
    parser.add_argument("--metrics", nargs="+", default=["recall", "ndcg"])
    parser.add_argument("--topks", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--out-csv", default="./figures/sampled_softmax_summary.csv")
    args = parser.parse_args()

    runs = load(args.result_dir)
    if not runs:
        print(f"no json found in {args.result_dir}")
        return

    # ---- Table 1: overall test ------------------------------------------------
    cols = [(f"{m}@{k}", f"test_{m}_{k}") for m in args.metrics for k in args.topks]
    print_table("Table 1 (overall test)", runs, cols)

    # ---- Table 2 / Fig. 2: head/tail x time window ------------------------------
    for m in args.metrics:
        for k in args.topks:
            cols = [(s, f"test_{s}_{m}_{k}") for s in SPLITS]
            print_table(f"Table 2 / Fig. 2 ({m}@{k})", runs, cols)

    # ---- long csv ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    pat = re.compile(r"^test_(?:(?P<split>" + "|".join(SPLITS) + r")_)?(?P<metric>[a-z]+)_(?P<k>\d+)$")
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "backbone", "split", "metric", "k", "mean", "std", "n_seeds"])
        for (ds, model) in sort_keys(runs.keys()):
            keys = sorted({k for m in runs[(ds, model)] for k in m})
            for key in keys:
                mt = pat.match(key)
                if not mt:
                    continue
                mean, std, n = mean_std(runs[(ds, model)], key)
                w.writerow([ds, model, mt.group("split") or "overall_test", mt.group("metric"), mt.group("k"),
                            f"{mean:.6f}", f"{std:.6f}", n])
    print(f"\n[saved] {args.out_csv}")


if __name__ == "__main__":
    main()
