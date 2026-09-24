"""Analysis 2 (task spec Sec. 5): empirical vs. learned-Hawkes popularity concentration,
compared at the *same* snapshot timestamps and the *same* top-q percentile basis (so the
comparison is apples-to-apples -- e.g. not "top-10 items" vs "top-5% of catalog").

For each dataset (SASRec+Ours checkpoint, seed=1, no retraining), at each of the daily
snapshot times from analyze_empirical_popularity.py's grid, we compute:
  - the empirical local distribution p_emp(v|t)         (recomputed here identically)
  - the Hawkes-implied distribution pi_H(v|t) = lambda_v(t;Phi) / sum_k lambda_k(t;Phi)
and their normalized entropy / effective support / top-{1,5,20}% mass, plus the
difference Delta = Hawkes - empirical for each statistic.

IMPORTANT (per task spec Sec. 5's explicit instruction): "over-concentration" is only
used as a label when Hawkes is *substantially* more concentrated than the empirical
baseline it was fit to explain; if the two are comparable, we report that instead.

Outputs:
    results/appendix6/hawkes_concentration_metrics.csv     (per dataset x snapshot)
    results/appendix6/hawkes_concentration_summary.csv     (per dataset, mean +/- std)
    results/appendix6/hawkes_vs_empirical_curves.npz       (time-averaged cumulative curves, Figure B)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from analyze_empirical_popularity import build_item_time_dict, snapshot_grid, counts_at_snapshots

MODEL_NAME = "sasrec"  # primary illustrative architecture, as used throughout the appendix
SEED = 1


def main():
    per_snapshot_rows = []
    curves = {}

    for dataset_name in C.DATASETS:
        t0 = time.time()
        ds = C.load_dataset(dataset_name, rescale_item_time_array=True)
        item_times = build_item_time_dict(ds, dataset_name)
        grid, tmin, tmax = snapshot_grid(ds)
        counts, item_ids = counts_at_snapshots(item_times, grid)
        m_item = ds.m_item
        full_counts = np.zeros((len(grid), m_item), dtype=np.int32)
        item_ids_arr = np.array(item_ids, dtype=np.int64)
        valid = item_ids_arr < m_item
        full_counts[:, item_ids_arr[valid]] = counts[:, valid]

        model, eta_default = C.load_ours_ckpt(dataset_name, MODEL_NAME, SEED, ds)
        lam_all, mu, alpha, beta = C.hawkes_lambda_at_times(model, ds, grid)
        print(f"[{dataset_name}] {len(grid)} snapshots computed ({time.time()-t0:.1f}s), "
              f"beta={beta:.4f}")

        emp_curves, hawkes_curves = [], []
        for si, t in enumerate(grid):
            c_emp = full_counts[si].astype(np.float64)
            lam = lam_all[si].astype(np.float64)

            row = dict(dataset=dataset_name, snapshot_idx=si, t_days=float(t))
            if c_emp.sum() > 0:
                row["H_norm_emp"] = C.normalized_entropy(c_emp)
                row["N_eff_emp"] = C.effective_support(c_emp)
                tq = C.top_q_mass(c_emp, (0.01, 0.05, 0.20))
                row["top1pct_emp"], row["top5pct_emp"], row["top20pct_emp"] = tq[0.01], tq[0.05], tq[0.20]
                g_emp, curve_emp = C.cumulative_popularity_curve(c_emp, 200)
                emp_curves.append(curve_emp)
            row["H_norm_hawkes"] = C.normalized_entropy(lam)
            row["N_eff_hawkes"] = C.effective_support(lam)
            tqh = C.top_q_mass(lam, (0.01, 0.05, 0.20))
            row["top1pct_hawkes"], row["top5pct_hawkes"], row["top20pct_hawkes"] = tqh[0.01], tqh[0.05], tqh[0.20]
            g_h, curve_h = C.cumulative_popularity_curve(lam, 200)
            hawkes_curves.append(curve_h)

            if c_emp.sum() > 0:
                row["delta_H_norm"] = row["H_norm_hawkes"] - row["H_norm_emp"]
                row["delta_top5pct"] = row["top5pct_hawkes"] - row["top5pct_emp"]
                row["delta_top1pct"] = row["top1pct_hawkes"] - row["top1pct_emp"]

            per_snapshot_rows.append(row)

        curves[dataset_name] = dict(
            grid_pct=g_h,
            emp_mean=np.array(emp_curves).mean(0) if emp_curves else None,
            hawkes_mean=np.array(hawkes_curves).mean(0),
            hawkes_std=np.array(hawkes_curves).std(0),
        )

    df = pd.DataFrame(per_snapshot_rows)
    df.to_csv(f"{C.RESULTS_DIR}/hawkes_concentration_metrics.csv", index=False)

    summary_rows = []
    for dataset_name in C.DATASETS:
        sub = df[df.dataset == dataset_name]
        row = dict(dataset=dataset_name, n_snapshots=len(sub))
        for col in ["H_norm_emp", "H_norm_hawkes", "top5pct_emp", "top5pct_hawkes",
                    "N_eff_emp", "N_eff_hawkes", "delta_H_norm", "delta_top5pct"]:
            if col in sub:
                row[f"{col}_mean"] = sub[col].mean()
                row[f"{col}_std"] = sub[col].std()
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{C.RESULTS_DIR}/hawkes_concentration_summary.csv", index=False)

    save_kwargs = {}
    for k, v in curves.items():
        save_kwargs[f"{k}_grid"] = v["grid_pct"]
        save_kwargs[f"{k}_hawkes_mean"] = v["hawkes_mean"]
        save_kwargs[f"{k}_hawkes_std"] = v["hawkes_std"]
        if v["emp_mean"] is not None:
            save_kwargs[f"{k}_emp_mean"] = v["emp_mean"]
    np.savez(f"{C.RESULTS_DIR}/hawkes_vs_empirical_curves.npz", **save_kwargs)

    print("\n=== Empirical vs. Hawkes concentration (mean over snapshots) ===")
    print(summary_df[["dataset", "H_norm_emp_mean", "H_norm_hawkes_mean", "delta_H_norm_mean",
                       "top5pct_emp_mean", "top5pct_hawkes_mean", "delta_top5pct_mean"]]
          .to_string(index=False))
    print(f"\nsaved to {C.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
