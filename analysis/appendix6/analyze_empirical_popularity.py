"""Analysis 1 (task spec Sec. 3): empirical temporal popularity regime of each dataset,
computed from raw interaction data only -- no trained model is used here.

For each dataset, at a grid of daily snapshot times t (starting once a full 3-day window
is available, ending at the dataset's last timestamp), we compute the local empirical
item distribution

    c_t(v) = # interactions with item v during [t - 3 days, t)
    p_emp(v|t) = c_t(v) / sum_k c_t(k)

and report, per snapshot: normalized entropy, effective support (raw and /|V|), top-{1,5,20}%
cumulative mass, and Gini. Temporal instability is captured by the Jensen-Shannon divergence
and Top-20% Head-set Jaccard turnover between consecutive snapshots.

This is a *different*, complementary quantity to the repository's own (static) Head/Tail
split used in Table 1/4 -- see common.py's module docstring.

Outputs:
    results/appendix6/empirical_popularity_snapshots.csv   (per dataset x snapshot)
    results/appendix6/empirical_popularity_summary.csv      (per dataset, mean +/- std over snapshots)
    results/appendix6/empirical_popularity_curves.npz       (time-averaged cumulative curves, for Figure A)
    results/appendix6/temporal_shift_metrics.csv            (per dataset x consecutive-snapshot-pair)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

WINDOW_DAYS = 3.0
SNAPSHOT_STEP_DAYS = 1.0


def build_item_time_dict(ds, dataset_name):
    """item -> sorted np.array of ALL raw interaction times (days since epoch), from
    ds.time_dict (train+valid+test), matching how module/dataset.py already builds
    item_time_array internally (same source, just kept per-item and un-truncated here so
    the empirical count is exact, not capped at the model's 50-event window)."""
    item_times = {}
    for u, items in ds.time_dict.items():
        for it, t in items.items():
            item_times.setdefault(it, []).append(t)
    for it in item_times:
        item_times[it] = np.sort(np.array(item_times[it], dtype=np.float64)) / 86400.0
    return item_times


def snapshot_grid(ds):
    all_times = []
    for u, items in ds.time_dict.items():
        all_times.extend(items.values())
    all_times = np.array(all_times, dtype=np.float64) / 86400.0
    tmin, tmax = all_times.min(), all_times.max()
    start = tmin + WINDOW_DAYS
    grid = np.arange(start, tmax, SNAPSHOT_STEP_DAYS)
    return grid, tmin, tmax


def counts_at_snapshots(item_times, grid):
    """Returns (n_snapshots, n_items_with_any_history) count matrix via one vectorized
    searchsorted call per item (not per (item, snapshot) pair)."""
    items = sorted(item_times.keys())
    n_items, n_snap = len(items), len(grid)
    counts = np.zeros((n_snap, n_items), dtype=np.int32)
    lo = grid - WINDOW_DAYS
    hi = grid
    query = np.concatenate([lo, hi])
    for j, it in enumerate(items):
        idx = np.searchsorted(item_times[it], query)
        lo_idx, hi_idx = idx[:n_snap], idx[n_snap:]
        counts[:, j] = hi_idx - lo_idx
    return counts, items


def main():
    per_snapshot_rows = []
    temporal_rows = []
    curves = {}

    for dataset_name in C.DATASETS:
        t0 = time.time()
        ds = C.load_dataset(dataset_name, rescale_item_time_array=False)
        item_times = build_item_time_dict(ds, dataset_name)
        grid, tmin, tmax = snapshot_grid(ds)
        counts, item_ids = counts_at_snapshots(item_times, grid)
        # pad to full catalog width (items with zero interactions anywhere still count
        # toward |V| for normalized-entropy's log-support denominator per Sec. 3.1/3.2)
        m_item = ds.m_item
        full_counts = np.zeros((len(grid), m_item), dtype=np.int32)
        item_ids_arr = np.array(item_ids, dtype=np.int64)
        valid = item_ids_arr < m_item
        full_counts[:, item_ids_arr[valid]] = counts[:, valid]

        print(f"[{dataset_name}] {len(grid)} snapshots, m_item={m_item}, span={tmax-tmin:.1f}d "
              f"({time.time()-t0:.1f}s to build)")

        curve_list = []
        prev_p = None
        head20_prev = None
        for si, t in enumerate(grid):
            c = full_counts[si].astype(np.float64)
            total = c.sum()
            row = dict(dataset=dataset_name, snapshot_idx=si, t_days=float(t), total_interactions=int(total))
            if total > 0:
                row["H_norm"] = C.normalized_entropy(c)
                neff = C.effective_support(c)
                row["N_eff"] = neff
                row["N_eff_over_V"] = neff / m_item
                tq = C.top_q_mass(c, (0.01, 0.05, 0.20))
                row["top1pct_mass"] = tq[0.01]
                row["top5pct_mass"] = tq[0.05]
                row["top20pct_mass"] = tq[0.20]
                row["gini"] = C.gini(c)
                grid_pct, curve = C.cumulative_popularity_curve(c, n_points=200)
                curve_list.append(curve)
                p = c / total
            else:
                row.update(H_norm=np.nan, N_eff=np.nan, N_eff_over_V=np.nan,
                           top1pct_mass=np.nan, top5pct_mass=np.nan, top20pct_mass=np.nan,
                           gini=np.nan)
                p = None

            if prev_p is not None and p is not None:
                jsd = C.jensen_shannon_divergence(prev_p, p, base=2.0)
                k20 = max(1, int(round(0.20 * m_item)))
                head20_now = set(np.argsort(-c)[:k20].tolist())
                jac = len(head20_prev & head20_now) / max(1, len(head20_prev | head20_now))
                temporal_rows.append(dict(dataset=dataset_name, snapshot_idx=si, t_days=float(t),
                                           jsd_bits=jsd, head20_jaccard=jac, head20_turnover=1 - jac))
                head20_prev = head20_now
            elif p is not None:
                k20 = max(1, int(round(0.20 * m_item)))
                head20_prev = set(np.argsort(-c)[:k20].tolist())
            prev_p = p

            per_snapshot_rows.append(row)

        if curve_list:
            curve_arr = np.array(curve_list)
            curves[dataset_name] = dict(grid_pct=grid_pct, mean=curve_arr.mean(0), std=curve_arr.std(0))

    snap_df = pd.DataFrame(per_snapshot_rows)
    snap_df.to_csv(f"{C.RESULTS_DIR}/empirical_popularity_snapshots.csv", index=False)

    summary_rows = []
    for dataset_name in C.DATASETS:
        sub = snap_df[snap_df.dataset == dataset_name]
        row = dict(dataset=dataset_name, n_snapshots=len(sub))
        for col in ["H_norm", "N_eff", "N_eff_over_V", "top1pct_mass", "top5pct_mass", "top20pct_mass", "gini"]:
            row[f"{col}_mean"] = sub[col].mean()
            row[f"{col}_std"] = sub[col].std()
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{C.RESULTS_DIR}/empirical_popularity_summary.csv", index=False)

    temporal_df = pd.DataFrame(temporal_rows)
    temporal_df.to_csv(f"{C.RESULTS_DIR}/temporal_shift_metrics.csv", index=False)
    temporal_summary = temporal_df.groupby("dataset")[["jsd_bits", "head20_turnover"]].agg(["mean", "std"])
    temporal_summary.to_csv(f"{C.RESULTS_DIR}/temporal_shift_summary.csv")

    np.savez(f"{C.RESULTS_DIR}/empirical_popularity_curves.npz",
              **{f"{k}_grid": v["grid_pct"] for k, v in curves.items()},
              **{f"{k}_mean": v["mean"] for k, v in curves.items()},
              **{f"{k}_std": v["std"] for k, v in curves.items()})

    print("\n=== Summary (mean over snapshots) ===")
    print(summary_df[["dataset", "H_norm_mean", "N_eff_over_V_mean", "top5pct_mass_mean", "gini_mean"]]
          .to_string(index=False))
    print("\n=== Temporal instability (mean over consecutive snapshot pairs) ===")
    print(temporal_summary.to_string())
    print(f"\nsaved to {C.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
