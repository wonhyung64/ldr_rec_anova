"""Gather the per-item (raw_count, learned mu) evidence, plus dataset-level timing
statistics (span, beta, re-interaction gaps, 50-event-cap window), for KuaiRand,
Micro-Video and MovieLens -- used by analysis_kuairand_make_evidence_figure.py and
analysis_representation/kuairand_dataset_comparison_table.tex.

See analysis_representation/kuairand_tail_diagnosis_summary.md Sec. 3b/3c.

Usage: python analysis_kuairand_gather_3dataset_evidence.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from collections import Counter, defaultdict
from module.dataset import UserItemTime
from module.model import MODEL_REGISTRY
from module.debias import build_unshared_debias_model

OUT = os.path.join("analysis_representation", "tail_diag")
os.makedirs(OUT, exist_ok=True)


def build_ours(model_name, ds, device="cpu"):
    """Same architecture as analysis_kuairand_tail_diagnosis.load_ours_ckpt, but
    checkpoint-agnostic (that helper's GAMMA_FOR/ALPHA1_FOR/WEIGHTS_OURS are KuaiRand-
    specific; here we load the state dict explicitly per dataset below instead)."""
    model_cls = MODEL_REGISTRY[model_name]
    debiased_cls = build_unshared_debias_model(model_cls)
    return debiased_cls(
        num_users=ds.n_user, num_items=ds.m_item, embedding_k=128,
        device=device, tau=0.1, depth=0, max_seq_len=50, n_heads=1, dropout=0.2,
        score_norm="normalized",
    ).to(device)

DATASETS = {
    "kuairand": dict(
        ckpt="./weights_hawkes_anova_generalize/kuairand/_sasrec_lambdacen0.5_tau0.1_scorenormnormalized_e500_seed1_ablationshared_hawkesanova.pt",
        train_dict="data/kuairand/training_dict.npy",
        time_dict="data/kuairand/interaction_time_dict.npy",
        ms_to_s=True,
    ),
    "micro_video": dict(
        ckpt="./weights_hawkes_anova_generalize/micro_video/_sasrec_lambdacen3.0_tau0.1_scorenormnormalized_e500_seed1_ablationshared_hawkesanova.pt",
        train_dict="data/micro_video/training_dict.npy",
        time_dict="data/micro_video/interaction_time_dict.npy",
        ms_to_s=False,
    ),
    "ml-1m": dict(
        ckpt="./weights_hawkes_anova_generalize/ml-1m/_sasrec_lambdacen0.5_tau0.1_scorenormnormalized_e500_seed1_ablationshared_hawkesanova.pt",
        train_dict="data/ml-1m/training_dict.npy",
        time_dict="data/ml-1m/interaction_time_dict.npy",
        ms_to_s=False,
    ),
}

results = {}
for name, cfg in DATASETS.items():
    ds = UserItemTime("./data", name, "d", 50, 50)
    model = build_ours("sasrec", ds)
    ckpt = torch.load(cfg["ckpt"], map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        mu, alpha, beta = model.prior_parameters_from_embeddings()
    mu = mu.numpy(); alpha = alpha.numpy(); beta = float(beta)

    train_dict = np.load(cfg["train_dict"], allow_pickle=True).item()
    c = Counter()
    for u, items in train_dict.items():
        c.update(items)
    raw_count = np.array([c.get(i, 0) for i in range(ds.m_item)])

    time_dict = np.load(cfg["time_dict"], allow_pickle=True).item()
    if cfg["ms_to_s"]:
        time_dict = {u: {it: t / 1000.0 for it, t in items.items()} for u, items in time_dict.items()}
    item_times = defaultdict(list)
    all_times = []
    for u, items in time_dict.items():
        for it, t in items.items():
            item_times[it].append(t)
            all_times.append(t)
    all_times = np.array(all_times)
    span_days = (all_times.max() - all_times.min()) / 86400.0

    gaps_hours = []
    for it, times in item_times.items():
        times = sorted(times)
        if len(times) < 2:
            continue
        d = np.diff(np.array(times)) / 3600.0
        gaps_hours.extend(d.tolist())
    gaps_hours = np.array(gaps_hours)

    # Days spanned by the retained last-50 raw event timestamps (time_len=50 cap),
    # for the single item with the largest learned mu.
    top_mu_item = int(np.argmax(mu))
    top_mu_item_times_days = np.array(sorted(item_times[top_mu_item])) / 86400.0
    if len(top_mu_item_times_days) >= 50:
        last50_window_days = float(top_mu_item_times_days[-1] - top_mu_item_times_days[-50])
    else:
        last50_window_days = float(top_mu_item_times_days[-1] - top_mu_item_times_days[0])

    order = np.argsort(-mu)
    top10_share = mu[order[:10]].sum() / mu.sum() * 100

    results[name] = dict(
        m_item=int(ds.m_item),
        raw_count=raw_count.tolist(),
        mu=mu.tolist(),
        beta=beta,
        span_days=float(span_days),
        median_gap_hours=float(np.median(gaps_hours)),
        top10_mu_share_pct=float(top10_share),
        mu_max_over_median=float(mu.max() / np.median(mu)),
        raw_count_max_over_median=float(raw_count.max() / np.median(raw_count[raw_count > 0])),
        mean_interactions_per_item=float(raw_count[raw_count > 0].mean()),
        top_item_raw_count=int(raw_count.max()),
        top_mu_item=top_mu_item,
        last50_window_days=last50_window_days,
    )
    print(f"{name}: span={span_days:.1f}d beta={beta:.3f} top10_mu_share={top10_share:.1f}% "
          f"median_gap={np.median(gaps_hours):.2f}h mu_ratio={mu.max()/np.median(mu):.0f}")

with open(f"{OUT}/three_dataset_evidence.json", "w") as f:
    json.dump(results, f)
print("saved", f"{OUT}/three_dataset_evidence.json")
