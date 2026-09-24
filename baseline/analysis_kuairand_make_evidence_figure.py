"""Render the 4-panel evidence figure for why KuaiRand's Hawkes popularity prior runs
away (Sec. 3b/3c of kuairand_tail_diagnosis_summary.md). Requires
analysis_kuairand_gather_3dataset_evidence.py to have been run first.

Usage: python analysis_kuairand_make_evidence_figure.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO_ROOT, "analysis_representation", "tail_diag")
FIG_DIR = os.path.join(REPO_ROOT, "figures")

with open(f"{OUT}/three_dataset_evidence.json") as f:
    data = json.load(f)

names = ["kuairand", "micro_video", "ml-1m"]
labels = ["KuaiRand", "Micro-Video", "MovieLens"]
colors = ["#d62728", "#1f77b4", "#2ca02c"]

fig, axes = plt.subplots(2, 2, figsize=(11, 9))

# Panel A: log-log scatter of raw count vs learned mu, all three datasets overlaid
ax = axes[0, 0]
for name, label, color in zip(names, labels, colors):
    raw = np.array(data[name]["raw_count"])
    mu = np.array(data[name]["mu"])
    mask = raw > 0
    ax.scatter(raw[mask], mu[mask], s=6, alpha=0.35, color=color, label=label, edgecolors="none")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Raw training interaction count (item)")
ax.set_ylabel(r"Learned Hawkes background rate $\mu_v$")
ax.set_title("(a) Learned $\\mu_v$ vs. raw popularity, per item")
ax.legend(fontsize=9, markerscale=3)
ax.grid(alpha=0.25)

# Panel B: mu concentration -- top-10 share of total mu mass
ax = axes[0, 1]
shares = [data[n]["top10_mu_share_pct"] for n in names]
bars = ax.bar(labels, shares, color=colors)
ax.set_ylabel("Top-10 items' share of total $\\mu$ mass (%)")
ax.set_title("(b) Popularity-mass concentration")
for b, v in zip(bars, shares):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10)
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.25)

# Panel C: learned beta half-life (log scale, hours)
ax = axes[1, 0]
half_life_hours = [np.log(2) / data[n]["beta"] * 24 for n in names]
median_gap_hours = [data[n]["median_gap_hours"] for n in names]
x = np.arange(len(names))
w = 0.35
b1 = ax.bar(x - w / 2, half_life_hours, w, label="Learned excitation half-life", color="#7f7f7f")
b2 = ax.bar(x + w / 2, median_gap_hours, w, label="Median same-item re-interaction gap", color="#bcbd22")
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Hours (log scale)")
ax.set_title("(c) Hawkes decay timescale vs. data's own re-interaction gap")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.25)

# Panel D: usable excitation window under the fixed last-50-events-globally cap,
# for the single top-mu item in each dataset (in absolute days, log scale)
ax = axes[1, 1]
vals = [data[n]["last50_window_days"] for n in names]
bars = ax.bar(labels, vals, color=colors)
ax.set_yscale("log")
ax.set_ylim(0.1, 5000)
ax.set_ylabel("Days spanned by the retained last-50 events (log)")
ax.set_title("(d) Usable excitation window for the top-$\\mu$ item\n(fixed 50-event-per-item history cap)")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.5, f"{v:.1f}d", ha="center", fontsize=10)
ax.grid(axis="y", alpha=0.25)

fig.suptitle("Why KuaiRand's Hawkes popularity prior runs away (SASRec+Ours, all three datasets)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])

fig.savefig(f"{FIG_DIR}/fig_kuairand_mu_runaway_evidence.pdf")
fig.savefig(f"{FIG_DIR}/fig_kuairand_mu_runaway_evidence.png", dpi=150)
print("saved figures to", FIG_DIR)
