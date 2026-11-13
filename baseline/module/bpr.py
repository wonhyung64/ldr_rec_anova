"""BPR loss for the loss-function control (counterpart of sampled_softmax_loss).

Reviewer: standard recommenders use BCE / BPR; compare the vanilla backbones
trained with BPR under the same scoring / init / negative sampler as the
sampled-softmax control and "+Ours".
"""
import torch
import torch.nn.functional as F

from .model import score_pair


def bpr_loss(model, candidate_items, hist_items, additional_feat):
    """Mean over (positive, negative_j) pairs of -log sigmoid(s_pos - s_neg_j).

    candidate_items[:, 0] is the observed item, [:, 1:] are M uniform
    negatives (M = contrast_size - 1; pass --contrast-size=2 for classic
    1-negative BPR).  Scores go through model.residual_score, i.e. the same
    cosine/tau scoring as the other runs.
    """
    util = score_pair(model, candidate_items, hist_items, additional_feat)
    pos, neg = util[:, :1], util[:, 1:]
    return -F.logsigmoid(pos - neg).mean()
