import math
import torch
import torch.nn.functional as F

from .model import score_pair


def importance_corrected_choice_loss(
    model, candidate_items, hist_items, user_idx, pos_time, batch_time_all, num_items, num_negatives,
):
    """Importance-corrected choice loss l_choice(Phi, Psi).

    p(v|x) = pi(v|t) * exp(f(x,v)) / Z is estimated with an importance-sampled
    normalizer: candidate_items[:, 0] is the observed item and the remaining
    columns are negatives drawn uniformly from V \\ {pos}. Under that uniform
    proposal q^-(v) = 1/(num_items-1), the correction a(v)/q(v) is the same
    constant for every negative, so it collapses to a single additive term on
    the negative logits (the usual sampled-softmax / log-uniform correction).
    """
    lam = model.prior(candidate_items, pos_time, batch_time_all)
    util = score_pair(model, candidate_items, hist_items, user_idx)
    log_a = torch.log(lam + 1e-12) + util

    log_correction = math.log(max(num_items - 1, 1)) - math.log(num_negatives)
    logits = torch.cat([log_a[:, :1], log_a[:, 1:] + log_correction], dim=1)

    return -F.log_softmax(logits, dim=-1)[:, 0].mean()


def anova_centering_penalty(model, candidate_items, hist_a, user_a, hist_b, user_b):
    """Split-sample estimator of the ANOVA-centering penalty R_cen(Psi).

    Two independently sampled users at the same event time give an unbiased
    estimate of E_u[f(x_u(t), v)]^2 via the cross term m_a * m_b, instead of
    the upward-biased square of a single sample.
    """
    m_a = score_pair(model, candidate_items, hist_a, user_a)
    m_b = score_pair(model, candidate_items, hist_b, user_b)
    return (m_a * m_b).mean()
