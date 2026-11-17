"""Streaming popularity states used by the Hawkes-fit analyses and eval_popularity_combo.py (numpy only).

Queries must be visited in increasing time.  EventStream = all interactions of the data set sorted by time;
HawkesState = lambda_v(t) = mu_v + alpha_v * sum_{t_j<t} exp(-beta (t - t_j)) by recursion over (a subset of) the events;
WindowCounter = per-item event count in [t+lo, t+hi).
"""
import math

import numpy as np


class EventStream:
    def __init__(self, ds, time_len=50):
        T, I = [], []
        for _, d in ds.time_dict.items():
            for i, t in d.items():
                T.append(t / 86400.0)
                I.append(i)
        order = np.argsort(np.array(T), kind="stable")
        self.T = np.array(T, dtype=np.float64)[order]
        self.I = np.array(I, dtype=np.int64)[order]
        # events kept in item_time_array = the last `time_len` events of every item (over the whole data set)
        keep = np.zeros(len(self.T), dtype=bool)
        seen = np.zeros(ds.m_item, dtype=np.int64)
        for k in range(len(self.T) - 1, -1, -1):
            v = self.I[k]
            if seen[v] < time_len:
                keep[k] = True
            seen[v] += 1
        self.keep_deployed = keep
        self.Tl, self.Il = self.T.tolist(), self.I.tolist()


class HawkesState:
    """lambda_v(t) = mu_v + alpha_v * h_v(t),  h_v(t) = sum_{j: t_j < t, kept} exp(-beta (t - t_j)), by recursion."""

    def __init__(self, mu, alpha, beta, stream, keep):
        self.mu, self.alpha, self.beta = mu, alpha, beta
        self.stream, self.keep = stream, keep.tolist()
        m = len(mu)
        self.H, self.last, self.p = np.zeros(m), np.zeros(m), 0

    def advance(self, t):
        Tl, Il, keep, H, last, b = self.stream.Tl, self.stream.Il, self.keep, self.H, self.last, self.beta
        p, n = self.p, len(Tl)
        while p < n and Tl[p] < t:
            if keep[p]:
                v, tj = Il[p], Tl[p]
                H[v] = H[v] * math.exp(-b * (tj - last[v])) + 1.0
                last[v] = tj
            p += 1
        self.p = p

    def weights(self, t, own=None):
        return self.mu + self.alpha * self.H * np.exp(-self.beta * (t - self.last))


class WindowCounter:
    """cnt[v] = # events of v in [t+lo, t+hi)."""

    def __init__(self, stream, m, lo, hi):
        self.stream, self.lo, self.hi = stream, lo, hi
        self.cnt = np.zeros(m, dtype=np.int64)
        self.pa = self.pr = 0

    def advance(self, t):
        Tl, Il, cnt = self.stream.Tl, self.stream.Il, self.cnt
        n = len(Tl)
        pa, pr = self.pa, self.pr
        hi_t, lo_t = t + self.hi, t + self.lo
        while pa < n and Tl[pa] < hi_t:
            cnt[Il[pa]] += 1
            pa += 1
        while pr < n and Tl[pr] < lo_t:
            cnt[Il[pr]] -= 1
            pr += 1
        self.pa, self.pr = pa, pr
