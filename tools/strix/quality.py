"""Quality metrics: KL divergence, perplexity/NLL, top-k agreement.

Matched-token teacher forcing per docs/TESTING.md.
"""

from __future__ import annotations

import numpy as np


def log_softmax(z, axis=-1):
    m = z.max(axis=axis, keepdims=True)
    zz = z - m
    return zz - np.log(np.exp(zz).sum(axis=axis, keepdims=True))


def softmax(z, axis=-1):
    zz = z - z.max(axis=axis, keepdims=True)
    return np.exp(zz) / np.exp(zz).sum(axis=axis, keepdims=True)


def kl_divergence(p_t, p_c, axis=-1):
    # KL = sum p_t * (log p_t - log p_c)
    return np.sum(p_t * (np.log(p_t) - np.log(p_c)), axis=axis)


def compare_logits(teacher_logits, candidate_logits, target_tokens=None):
    """Distribution metrics between teacher and candidate logit vectors.

    teacher_logits, candidate_logits: float32 numpy [positions, vocab].
    Returns dict of aggregates.
    """
    T, V = teacher_logits.shape
    lp_t = log_softmax(teacher_logits)
    lp_c = log_softmax(candidate_logits)
    p_t = np.exp(lp_t)
    p_c = np.exp(lp_c)
    kl = kl_divergence(p_t, p_c)

    top1_t = np.argmax(teacher_logits, axis=1)
    top1_c = np.argmax(candidate_logits, axis=1)
    top5_t = np.argsort(teacher_logits, axis=1)[:, -5:]
    top5_c = np.argsort(candidate_logits, axis=1)[:, -5:]

    top1_agree = (top1_t == top1_c).mean()
    # top-5 set overlap per position
    top5_overlap = []
    for i in range(T):
        s_t = set(top5_t[i].tolist())
        s_c = set(top5_c[i].tolist())
        top5_overlap.append(len(s_t & s_c) / 5.0)
    top5_agree = float(np.mean(top5_overlap))

    # teacher top-1 prob assigned by candidate
    cand_prob_on_t1 = p_c[np.arange(T), top1_t]

    nll = -np.log(np.clip(p_c[np.arange(T), target_tokens], 1e-30, 1.0)) if target_tokens is not None else None

    metrics = {
        "positions": T,
        "kl_mean": float(kl.mean()),
        "kl_median": float(np.median(kl)),
        "kl_p95": float(np.percentile(kl, 95)),
        "kl_p99": float(np.percentile(kl, 99)),
        "kl_max": float(kl.max()),
        "top1_agreement": float(top1_agree),
        "top5_agreement": float(top5_agree),
        "teacher_top1_candidate_prob_mean": float(cand_prob_on_t1.mean()),
        "nonfinite_teacher": int(np.isnan(teacher_logits).sum() + np.isinf(teacher_logits).sum()),
        "nonfinite_candidate": int(np.isnan(candidate_logits).sum() + np.isinf(candidate_logits).sum()),
    }
    if nll is not None:
        metrics["nll_mean"] = float(nll.mean())
        metrics["perplexity"] = float(np.exp(nll.mean()))
    return metrics
