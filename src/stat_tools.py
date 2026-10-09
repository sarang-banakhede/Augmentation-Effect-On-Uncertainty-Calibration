import numpy as np
from scipy import stats


def auroc_auprc_from_hist(neg, pos):
    tp = np.cumsum(pos[..., ::-1], -1)
    fp = np.cumsum(neg[..., ::-1], -1)
    n_pos = np.maximum(tp[..., -1:], 1)
    n_neg = np.maximum(fp[..., -1:], 1)
    tpr, fpr = tp / n_pos, fp / n_neg
    zero = np.zeros_like(tpr[..., :1])
    tpr0, fpr0 = np.concatenate([zero, tpr], -1), np.concatenate([zero, fpr], -1)
    auroc = ((fpr0[..., 1:] - fpr0[..., :-1]) * (tpr0[..., 1:] + tpr0[..., :-1]) / 2).sum(-1)
    precision = tp / np.maximum(tp + fp, 1)
    average_precision = ((tpr0[..., 1:] - tpr0[..., :-1]) * precision).sum(-1)
    return auroc, average_precision


def aurc_weighted(risk, uncertainty, weights):
    order = np.argsort(uncertainty, kind="stable")
    repeated = np.repeat(risk[order], weights[order].astype(int))
    if not len(repeated):
        return np.nan
    return float(np.mean(np.cumsum(repeated) / np.arange(1, len(repeated) + 1)))


def auroc_weighted(label, score, weights):
    labels = np.repeat(label, weights.astype(int))
    scores = np.repeat(score, weights.astype(int))
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = stats.rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def holm(p_values):
    p_values = np.asarray(p_values, float)
    adjusted = np.full(len(p_values), np.nan)
    valid = np.flatnonzero(~np.isnan(p_values))
    m, running = len(valid), 0.0
    for rank, i in enumerate(valid[np.argsort(p_values[valid])]):
        running = max(running, (m - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted
