import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

REGIONS = ["all", "fg", "bg", "boundary"]


@torch.no_grad()
def per_image_dice_iou(logits, targets, eps=1e-6):
    pred = (torch.sigmoid(logits) > 0.5).float().flatten(1)
    target = (targets > 0.5).float().flatten(1)
    intersection, total = (pred * target).sum(1), pred.sum(1) + target.sum(1)
    dice = (2 * intersection + eps) / (total + eps)
    iou = (intersection + eps) / (total - intersection + eps)
    return dice.cpu().numpy(), iou.cpu().numpy()


def hd95(pred, gt):
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan
    pred_edge = pred & (distance_transform_edt(pred) <= 1)
    gt_edge = gt & (distance_transform_edt(gt) <= 1)
    to_gt = distance_transform_edt(~gt_edge)
    to_pred = distance_transform_edt(~pred_edge)
    return float(np.percentile(np.concatenate([to_gt[pred_edge], to_pred[gt_edge]]), 95))


def boundary_band(gt, pixels):
    inside = gt & (distance_transform_edt(gt) <= pixels)
    outside = ~gt & (distance_transform_edt(~gt) <= pixels)
    return inside | outside


def bin_stats(p, y, n_bins):
    idx = np.minimum((p * n_bins).astype(np.int64), n_bins - 1)
    return np.stack([np.bincount(idx, minlength=n_bins),
                     np.bincount(idx, weights=p, minlength=n_bins),
                     np.bincount(idx, weights=y, minlength=n_bins),
                     np.bincount(idx, weights=(p - y) ** 2, minlength=n_bins)], axis=1)


def ece_brier(bins):
    n = np.maximum(bins[..., 0].sum(-1), 1)
    return np.abs(bins[..., 1] - bins[..., 2]).sum(-1) / n, bins[..., 3].sum(-1) / n


def entropy_map(p):
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    return -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped))


def region_masks(pred, gt, band):
    foreground = pred | gt
    return [np.ones_like(gt), foreground, ~foreground, band]


def score_edges(hist_bins):
    entropy_edges = np.concatenate([[0.0], np.logspace(-10, np.log10(np.log(2)), hist_bins - 1)])
    variance_edges = np.concatenate([[0.0], np.logspace(-12, np.log10(0.25), hist_bins - 1)])
    return entropy_edges, variance_edges


class ImageEvaluator:
    def __init__(self, n_bins, hist_bins):
        self.n_bins = n_bins
        self.hist_bins = hist_bins
        self.entropy_edges, self.variance_edges = score_edges(hist_bins)

    def __call__(self, p, v, gt, band, full=True):
        pred = p > 0.5
        intersection, total = (pred & gt).sum(), pred.sum() + gt.sum()
        entropy = entropy_map(p)
        target = gt.astype(np.float64)
        masks = region_masks(pred, gt, band)
        bins = np.stack([bin_stats(p[m], target[m], self.n_bins) for m in masks])
        ece, brier = ece_brier(bins)
        row = {"dice": (2 * intersection + 1e-6) / (total + 1e-6),
               "variance": float(v.mean()), "entropy": float(entropy.mean())}
        if not full:
            return row, bins, None
        row.update({"iou": (intersection + 1e-6) / (total - intersection + 1e-6),
                    "hd95": hd95(pred, gt), "empty_pred": int(pred.sum() == 0)})
        for k, region in enumerate(REGIONS):
            mask = masks[k]
            row[f"variance_{region}"] = float(v[mask].mean()) if mask.any() else np.nan
            row[f"entropy_{region}"] = float(entropy[mask].mean()) if mask.any() else np.nan
            row[f"ece_{region}"], row[f"brier_{region}"] = ece[k], brier[k]
        errors = (pred != gt).ravel()
        hist = np.zeros((2, 2, self.hist_bins), np.uint32)
        scores = [(entropy.ravel(), self.entropy_edges), (v.ravel(), self.variance_edges)]
        for s, (score, edges) in enumerate(scores):
            idx = np.clip(np.searchsorted(edges, score, side="right") - 1, 0, self.hist_bins - 1)
            hist[s, 0] = np.bincount(idx[~errors], minlength=self.hist_bins)
            hist[s, 1] = np.bincount(idx[errors], minlength=self.hist_bins)
        return row, bins, hist
