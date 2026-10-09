import gc
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import load_cache
from .metrics import REGIONS, ImageEvaluator, boundary_band, ece_brier
from .model import UNet, enable_mc_dropout
from .utils import claim, get_device, log

DATASET_KEYS = ("isic", "ph2")


@torch.no_grad()
def mc_snapshots(model, x, t_list, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    total = torch.zeros_like(x[:, :1], dtype=torch.float64)
    total_sq = torch.zeros_like(total)
    snapshots = {}
    for t in range(1, max(t_list) + 1):
        p = torch.sigmoid(model(x)).double()
        total += p
        total_sq += p * p
        if t in t_list:
            mean = total / t
            variance = (total_sq / t - mean * mean).clamp_min(0)
            snapshots[t] = (mean.cpu().numpy()[:, 0], variance.cpu().numpy()[:, 0])
    return snapshots


def output_paths(out_dir, dataset):
    return {
        "per_image": out_dir / f"{dataset}_per_image.csv",
        "bins": out_dir / f"{dataset}_bins.npz",
        "summary": out_dir / f"{dataset}_summary.csv",
        "sweep": out_dir / f"{dataset}_T_sweep.npz",
    }


def infer_one(cfg, evaluator, strategy, seed, dataset, data, out_dir, device):
    paths = output_paths(out_dir, dataset)
    sweep = seed in cfg.inference.t_sweep_seeds
    required = [paths["per_image"], paths["bins"], paths["summary"]] + ([paths["sweep"]] if sweep else [])
    if all(p.exists() for p in required):
        log(f"[SKIP] {strategy}/seed_{seed}/{dataset}")
        return

    imgs, masks, names = data
    weights = Path(cfg.paths.output_root) / "train" / strategy / f"seed_{seed}" / "best_weights.pth"
    model = UNet(dropout_p=cfg.model.dropout).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    enable_mc_dropout(model)

    t_main = cfg.inference.mc_passes
    t_list = sorted(set(cfg.inference.t_sweep) | {t_main}) if sweep else [t_main]
    batch_size = cfg.inference.batch_size
    rows, all_bins, all_hist = [], [], []
    sweep_rows = {t: [] for t in t_list}
    sweep_bins = {t: [] for t in t_list}
    started = time.time()

    for batch_index, start in enumerate(range(0, len(names), batch_size)):
        x = torch.from_numpy(np.ascontiguousarray(imgs[start:start + batch_size]))
        x = x.permute(0, 3, 1, 2).float().div(255.0).to(device)
        snapshots = mc_snapshots(model, x, t_list, cfg.inference.mc_seed + batch_index)
        for j in range(x.shape[0]):
            gt = np.asarray(masks[start + j]) > 0.5
            band = boundary_band(gt, cfg.inference.boundary_px)
            for t in t_list:
                p, v = snapshots[t][0][j], snapshots[t][1][j]
                full = t == t_main
                row, bins, hist = evaluator(p, v, gt, band, full=full)
                if full:
                    rows.append({"image": names[start + j], **row})
                    all_bins.append(bins)
                    all_hist.append(hist)
                sweep_rows[t].append([row["dice"], row["variance"], row["entropy"]])
                sweep_bins[t].append(bins)

    frame = pd.DataFrame(rows)
    frame.to_csv(paths["per_image"], index=False, float_format="%.6f")
    bins, hist = np.stack(all_bins), np.stack(all_hist)
    np.savez_compressed(paths["bins"], names=np.array(names), bins=bins, regions=np.array(REGIONS),
                        err_hist=hist, ent_edges=evaluator.entropy_edges,
                        var_edges=evaluator.variance_edges)
    if sweep:
        np.savez_compressed(paths["sweep"], names=np.array(names), T=np.array(t_list),
                            per_image=np.stack([np.array(sweep_rows[t]) for t in t_list], 1),
                            bins=np.stack([np.stack(sweep_bins[t]) for t in t_list], 1),
                            regions=np.array(REGIONS))

    ece, brier = ece_brier(bins.sum(0))
    summary = {"strategy": strategy, "seed": seed, "dataset": dataset, "n": len(frame),
               "mc_passes": t_main, "dice": frame.dice.mean(), "iou": frame.iou.mean(),
               "hd95_mean": frame.hd95.mean(), "hd95_median": frame.hd95.median(),
               "n_empty_pred": int(frame.empty_pred.sum()), "variance": frame.variance.mean(),
               "entropy": frame.entropy.mean(),
               **{f"ece_{r}": ece[k] for k, r in enumerate(REGIONS)},
               **{f"brier_{r}": brier[k] for k, r in enumerate(REGIONS)}}
    pd.DataFrame([summary]).to_csv(paths["summary"], index=False, float_format="%.6f")
    log(f"[{strategy}/seed{seed}/{dataset}] Dice {summary['dice']:.4f} "
        f"HD95 {summary['hd95_median']:.2f} (median) ECE all {ece[0]:.4f} fg {ece[1]:.4f} "
        f"bg {ece[2]:.4f} bd {ece[3]:.4f} Brier {brier[0]:.4f} | T={t_list} "
        f"{time.time() - started:.0f}s")
    del model
    torch.cuda.empty_cache()
    gc.collect()


def run_worker(cfg, worker_id):
    device = get_device()
    evaluator = ImageEvaluator(cfg.inference.ece_bins, cfg.inference.err_hist_bins)
    data = {key: load_cache(key, cfg.paths.cache_dir) for key in DATASET_KEYS}
    log(f"worker {worker_id} on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    for seed in cfg.experiment.seeds:
        for strategy in cfg.experiment.strategies:
            weights = Path(cfg.paths.output_root) / "train" / strategy / f"seed_{seed}" / "best_weights.pth"
            out_dir = Path(cfg.paths.output_root) / "infer" / strategy / f"seed_{seed}"
            if not weights.exists():
                log(f"[MISSING] {weights}")
                continue
            if not claim(out_dir):
                continue
            for dataset in DATASET_KEYS:
                try:
                    infer_one(cfg, evaluator, strategy, seed, dataset, data[dataset], out_dir, device)
                except Exception:
                    log(f"[FAILED] {strategy}/seed_{seed}/{dataset}\n{traceback.format_exc()}")
                    torch.cuda.empty_cache()
                    gc.collect()
    log(f"worker {worker_id} finished")
