import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import REGIONS, ece_brier
from .stat_tools import auroc_auprc_from_hist, aurc_weighted, auroc_weighted, holm
from .utils import log

SCORES = ("entropy", "variance")
MEAN_METRICS = ["dice", "iou", "hd95", "variance", "entropy"]
POOLED_METRICS = [f"{k}_{r}" for k in ("ece", "brier") for r in REGIONS]
PIXEL_METRICS = [f"pix_{m}_{s}" for m in ("auroc", "auprc") for s in SCORES]
IMAGE_METRICS = [f"aurc_{s}" for s in SCORES] + [f"img_auroc_{s}" for s in SCORES]
ALL_METRICS = MEAN_METRICS + POOLED_METRICS + PIXEL_METRICS + IMAGE_METRICS


def load_results(infer_root, datasets, strategies, seeds):
    missing = [f"{s}/seed_{seed}/{ds}" for ds in datasets for s in strategies for seed in seeds
               if not (infer_root / s / f"seed_{seed}" / f"{ds}_bins.npz").exists()]
    if missing:
        raise FileNotFoundError(
            f"Inference not finished for {len(missing)} run(s), e.g. {missing[:5]}. "
            "Run scripts/infer.py first.")
    data = {}
    for ds in datasets:
        reference = None
        for strategy in strategies:
            frames, bins, hists = [], [], []
            for seed in seeds:
                run_dir = infer_root / strategy / f"seed_{seed}"
                frame = pd.read_csv(run_dir / f"{ds}_per_image.csv")
                archive = np.load(run_dir / f"{ds}_bins.npz")
                names = [str(n) for n in archive["names"]]
                if names != list(frame.image.astype(str)):
                    raise ValueError(f"Row order mismatch in {strategy}/seed_{seed}/{ds}")
                reference = reference or names
                if names != reference:
                    raise ValueError(f"Image set differs in {strategy}/seed_{seed}/{ds}")
                frames.append(frame)
                bins.append(archive["bins"])
                hists.append(archive["err_hist"].astype(np.float64))
            data[(ds, strategy)] = {
                "bins": np.stack(bins), "hist": np.stack(hists),
                **{m: np.stack([f[m].to_numpy(float) for f in frames]) for m in MEAN_METRICS},
            }
        log(f"{ds}: {len(reference)} images x {len(seeds)} seeds x {len(strategies)} strategies")
    return data


def seed_metrics(entry, n_seeds, fail_dice, weights=None):
    out = {}
    for metric in MEAN_METRICS:
        values = entry[metric]
        valid = ~np.isnan(values)
        filled = np.where(valid, values, 0.0)
        if weights is None:
            out[metric] = filled.sum(1) / np.maximum(valid.sum(1), 1)
        else:
            out[metric] = (weights @ filled.T) / np.maximum(weights @ valid.T.astype(float), 1)

    bins = entry["bins"]
    flat = bins.reshape(bins.shape[0], bins.shape[1], -1)
    pooled = flat.sum(1) if weights is None else np.einsum("kn,snf->ksf", weights, flat)
    ece, brier = ece_brier(pooled.reshape(*pooled.shape[:-1], len(REGIONS), -1, 4))
    for k, region in enumerate(REGIONS):
        out[f"ece_{region}"], out[f"brier_{region}"] = ece[..., k], brier[..., k]

    hist = entry["hist"]
    flat_hist = hist.reshape(hist.shape[0], hist.shape[1], -1)
    pooled_hist = flat_hist.sum(1) if weights is None else np.einsum("kn,snf->ksf", weights, flat_hist)
    pooled_hist = pooled_hist.reshape(*pooled_hist.shape[:-1], 2, 2, -1)
    auroc, auprc = auroc_auprc_from_hist(pooled_hist[..., 0, :], pooled_hist[..., 1, :])
    for k, score in enumerate(SCORES):
        out[f"pix_auroc_{score}"], out[f"pix_auprc_{score}"] = auroc[..., k], auprc[..., k]

    n_images = entry["dice"].shape[1]
    image_weights = np.ones((1, n_images)) if weights is None else weights
    for score in SCORES:
        aurc = np.array([[aurc_weighted(1 - entry["dice"][s], entry[score][s], w)
                          for s in range(n_seeds)] for w in image_weights])
        image_auroc = np.array([[auroc_weighted(entry["dice"][s] < fail_dice, entry[score][s], w)
                                 for s in range(n_seeds)] for w in image_weights])
        out[f"aurc_{score}"] = aurc[0] if weights is None else aurc
        out[f"img_auroc_{score}"] = image_auroc[0] if weights is None else image_auroc
    return out


def run_bootstrap(data, datasets, strategies, seeds, n_boot, boot_seed, fail_dice):
    n_seeds = len(seeds)
    per_run, point, boot = [], {}, {}
    for ds in datasets:
        n_images = data[(ds, strategies[0])]["dice"].shape[1]
        rng = np.random.default_rng(boot_seed)
        image_draws = rng.multinomial(n_images, np.full(n_images, 1.0 / n_images), size=n_boot).astype(float)
        seed_draws = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
        for strategy in strategies:
            entry = data[(ds, strategy)]
            point[(ds, strategy)] = seed_metrics(entry, n_seeds, fail_dice)
            resampled = seed_metrics(entry, n_seeds, fail_dice, image_draws)
            boot[(ds, strategy)] = {
                m: np.nanmean(np.take_along_axis(resampled[m], seed_draws, 1), 1) for m in ALL_METRICS}
            for k, seed in enumerate(seeds):
                per_run.append({"dataset": ds, "strategy": strategy, "seed": seed,
                                **{m: point[(ds, strategy)][m][k] for m in ALL_METRICS}})
            log(f"bootstrap done: {ds}/{strategy}")
    return pd.DataFrame(per_run), point, boot


def confidence_intervals(data, point, boot):
    rows = []
    for (ds, strategy), metrics in point.items():
        for metric in ALL_METRICS:
            low, high = np.nanpercentile(boot[(ds, strategy)][metric], [2.5, 97.5])
            rows.append({"dataset": ds, "strategy": strategy, "metric": metric,
                         "mean": np.nanmean(metrics[metric]),
                         "seed_sd": np.nanstd(metrics[metric], ddof=1),
                         "ci_low": low, "ci_high": high})
        hd95_per_image = np.nanmean(data[(ds, strategy)]["hd95"], 0)
        q1, median, q3 = np.nanpercentile(hd95_per_image, [25, 50, 75])
        rows.append({"dataset": ds, "strategy": strategy, "metric": "hd95_median",
                     "mean": median, "seed_sd": np.nan, "ci_low": q1, "ci_high": q3})
    return pd.DataFrame(rows)


def paired_differences(point, boot, datasets, strategies, n_seeds, alpha):
    rows = []
    for ds in datasets:
        for metric in ALL_METRICS:
            block = []
            for a, b in combinations(strategies, 2):
                diff = boot[(ds, a)][metric] - boot[(ds, b)][metric]
                diff = diff[~np.isnan(diff)]
                low, high = np.percentile(diff, [2.5, 97.5]) if len(diff) else (np.nan, np.nan)
                point_a, point_b = point[(ds, a)][metric], point[(ds, b)][metric]
                p_boot = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean())) if len(diff) else np.nan
                block.append({"dataset": ds, "metric": metric, "a": a, "b": b,
                              "a_minus_b": np.nanmean(point_a) - np.nanmean(point_b),
                              "ci_low": low, "ci_high": high, "p_boot": p_boot,
                              "seeds_a_lt_b": f"{int((point_a < point_b).sum())}/{n_seeds}"})
            for row, p_holm in zip(block, holm([r["p_boot"] for r in block])):
                row["p_holm"] = p_holm
                row["significant"] = bool(p_holm < alpha)
            rows += block
    return pd.DataFrame(rows)


def friedman_wilcoxon(data, datasets, strategies):
    rows = []
    pairs = list(combinations(range(len(strategies)), 2))
    for ds in datasets:
        for metric in ["dice", "iou", "hd95"]:
            matrix = np.stack([np.nanmean(data[(ds, s)][metric], 0) for s in strategies], 1)
            matrix = matrix[~np.isnan(matrix).any(1)]
            if len(strategies) >= 3:
                statistic, p = stats.friedmanchisquare(*matrix.T)
                rows.append({"dataset": ds, "metric": metric, "test": "friedman", "a": "all", "b": "",
                             "stat": statistic, "p": p, "p_holm": p, "n_images": len(matrix)})
            results = [stats.wilcoxon(matrix[:, i], matrix[:, j]) for i, j in pairs]
            adjusted = holm([r.pvalue for r in results])
            for (i, j), result, p_holm in zip(pairs, results, adjusted):
                rows.append({"dataset": ds, "metric": metric, "test": "wilcoxon",
                             "a": strategies[i], "b": strategies[j], "stat": result.statistic,
                             "p": result.pvalue, "p_holm": p_holm, "n_images": len(matrix)})
    return pd.DataFrame(rows)


def t_sensitivity(infer_root, datasets, strategies, seeds):
    rows = []
    for ds in datasets:
        for strategy in strategies:
            for seed in seeds:
                path = infer_root / strategy / f"seed_{seed}" / f"{ds}_T_sweep.npz"
                if not path.exists():
                    continue
                archive = np.load(path)
                ece, brier = ece_brier(archive["bins"].sum(0))
                for k, t in enumerate(archive["T"]):
                    rows.append({
                        "dataset": ds, "strategy": strategy, "seed": seed, "T": int(t),
                        "dice": archive["per_image"][:, k, 0].mean(),
                        "variance": archive["per_image"][:, k, 1].mean(),
                        "entropy": archive["per_image"][:, k, 2].mean(),
                        **{f"ece_{r}": ece[k, i] for i, r in enumerate(REGIONS)},
                        **{f"brier_{r}": brier[k, i] for i, r in enumerate(REGIONS)}})
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["dataset", "strategy", "T"]).agg(["mean", "std"]).drop(columns="seed")
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def training_summary(root, strategies, seeds):
    rows = []
    for strategy in strategies:
        for seed in seeds:
            path = root / "train" / strategy / f"seed_{seed}" / "training_log.csv"
            if not path.exists():
                continue
            history = pd.read_csv(path)
            best = history.val_dice.idxmax()
            final_train, final_val = history.train_dice.iloc[-1], history.val_dice.iloc[-1]
            rows.append({"strategy": strategy, "seed": seed, "best_epoch": int(history.epoch[best]),
                         "best_val_dice": history.val_dice[best], "final_val_dice": final_val,
                         "drop_after_best": history.val_dice[best] - final_val,
                         "final_train_dice": final_train,
                         "final_gap_train_minus_val": final_train - final_val})
    return pd.DataFrame(rows)


def significance_marker(diffs, dataset, strategy, metric, reference):
    if strategy == reference:
        return ""
    pair = (((diffs.a == reference) & (diffs.b == strategy)) |
            ((diffs.a == strategy) & (diffs.b == reference)))
    selected = diffs[(diffs.dataset == dataset) & (diffs.metric == metric) & pair]
    return "†" if len(selected) and selected.significant.iloc[0] else ""


def format_cell(ci, diffs, dataset, strategy, metric, reference, with_ci):
    row = ci[(ci.dataset == dataset) & (ci.strategy == strategy) & (ci.metric == metric)].iloc[0]
    fmt = "{:.2f}" if metric.startswith("hd95") else "{:.4f}"
    if metric == "hd95_median":
        return f"{fmt.format(row['mean'])} [{fmt.format(row.ci_low)}–{fmt.format(row.ci_high)}]"
    if with_ci:
        text = f"{fmt.format(row['mean'])} [{fmt.format(row.ci_low)}, {fmt.format(row.ci_high)}]"
    else:
        text = f"{fmt.format(row['mean'])} ± {fmt.format(row.seed_sd)}"
    return text + significance_marker(diffs, dataset, strategy, metric, reference)


def make_table(ci, diffs, datasets, strategies, reference, columns, ci_columns):
    rows = []
    for ds in datasets:
        for strategy in strategies:
            rows.append({"dataset": ds, "strategy": strategy,
                         **{c: format_cell(ci, diffs, ds, strategy, c, reference, c in ci_columns)
                            for c in columns}})
    return pd.DataFrame(rows)


def run_analysis(cfg):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    root = Path(cfg.paths.output_root)
    infer_root, out_dir = root / "infer", root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    strategies, seeds, datasets = cfg.experiment.strategies, cfg.experiment.seeds, cfg.experiment.datasets
    options = cfg.statistics

    data = load_results(infer_root, datasets, strategies, seeds)
    per_run, point, boot = run_bootstrap(data, datasets, strategies, seeds, options.n_boot,
                                         options.boot_seed, options.fail_dice)
    per_run.to_csv(out_dir / "per_run.csv", index=False, float_format="%.6f")

    ci = confidence_intervals(data, point, boot)
    ci.to_csv(out_dir / "bootstrap_ci.csv", index=False, float_format="%.6f")
    diffs = paired_differences(point, boot, datasets, strategies, len(seeds), options.alpha)
    diffs.to_csv(out_dir / "paired_diffs.csv", index=False, float_format="%.6f")
    friedman_wilcoxon(data, datasets, strategies).to_csv(
        out_dir / "friedman_wilcoxon.csv", index=False, float_format="%.6g")

    sensitivity = t_sensitivity(infer_root, datasets, strategies, seeds)
    if sensitivity is not None:
        sensitivity.to_csv(out_dir / "T_sensitivity.csv", index=False, float_format="%.6f")
    training = training_summary(root, strategies, seeds)
    if len(training):
        training.to_csv(out_dir / "training_summary.csv", index=False, float_format="%.4f")

    reference = options.reference
    tables = {
        "table_II_segmentation": make_table(
            ci, diffs, datasets, strategies, reference,
            ["dice", "iou", "hd95", "hd95_median"], {"dice", "hd95"}),
        "table_III_calibration": make_table(
            ci, diffs, datasets, strategies, reference,
            POOLED_METRICS + ["variance", "entropy"],
            {"ece_all", "ece_fg", "ece_bg", "ece_boundary", "brier_all"}),
        "table_IV_usefulness": make_table(
            ci, diffs, datasets, strategies, reference,
            IMAGE_METRICS + PIXEL_METRICS, {"aurc_entropy", "pix_auroc_entropy"}),
    }
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    return {"tables": tables, "diffs": diffs, "training": training, "out_dir": out_dir}


def print_report(report, reference, alpha):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 40)
    for name, table in report["tables"].items():
        print(f"\n{name} (mean [95% CI] or mean +/- seed SD; dagger = differs from {reference}, "
              f"Holm p < {alpha})")
        print(table.to_string(index=False))
    diffs = report["diffs"]
    selected = diffs[(diffs.a == "mixing") & (diffs.b == "occlusion") &
                     diffs.metric.isin(["ece_all", "ece_fg", "ece_boundary", "dice"])]
    if len(selected):
        print("\nmixing - occlusion")
        print(selected[["dataset", "metric", "a_minus_b", "ci_low", "ci_high", "p_holm",
                        "seeds_a_lt_b"]].to_string(index=False))
    training = report["training"]
    if len(training):
        columns = ["best_epoch", "best_val_dice", "drop_after_best", "final_gap_train_minus_val"]
        print("\ntraining (best epoch and drop after best)")
        print(training.groupby("strategy")[columns].agg(["mean", "std"]).round(3).to_string())
    print(f"\nAll tables saved in {report['out_dir']}")
