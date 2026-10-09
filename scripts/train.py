import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import apply_overrides, load_config, save_config  # noqa: E402
from src.data import prepare_train_data  # noqa: E402
from src.training import run_worker  # noqa: E402
from src.utils import clear_claims, launch_workers, log  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train all augmentation strategies and seeds.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--strategies", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--resume-from", help="outputs folder of an interrupted session")
    parser.add_argument("--worker", type=int, help="internal: run as a single GPU worker")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.strategies, args.seeds, args.output_root)
    if args.worker is not None:
        run_worker(cfg, args.worker)
        return

    out_root = Path(cfg.paths.output_root)
    if args.resume_from and Path(args.resume_from).is_dir():
        shutil.copytree(args.resume_from, out_root, dirs_exist_ok=True)
        log(f"Copied previous outputs from {args.resume_from}")
    out_root.mkdir(parents=True, exist_ok=True)
    clear_claims(out_root, "train/*/seed_*/CLAIMED")

    names, split = prepare_train_data(cfg)
    log(f"Split: {len(split['train_idx'])} train / {len(split['val_idx'])} val "
        f"(split seed {cfg.data.split_seed})")
    (out_root / "split.json").write_text(json.dumps(split))
    pd.DataFrame({"image": [names[i] for i in split["val_idx"]]}).to_csv(
        out_root / "val_split.csv", index=False)

    config_path = out_root / "train_config.yaml"
    save_config(cfg, config_path)
    failed = launch_workers(Path(__file__).resolve(), config_path, out_root / "logs", "train_gpu",
                            poll_seconds=15)

    log("\nTraining summary")
    for seed in cfg.experiment.seeds:
        for strategy in cfg.experiment.strategies:
            done = out_root / "train" / strategy / f"seed_{seed}" / "DONE"
            status = done.read_text().replace("\n", "  ") if done.exists() else "NOT DONE"
            log(f"  {strategy:10s} seed {seed}: {status}")
    if failed:
        raise RuntimeError(f"Worker(s) {failed} crashed, see {out_root / 'logs'}")


if __name__ == "__main__":
    main()
