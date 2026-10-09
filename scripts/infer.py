import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import apply_overrides, load_config, save_config  # noqa: E402
from src.data import prepare_eval_data  # noqa: E402
from src.inference import DATASET_KEYS, run_worker  # noqa: E402
from src.utils import clear_claims, launch_workers, log  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="MC Dropout inference on ISIC-2016 test and PH2.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--strategies", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--worker", type=int, help="internal: run as a single GPU worker")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.strategies, args.seeds, args.output_root)
    if args.worker is not None:
        run_worker(cfg, args.worker)
        return

    out_root = Path(cfg.paths.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    clear_claims(out_root, "infer/*/seed_*/CLAIMED")
    prepare_eval_data(cfg)

    protocol = {
        "threshold": 0.5,
        "foreground": "reference union prediction",
        "background": "complement of foreground",
        "boundary": f"within {cfg.inference.boundary_px} px of the reference contour",
        "checkpoint": "best_weights.pth (max validation Dice)",
        "torch": torch.__version__,
    }
    (out_root / "infer_protocol.json").write_text(json.dumps(protocol, indent=2))

    config_path = out_root / "infer_config.yaml"
    save_config(cfg, config_path)
    failed = launch_workers(Path(__file__).resolve(), config_path, out_root / "logs", "infer_gpu",
                            poll_seconds=10)

    log("\nInference summary")
    missing = []
    for seed in cfg.experiment.seeds:
        for strategy in cfg.experiment.strategies:
            run_dir = out_root / "infer" / strategy / f"seed_{seed}"
            done = all((run_dir / f"{ds}_summary.csv").exists() for ds in DATASET_KEYS)
            log(f"  {strategy:10s} seed {seed}: {'done' if done else 'NOT DONE'}")
            if not done:
                missing.append(f"{strategy}/seed_{seed}")
    if missing or failed:
        raise RuntimeError(f"Incomplete: {missing}, see {out_root / 'logs'}")


if __name__ == "__main__":
    main()
