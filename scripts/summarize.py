import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import print_report, run_analysis  # noqa: E402
from src.config import apply_overrides, load_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Tables and statistical tests for the manuscript.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--strategies", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output-root")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.strategies, args.seeds, args.output_root)
    report = run_analysis(cfg)
    print_report(report, cfg.statistics.reference, cfg.statistics.alpha)


if __name__ == "__main__":
    main()
