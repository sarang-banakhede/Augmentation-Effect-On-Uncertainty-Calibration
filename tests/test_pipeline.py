import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")

from src.config import load_config, save_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SIZE = 64


def write_pairs(img_dir, mask_dir, prefix, count, extension, mask_suffix, rng):
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    for i in range(count):
        cy, cx = rng.randint(22, 42, 2)
        radius = rng.randint(8, 16)
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 < radius ** 2
        image = rng.randint(90, 170, (SIZE, SIZE, 3)).astype(np.uint8)
        image[mask] = (image[mask] * 0.5).astype(np.uint8)
        Image.fromarray(image).save(img_dir / f"{prefix}{i:03d}.{extension}")
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            mask_dir / f"{prefix}{i:03d}{mask_suffix}.{extension}")


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    rng = np.random.RandomState(0)
    write_pairs(root / "isic/train/images", root / "isic/train/masks", "ISIC_", 20, "png", "", rng)
    write_pairs(root / "isic/test/images", root / "isic/test/masks", "ISIC_T", 8, "png", "", rng)
    write_pairs(root / "ph2/image", root / "ph2/label", "IMD", 6, "bmp", "_lesion", rng)

    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg["paths"] = {
        "isic_train_images": str(root / "isic/train/images"),
        "isic_train_masks": str(root / "isic/train/masks"),
        "isic_test_images": str(root / "isic/test/images"),
        "isic_test_masks": str(root / "isic/test/masks"),
        "ph2_images": str(root / "ph2/image"),
        "ph2_masks": str(root / "ph2/label"),
        "output_root": str(root / "outputs"),
        "cache_dir": str(root / "cache"),
    }
    cfg["experiment"].update({"strategies": ["baseline", "mixing"], "seeds": [0, 1],
                              "expected_n": {"isic_train": 20, "isic": 8, "ph2": 6}})
    cfg["data"].update({"image_size": 32, "num_workers": 1})
    cfg["train"].update({"epochs": 2, "batch_size": 8})
    cfg["inference"].update({"mc_passes": 3, "t_sweep": [2, 3], "t_sweep_seeds": [0],
                             "batch_size": 4, "err_hist_bins": 64})
    cfg["statistics"].update({"n_boot": 20})
    config_path = root / "config.yaml"
    save_config(cfg, config_path)
    return root, config_path


def run_script(name, config_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / name), "--config", str(config_path)],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]


def test_train_infer_summarize(workspace):
    root, config_path = workspace
    out = root / "outputs"

    run_script("train.py", config_path)
    for strategy in ("baseline", "mixing"):
        for seed in (0, 1):
            run = out / "train" / strategy / f"seed_{seed}"
            assert (run / "DONE").exists() and (run / "best_weights.pth").exists()
    assert len((out / "val_split.csv").read_text().splitlines()) == 3

    run_script("infer.py", config_path)
    for strategy in ("baseline", "mixing"):
        for seed in (0, 1):
            run = out / "infer" / strategy / f"seed_{seed}"
            for dataset in ("isic", "ph2"):
                assert (run / f"{dataset}_per_image.csv").exists()
                assert (run / f"{dataset}_bins.npz").exists()
    assert (out / "infer/baseline/seed_0/isic_T_sweep.npz").exists()
    assert not (out / "infer/baseline/seed_1/isic_T_sweep.npz").exists()

    run_script("summarize.py", config_path)
    summary = out / "summary"
    for name in ("table_II_segmentation", "table_III_calibration", "table_IV_usefulness",
                 "paired_diffs", "bootstrap_ci", "per_run", "T_sensitivity", "training_summary"):
        assert (summary / f"{name}.csv").exists(), name
