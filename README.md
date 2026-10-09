# Augmentation, Uncertainty and Boundary Calibration in Dermoscopic Lesion Segmentation

Official code for the manuscript **"Beyond Dice: How Training-Time Augmentation Reshapes Uncertainty and Boundary Calibration of a U-Net for Dermoscopic Lesion Segmentation"**.

**Authors:** Sarang Banakhede, Akshra Tiwari, Vibha Tiwari, Hemant Ghayvat

> **Status:** the manuscript is currently under review. This repository documents the experimental pipeline. Results, figures and discussion are in the manuscript and will be linked here after publication.

## Overview

Data augmentation is standard practice in medical image segmentation, but its effect on the reliability of a model's confidence is rarely examined. This study holds the architecture, optimizer and inference protocol fixed and varies only the training-time augmentation pipeline. It measures how each pipeline changes uncertainty and calibration, both over all pixels and in a band around the lesion contour, on the in-distribution ISIC-2016 test set and on the external PH2 dataset.

| Pipeline | Transforms |
| --- | --- |
| `baseline` | none (resize and scaling only) |
| `geometric` | horizontal and vertical flip, rotation up to 30 degrees, elastic deformation |
| `intensity` | brightness and contrast, Gaussian noise, gamma |
| `mixing` | horizontal flip, then MixUp or CutMix on each batch |
| `occlusion` | horizontal flip, GridDropout, CoarseDropout |

## Protocol

| Component | Setting |
| --- | --- |
| Model | U-Net, encoder widths 64 to 512, 1024-channel bottleneck, spatial dropout (p = 0.3) in every block |
| Data | ISIC-2016 (900 train, 379 test), PH2 (200, external), images resized to 256 x 256 |
| Validation | fixed 10% hold-out of the training set (90 images, split seed 42), identical for every run |
| Training | Adam, learning rate 1e-4, batch size 16, 50 epochs, BCE + soft Dice, no scheduler, no early stopping |
| Checkpoint | highest validation Dice, the test sets are never used for selection |
| Seeds | 0, 1, 2 for each pipeline (15 models) |
| Uncertainty | MC Dropout, T = 20 passes, BatchNorm in eval mode, predictive mean, variance and entropy |
| Calibration | pixel-level ECE (10 bins) and Brier score over all, foreground, background and boundary pixels |
| Boundary band | pixels within 5 px of the reference contour on either side |
| Usefulness of uncertainty | risk-coverage (AURC), pixel and image-level error detection (AUROC, AUPRC) |
| Statistics | hierarchical bootstrap over seeds and images (B = 2000), paired differences, Holm correction, Friedman and Wilcoxon on per-image scores |

## Repository structure

```
.
├── configs/
│   └── default.yaml          all paths and hyperparameters
├── scripts/
│   ├── train.py              train every pipeline and seed
│   ├── infer.py              MC Dropout inference on ISIC-2016 test and PH2
│   └── summarize.py          tables, confidence intervals and statistical tests
├── src/
│   ├── augmentations.py      pipelines, MixUp and CutMix
│   ├── analysis.py           bootstrap, paired tests, paper tables
│   ├── config.py             YAML loading and overrides
│   ├── data.py               pairing, caching, split, dataset
│   ├── inference.py          MC Dropout inference and per-image metrics
│   ├── losses.py             BCE + soft Dice
│   ├── metrics.py            Dice, IoU, HD95, ECE, Brier, entropy, error histograms
│   ├── model.py              U-Net with spatial dropout
│   ├── stat_tools.py         AUROC, AURC, Holm
│   ├── training.py           resumable training loop
│   └── utils.py              seeding, job claiming, multi-GPU launcher
├── tests/
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/sarang-banakhede/Augmentation-Uncertainty-Calibration-Study.git
cd Augmentation-Uncertainty-Calibration-Study
pip install -r requirements.txt
```

Python 3.10 or newer is required. The reported runs used Python 3.12, PyTorch 2.10 (CUDA 12.8), Albumentations 2.0.8 and NumPy 2.0.2.

## Data

Download the datasets from their providers and arrange them as follows.

```
data/
├── ISIC2016/
│   ├── train/{images,masks}/     PNG, mask has the same file name as the image
│   └── test/{images,masks}/
└── PH2Dataset/
    ├── image/                    BMP
    └── label/                    BMP, {id}_lesion.bmp
```

- ISIC-2016 Skin Lesion Analysis Toward Melanoma Detection: https://challenge.isic-archive.com/data/
- PH2: https://www.fc.up.pt/addi/ph2%20database.html

Masks are binarized at 127. Paths can be changed in `configs/default.yaml`. The scripts check the image counts (900, 379, 200) and stop on a mismatch. Set a count to `null` in the config to skip the check.

## Usage

Run the three stages in order. Each stage is resumable and skips finished runs.

```bash
python scripts/train.py
python scripts/infer.py
python scripts/summarize.py
```

One worker process is started per visible GPU, and each run stays on a single GPU, so results do not depend on the number of GPUs. Without a GPU the scripts run on the CPU, which is only practical for small tests.

Useful options (available on all three scripts):

```bash
python scripts/train.py --config configs/default.yaml
python scripts/train.py --strategies baseline mixing --seeds 0 1
python scripts/train.py --output-root /path/to/outputs
python scripts/train.py --resume-from /path/to/previous/outputs
```

## Outputs

```
outputs/
├── split.json, val_split.csv
├── train/<strategy>/seed_<s>/
│   ├── best_weights.pth, last_weights.pth
│   ├── training_log.csv, run_config.json
├── infer/<strategy>/seed_<s>/
│   ├── {isic,ph2}_per_image.csv     Dice, IoU, HD95, variance, entropy, ECE and Brier by region
│   ├── {isic,ph2}_bins.npz          per-image calibration bins and error-detection histograms
│   ├── {isic,ph2}_summary.csv       dataset-level numbers for the run
│   └── {isic,ph2}_T_sweep.npz       results for T = 10, 20, 50, 100 from the same passes
└── summary/
    ├── table_II_segmentation.csv
    ├── table_III_calibration.csv
    ├── table_IV_usefulness.csv
    ├── per_run.csv, bootstrap_ci.csv, paired_diffs.csv
    ├── friedman_wilcoxon.csv, T_sensitivity.csv, training_summary.csv
```

In the paper tables, a dagger marks a difference from the baseline with Holm-adjusted p < 0.05.

## Reproducibility

- All randomness is seeded: weight initialization, batch order, augmentation, MixUp and CutMix draws and dropout masks.
- cuDNN runs in deterministic mode, and `data.num_workers` must be at least 1 so that augmentation seeding is deterministic.
- MC Dropout masks are seeded per batch from `inference.mc_seed` plus the batch index, so the first 20 of 100 passes are identical to the T = 20 run.
- Each run writes `run_config.json` with the settings and library versions.
- Training one model takes about 38 minutes on an NVIDIA Tesla T4.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The tests cover the metric and statistics helpers and run the full train, infer and summarize pipeline on a small synthetic dataset.

## Citation

A citation entry will be added when the manuscript is published.

## License

Released under the MIT License. See `LICENSE`.
