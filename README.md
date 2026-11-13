# Augmentation, Uncertainty and Boundary Calibration in Dermoscopic Lesion Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code of the paper **"Beyond Dice: How Training-Time augmentation reshapes uncertainty and boundary calibration of a
U-Net for dermoscopic lesion segmentation"**, Biomedical Signal Processing and Control (2026).
Paper: [https://doi.org/10.1016/j.bspc.2026.111818](https://doi.org/10.1016/j.bspc.2026.111818)

## Overview

High overlap with a reference mask does not say how reliable the confidence of a segmentation network is. This project
holds the architecture, optimisation and inference protocol fixed, varies only the training-time augmentation
pipeline, and measures how it reshapes the uncertainty and calibration of a U-Net, over all pixels and within a band
around the lesion contour, on in-distribution and external images.

| | |
|---|---|
| **Task** | Dermoscopic lesion segmentation, ISIC 2016 (train and test) |
| **External site** | PH2 |
| **Model** | U-Net with spatial dropout (p = 0.3) in every block, MC Dropout inference (T = 20) |
| **Pipelines** | baseline, geometric, intensity, mixing (MixUp or CutMix), occlusion |
| **Scale** | 15 networks (5 pipelines x 3 seeds), evaluated on 379 ISIC 2016 test images and 200 PH2 images |
| **Metrics** | Dice, IoU, HD95, ECE and Brier score (all pixels and 5 px boundary band), predictive variance and entropy, risk-coverage (AURC), pixel-level error detection (AUROC, AUPRC) |
| **Analysis** | Hierarchical bootstrap over seeds and images, paired differences with Holm correction, Friedman and Wilcoxon tests, sensitivity to the number of MC passes |

| Pipeline | Transforms |
|---|---|
| `baseline` | none |
| `geometric` | flips, rotation, elastic deformation |
| `intensity` | brightness/contrast, Gaussian noise, gamma |
| `mixing` | horizontal flip, MixUp or CutMix per batch |
| `occlusion` | horizontal flip, GridDropout, CoarseDropout |

## Contents

1. [Pipeline](#1-pipeline)
2. [Repository layout](#2-repository-layout)
3. [Installation](#3-installation)
4. [Data](#4-data)
5. [Run](#5-run)
6. [Configuration](#6-configuration)
7. [Outputs](#7-outputs)
8. [Reproducibility](#8-reproducibility)
9. [Released outputs](#9-released-outputs)
10. [License](#10-license)
11. [Citation](#11-citation)

## 1. Pipeline

```
train -> infer -> summarize
  |        |         |
  |        |         +- tables, bootstrap confidence intervals, paired differences, statistical tests
  |        +- MC Dropout on the ISIC 2016 test set and on PH2, per-image metrics, calibration bins, T sweep
  +- 5 pipelines x 3 seeds, best checkpoint by validation Dice
```

Every stage is **safe to rerun**: finished work is skipped, and interrupted training resumes where it stopped. One
process is started per GPU.

## 2. Repository layout

```
configs/default.yaml     paths and settings of the paper run
scripts/
  train.py               train all pipelines and seeds
  infer.py               MC Dropout inference on ISIC 2016 test and PH2
  summarize.py           tables, confidence intervals, statistical tests
src/                     model, augmentations, losses, metrics, training, inference, analysis
tests/                   unit tests and a small end-to-end test
requirements.txt         dependencies
requirements-dev.txt     dependencies plus pytest
LICENSE                  MIT license of the code
```

## 3. Installation

```bash
git clone https://github.com/sarang-banakhede/Augmentation-Effect-On-Uncertainty-Calibration.git
cd Augmentation-Effect-On-Uncertainty-Calibration
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                          # unit tests, no data and no GPU needed
```

Versions of the paper runs: Python 3.12.13, PyTorch 2.10.0 (CUDA 12.8), Albumentations 2.0.8, NumPy 2.0.2.

## 4. Data

Both datasets are public. Download each set from its provider and arrange it as below, or change the paths in
`configs/default.yaml`. The scripts check that 900 training, 379 test and 200 PH2 image and mask pairs are found.

```
data/
├── ISIC2016/
│   ├── train/{images,masks}/
│   └── test/{images,masks}/
└── PH2Dataset/
    ├── image/                 .bmp
    └── label/                 .bmp, named {id}_lesion.bmp
```

| Set | Role | Images | Split rule |
|---|---|---|---|
| ISIC 2016 train | training and validation | 900 | 10% held out for checkpoint selection (split seed 42): 810 train, 90 validation |
| ISIC 2016 test | in-distribution evaluation | 379 | official test set |
| PH2 | external evaluation | 200 | not used for training or any design decision |

- ISIC 2016: <https://challenge.isic-archive.com/data/>
- PH2: <https://www.fc.up.pt/addi/ph2%20database.html>

All images are resized to 256 x 256 and scaled to [0, 1]. Masks are binarised at 127.

## 5. Run

```bash
python scripts/train.py
python scripts/infer.py
python scripts/summarize.py
```

Options:

```bash
python scripts/train.py --strategies baseline mixing --seeds 0 1
python scripts/train.py --output-root /path/to/outputs
python scripts/train.py --resume-from /path/to/previous/outputs
```

Training takes about 38 minutes per model on a Kaggle T4.

## 6. Configuration

`configs/default.yaml` holds every path and setting of the paper run.

| Section | Main settings |
|---|---|
| `experiment` | pipelines, seeds (0, 1, 2), datasets, expected image counts |
| `data` | image size 256, validation fraction 0.10, split seed 42 |
| `model` | spatial dropout 0.3 |
| `train` | 50 epochs, batch size 16, Adam with learning rate 1e-4 |
| `inference` | 20 MC passes, T sweep (10, 20, 50, 100), MC seed 1234, 10 ECE bins, 5 px boundary band |
| `statistics` | 2000 bootstrap draws, reference pipeline `baseline`, alpha 0.05 |

## 7. Outputs

```
outputs/
├── split.json, val_split.csv     fixed train/validation split
├── train/<strategy>/seed_<s>/    weights, training_log.csv, run_config.json
├── infer/<strategy>/seed_<s>/    per-image metrics, calibration bins, T sweep
└── summary/                      table_II_segmentation.csv, table_III_calibration.csv,
                                  table_IV_usefulness.csv, paired_diffs.csv,
                                  bootstrap_ci.csv, friedman_wilcoxon.csv,
                                  per_run.csv, T_sensitivity.csv, training_summary.csv
```

- `paired_diffs.csv` lists each pair of pipelines as `a` and `b`, with `a_minus_b` giving the value of `a` minus the
  value of `b`. Pipelines are ordered baseline, geometric, intensity, mixing, occlusion.
- Calibration is reported for four pixel regions: `all`, `fg` (union of reference and predicted lesion), `bg` (its
  complement) and `boundary` (the 5 px band around the reference contour).
- HD95 is the 95th percentile of the pooled contour distances in both directions, in pixels at 256 x 256.
- Pixel-level AUROC and AUPRC are computed from 1024-bin histograms of the uncertainty scores.
- MC Dropout masks for batch `k` use seed `1234 + k`, so all models are evaluated under identical sampling
  conditions.
- The summary tables give the mean with a 95% bootstrap confidence interval where available, otherwise the mean with
  the seed standard deviation. A dagger marks a difference from the baseline with Holm-adjusted p below 0.05.

## 8. Reproducibility

Training uses fixed seeds (0, 1, 2), a fixed validation split and deterministic cuDNN; the seed controls weight
initialisation, mini-batch order, augmentation sampling and dropout masks. Every run writes its settings and library
versions to `run_config.json`. The paper runs used two NVIDIA Tesla T4 GPUs on Kaggle. The 379 test images and the
200 PH2 images are used only for final evaluation, never for checkpoint selection or any other design decision.

## 9. Released outputs

The complete outputs of the paper run (checkpoints, per-image results, calibration bins, statistics, training logs and
the validation split) are released on Zenodo under the CC BY-NC 4.0 licence. The link will be available on
publication.

The released folder has the structure described in [section 7](#7-outputs). Place it in the repository root, where
`paths.output_root` expects it, to rerun `summarize.py` or inference from the released checkpoints.

## 10. License

* **Code:** released under the [MIT License](LICENSE).
* **Released outputs** (checkpoints, per-image results, statistics and logs): released under
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
* **Datasets:** the original datasets remain subject to the licences and terms of their providers.

## 11. Citation

**Paper.** If you use this code, please cite:

> Banakhede, S., Tiwari, A., Tiwari, V., Ghayvat, H. (2026). Beyond Dice: How Training-Time augmentation reshapes
> uncertainty and boundary calibration of a U-Net for dermoscopic lesion segmentation. *Biomedical Signal Processing
> and Control*. https://doi.org/10.1016/j.bspc.2026.111818

```bibtex
@article{banakhede2026beyonddice,
  author  = {Banakhede, Sarang and Tiwari, Akshra and Tiwari, Vibha and Ghayvat, Hemant},
  title   = {{Beyond Dice}: How {Training-Time} augmentation reshapes uncertainty and boundary calibration of a {U-Net} for dermoscopic lesion segmentation},
  journal = {Biomedical Signal Processing and Control},
  year    = {2026},
  doi     = {10.1016/j.bspc.2026.111818},
  url     = {https://doi.org/10.1016/j.bspc.2026.111818}
}
```

**Released outputs.** If you use the released outputs, please cite the corresponding Zenodo record. The citation will
be available on publication.

MixUp and CutMix follow Zhang et al. (ICLR 2018) and Yun et al. (ICCV 2019).
