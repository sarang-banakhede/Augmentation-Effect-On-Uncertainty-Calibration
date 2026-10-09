# Augmentation, Uncertainty and Boundary Calibration in Dermoscopic Lesion Segmentation

Code for the manuscript **"Beyond Dice: How Training-Time Augmentation Reshapes Uncertainty and Boundary Calibration of a U-Net for Dermoscopic Lesion Segmentation"** (under review).

Sarang Banakhede, Akshra Tiwari, Vibha Tiwari, Hemant Ghayvat

## What it does

The same U-Net is trained on ISIC-2016 with five augmentation pipelines and three seeds each (15 models). Uncertainty is estimated with MC Dropout. Every model is tested on the ISIC-2016 test set and on PH2 (external). The code measures:

- segmentation: Dice, IoU, HD95
- calibration: ECE and Brier score over all pixels and within a 5 px band around the lesion contour
- usefulness of uncertainty: risk-coverage, error detection, correlation with error
- statistics: hierarchical bootstrap over seeds and images, paired differences with Holm correction

| Pipeline | Transforms |
| --- | --- |
| `baseline` | none |
| `geometric` | flips, rotation, elastic deformation |
| `intensity` | brightness/contrast, Gaussian noise, gamma |
| `mixing` | horizontal flip, MixUp or CutMix per batch |
| `occlusion` | horizontal flip, GridDropout, CoarseDropout |

Training setup: Adam (lr 1e-4), batch size 16, 50 epochs, BCE + soft Dice, 256 x 256 input, spatial dropout p = 0.3. The checkpoint with the best Dice on a fixed 90-image validation split is kept. MC Dropout uses T = 20 passes.

## Structure

```
configs/default.yaml     paths and hyperparameters
scripts/
  train.py               train all pipelines and seeds
  infer.py               MC Dropout inference on ISIC-2016 test and PH2
  summarize.py           tables, confidence intervals, statistical tests
src/                     model, augmentations, metrics, training, inference, analysis
tests/
requirements.txt
```

## Setup

```bash
git clone https://github.com/sarang-banakhede/Augmentation-Uncertainty-Calibration-Study.git
cd Augmentation-Uncertainty-Calibration-Study
pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch 2.10, Albumentations 2.0.8.

Download the datasets and arrange them like this (or change the paths in `configs/default.yaml`):

```
data/
├── ISIC2016/
│   ├── train/{images,masks}/
│   └── test/{images,masks}/
└── PH2Dataset/
    ├── image/                 .bmp
    └── label/                 .bmp, named {id}_lesion.bmp
```

- ISIC-2016: https://challenge.isic-archive.com/data/
- PH2: https://www.fc.up.pt/addi/ph2%20database.html

## Run

```bash
python scripts/train.py
python scripts/infer.py
python scripts/summarize.py
```

Each step skips finished runs, so an interrupted run can be restarted. One process is started per GPU. Options:

```bash
python scripts/train.py --strategies baseline mixing --seeds 0 1
python scripts/train.py --output-root /path/to/outputs
python scripts/train.py --resume-from /path/to/previous/outputs
```

## Outputs

```
outputs/
├── train/<strategy>/seed_<s>/    weights, training_log.csv
├── infer/<strategy>/seed_<s>/    per-image metrics, calibration bins, T sweep
└── summary/                      table_II_segmentation.csv, table_III_calibration.csv,
                                  table_IV_usefulness.csv, paired_diffs.csv, ...
```
