import inspect

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

STRATEGIES = ("baseline", "geometric", "intensity", "mixing", "occlusion")


def _has(cls, arg):
    return arg in inspect.signature(cls.__init__).parameters


def gauss_noise(p):
    if _has(A.GaussNoise, "std_range"):
        return A.GaussNoise(std_range=(np.sqrt(10) / 255, np.sqrt(50) / 255),
                            mean_range=(0.0, 0.0), p=p)
    return A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=p)


def coarse_dropout(p):
    if _has(A.CoarseDropout, "num_holes_range"):
        return A.CoarseDropout(num_holes_range=(8, 8), hole_height_range=(32, 32),
                               hole_width_range=(32, 32), p=p)
    return A.CoarseDropout(max_holes=8, max_height=32, max_width=32,
                           min_holes=8, min_height=32, min_width=32, fill_value=0, p=p)


def elastic(p):
    if _has(A.ElasticTransform, "alpha_affine"):
        return A.ElasticTransform(alpha=120, sigma=6, alpha_affine=0, p=p)
    return A.ElasticTransform(alpha=120, sigma=6, p=p)


def _compose(transforms):
    return A.Compose([*transforms, A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
                      ToTensorV2()])


def get_train_transform(strategy):
    if strategy == "baseline":
        return _compose([])
    if strategy == "geometric":
        return _compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3),
                         A.Rotate(limit=30, p=0.5), elastic(p=0.3)])
    if strategy == "intensity":
        return _compose([A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                         gauss_noise(p=0.3), A.RandomGamma(gamma_limit=(80, 120), p=0.3)])
    if strategy == "mixing":
        return _compose([A.HorizontalFlip(p=0.5)])
    if strategy == "occlusion":
        return _compose([A.HorizontalFlip(p=0.5), A.GridDropout(ratio=0.3, p=0.4),
                         coarse_dropout(p=0.4)])
    raise ValueError(f"Unknown strategy: {strategy}")


def get_eval_transform():
    return _compose([])


def mixup_batch(imgs, masks, alpha=0.4):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    return lam * imgs + (1 - lam) * imgs[idx], lam * masks + (1 - lam) * masks[idx]


def cutmix_batch(imgs, masks, alpha=1.0):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    _, _, height, width = imgs.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(height * cut_ratio), int(width * cut_ratio)
    cy, cx = np.random.randint(height), np.random.randint(width)
    y1, y2 = np.clip(cy - cut_h // 2, 0, height), np.clip(cy + cut_h // 2, 0, height)
    x1, x2 = np.clip(cx - cut_w // 2, 0, width), np.clip(cx + cut_w // 2, 0, width)
    imgs, masks = imgs.clone(), masks.clone()
    imgs[:, :, y1:y2, x1:x2] = imgs[idx][:, :, y1:y2, x1:x2]
    masks[:, :, y1:y2, x1:x2] = masks[idx][:, :, y1:y2, x1:x2]
    return imgs, masks


def apply_mixing(imgs, masks):
    if np.random.rand() < 0.5:
        return mixup_batch(imgs, masks)
    return cutmix_batch(imgs, masks)
