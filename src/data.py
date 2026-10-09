import json
from pathlib import Path

import albumentations as A
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_SUFFIXES = ["", "_Segmentation", "_segmentation", "_mask", "_Mask", "_lesion"]


def pair_images_masks(img_dir, mask_dir):
    img_dir, mask_dir = Path(img_dir), Path(mask_dir)
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    pairs, missing = [], []
    for image_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
        mask_path = next((masks[image_path.stem + s] for s in MASK_SUFFIXES
                          if image_path.stem + s in masks), None)
        if mask_path is None:
            missing.append(image_path.name)
        else:
            pairs.append((image_path.stem, image_path, mask_path))
    if missing:
        raise FileNotFoundError(f"{len(missing)} images without a mask, e.g. {missing[:5]}")
    if not pairs:
        raise FileNotFoundError(f"No images found in {img_dir}")
    return pairs


def check_count(name, found, expected):
    if expected is not None and found != expected:
        raise ValueError(f"{name}: expected {expected} image/mask pairs, found {found}")


def build_cache(name, pairs, image_size, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = {"names": [n for n, _, _ in pairs], "image_size": image_size}
    meta_path = cache_dir / f"{name}_meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) == meta:
        return
    resize = A.Resize(image_size, image_size)
    shape = (len(pairs), image_size, image_size)
    imgs = np.lib.format.open_memmap(cache_dir / f"{name}_imgs.npy", mode="w+",
                                     dtype=np.uint8, shape=shape + (3,))
    masks = np.lib.format.open_memmap(cache_dir / f"{name}_masks.npy", mode="w+",
                                      dtype=np.float32, shape=shape)
    for k, (_, image_path, mask_path) in enumerate(pairs):
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = (np.array(Image.open(mask_path).convert("L")) > 127).astype(np.float32)
        out = resize(image=image, mask=mask)
        imgs[k], masks[k] = out["image"], out["mask"]
    imgs.flush()
    masks.flush()
    meta_path.write_text(json.dumps(meta))


def load_cache(name, cache_dir):
    cache_dir = Path(cache_dir)
    imgs = np.load(cache_dir / f"{name}_imgs.npy", mmap_mode="r")
    masks = np.load(cache_dir / f"{name}_masks.npy", mmap_mode="r")
    names = json.loads((cache_dir / f"{name}_meta.json").read_text())["names"]
    return imgs, masks, names


def make_split(n_items, val_fraction, split_seed):
    permutation = np.random.RandomState(split_seed).permutation(n_items)
    n_val = int(round(val_fraction * n_items))
    return {"train_idx": sorted(permutation[n_val:].tolist()),
            "val_idx": sorted(permutation[:n_val].tolist())}


def prepare_train_data(cfg):
    pairs = pair_images_masks(cfg.paths.isic_train_images, cfg.paths.isic_train_masks)
    check_count("isic_train", len(pairs), cfg.experiment.expected_n.get("isic_train"))
    build_cache("isic_train", pairs, cfg.data.image_size, cfg.paths.cache_dir)
    names = [n for n, _, _ in pairs]
    split = make_split(len(names), cfg.data.val_fraction, cfg.data.split_seed)
    return names, split


def prepare_eval_data(cfg):
    sources = {
        "isic": (cfg.paths.isic_test_images, cfg.paths.isic_test_masks),
        "ph2": (cfg.paths.ph2_images, cfg.paths.ph2_masks),
    }
    for key, (img_dir, mask_dir) in sources.items():
        pairs = pair_images_masks(img_dir, mask_dir)
        check_count(key, len(pairs), cfg.experiment.expected_n.get(key))
        build_cache(key, pairs, cfg.data.image_size, cfg.paths.cache_dir)


class CachedDataset(Dataset):
    def __init__(self, imgs, masks, names, transform):
        self.imgs, self.masks, self.ids, self.transform = imgs, masks, names, transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        out = self.transform(image=np.array(self.imgs[idx]), mask=np.array(self.masks[idx]))
        return out["image"].float(), out["mask"].unsqueeze(0).float(), self.ids[idx]
