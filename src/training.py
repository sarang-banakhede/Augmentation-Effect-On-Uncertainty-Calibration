import gc
import json
import platform
import random
import time
import traceback
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from .augmentations import apply_mixing, get_eval_transform, get_train_transform
from .data import CachedDataset, load_cache
from .losses import DiceBCELoss
from .metrics import per_image_dice_iou
from .model import UNet
from .utils import claim, get_device, log, set_seed


def run_dir_for(cfg, strategy, seed):
    return Path(cfg.paths.output_root) / "train" / strategy / f"seed_{seed}"


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    dataset = torch.utils.data.get_worker_info().dataset
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    if hasattr(dataset.transform, "set_random_seed"):
        dataset.transform.set_random_seed(worker_seed)


def write_run_config(cfg, run_dir, strategy, seed, n_train, n_val):
    info = {
        "strategy": strategy, "seed": seed, "epochs": cfg.train.epochs, "lr": cfg.train.lr,
        "batch_size": cfg.train.batch_size, "image_size": cfg.data.image_size,
        "dropout": cfg.model.dropout, "optimizer": "Adam", "scheduler": None,
        "early_stopping": None, "selection": "max val Dice (threshold 0.5)",
        "val_frac": cfg.data.val_fraction, "split_seed": cfg.data.split_seed,
        "n_train": n_train, "n_val": n_val,
        "train_transform": f"Resize({cfg.data.image_size}) -> {get_train_transform(strategy)}",
        "torch": torch.__version__, "albumentations": A.__version__, "numpy": np.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (run_dir / "run_config.json").write_text(json.dumps(info, indent=2))


def restore_checkpoint(path, model, optimizer, loader_generator, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["py_rng"])
    np.random.set_state(state["np_rng"])
    torch.set_rng_state(state["torch_rng"])
    loader_generator.set_state(state["loader_rng"])
    if torch.cuda.is_available() and state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"], device)
    return state["epoch"] + 1, state["best_val_dice"], state["best_epoch"]


def save_checkpoint(path, epoch, model, optimizer, best_val_dice, best_epoch, loader_generator, device):
    torch.save({
        "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "best_val_dice": best_val_dice, "best_epoch": best_epoch,
        "py_rng": random.getstate(), "np_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(), "loader_rng": loader_generator.get_state(),
        "cuda_rng": torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None,
    }, path)


def evaluate(model, loader, criterion, device):
    model.eval()
    losses, dices, ious = [], [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losses.append(criterion(logits, y).item() * x.size(0))
            dice, iou = per_image_dice_iou(logits, y)
            dices.extend(dice)
            ious.extend(iou)
    return sum(losses) / len(dices), float(np.mean(dices)), float(np.mean(ious))


def train_one_run(cfg, strategy, seed, imgs, masks, names, split, device):
    run_dir = run_dir_for(cfg, strategy, seed)
    checkpoint_path, best_path = run_dir / "last_checkpoint.pth", run_dir / "best_weights.pth"
    last_path, log_path = run_dir / "last_weights.pth", run_dir / "training_log.csv"
    epochs, num_workers = cfg.train.epochs, cfg.data.num_workers

    set_seed(seed)
    train_set = Subset(CachedDataset(imgs, masks, names, get_train_transform(strategy)),
                       split["train_idx"])
    val_set = Subset(CachedDataset(imgs, masks, names, get_eval_transform()), split["val_idx"])
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              worker_init_fn=seed_worker, generator=generator)
    val_loader = DataLoader(val_set, batch_size=cfg.train.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker)

    model = UNet(dropout_p=cfg.model.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    criterion = DiceBCELoss()

    start_epoch, best_val_dice, best_epoch, rows = 0, -1.0, -1, []
    if checkpoint_path.exists():
        start_epoch, best_val_dice, best_epoch = restore_checkpoint(
            checkpoint_path, model, optimizer, generator, device)
        rows = pd.read_csv(log_path).to_dict("records") if log_path.exists() else []
        log(f"[RESUME] {strategy}/seed_{seed} from epoch {start_epoch + 1}")

    write_run_config(cfg, run_dir, strategy, seed, len(split["train_idx"]), len(split["val_idx"]))

    for epoch in range(start_epoch, epochs):
        started = time.time()
        model.train()
        train_losses, train_dices = [], []
        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if strategy == "mixing":
                x, y = apply_mixing(x, y)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_dices.extend(per_image_dice_iou(logits.detach(), y)[0])

        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion, device)
        if val_dice > best_val_dice:
            best_val_dice, best_epoch = val_dice, epoch + 1
            torch.save(model.state_dict(), best_path)
        torch.save(model.state_dict(), last_path)

        rows.append({"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)),
                     "train_dice": float(np.mean(train_dices)), "val_loss": val_loss,
                     "val_dice": val_dice, "val_iou": val_iou, "best_epoch": best_epoch,
                     "sec": round(time.time() - started, 1)})
        pd.DataFrame(rows).to_csv(log_path, index=False)
        save_checkpoint(checkpoint_path, epoch, model, optimizer, best_val_dice, best_epoch,
                        generator, device)
        log(f"[{strategy}/seed{seed}] ep {epoch + 1:02d}/{epochs} loss {np.mean(train_losses):.4f} "
            f"train_dice {np.mean(train_dices):.4f} val_dice {val_dice:.4f} "
            f"(best {best_val_dice:.4f} @ {best_epoch}) {time.time() - started:.0f}s")

    checkpoint_path.unlink(missing_ok=True)
    (run_dir / "DONE").write_text(f"best_epoch={best_epoch}\nbest_val_dice={best_val_dice:.6f}\n")
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()


def run_worker(cfg, worker_id):
    if cfg.data.num_workers < 1:
        raise ValueError("data.num_workers must be >= 1 so that augmentation seeding is deterministic")
    device = get_device()
    imgs, masks, names = load_cache("isic_train", cfg.paths.cache_dir)
    split = json.loads((Path(cfg.paths.output_root) / "split.json").read_text())
    log(f"worker {worker_id} on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    for seed in cfg.experiment.seeds:
        for strategy in cfg.experiment.strategies:
            run_dir = run_dir_for(cfg, strategy, seed)
            if (run_dir / "DONE").exists() or not claim(run_dir):
                continue
            log(f"[START] {strategy}/seed_{seed}")
            try:
                train_one_run(cfg, strategy, seed, imgs, masks, names, split, device)
            except Exception:
                log(f"[FAILED] {strategy}/seed_{seed}\n{traceback.format_exc()}")
                torch.cuda.empty_cache()
                gc.collect()
    log(f"worker {worker_id} finished")
