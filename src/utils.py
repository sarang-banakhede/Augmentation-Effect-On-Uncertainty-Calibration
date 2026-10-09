import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

LOG_FILTERS = ("UserWarning", "super().__init__")


def log(message):
    print(message, flush=True)


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def claim(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(run_dir / "CLAIMED", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def clear_claims(root, pattern):
    for path in Path(root).glob(pattern):
        path.unlink()


def launch_workers(script, config_path, log_dir, log_prefix, poll_seconds=10):
    n_gpu = torch.cuda.device_count()
    log_dir.mkdir(parents=True, exist_ok=True)
    workers = []
    for k in range(max(1, n_gpu)):
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        if n_gpu:
            env["CUDA_VISIBLE_DEVICES"] = str(k)
        log_file = open(log_dir / f"{log_prefix}{k}.log", "w")
        process = subprocess.Popen(
            [sys.executable, str(script), "--config", str(config_path), "--worker", str(k)],
            env=env, stdout=log_file, stderr=subprocess.STDOUT,
        )
        workers.append((k, process, log_file))
        time.sleep(2)
    log(f"Started {len(workers)} worker(s) on {n_gpu} GPU(s)")

    offsets = {k: 0 for k, _, _ in workers}
    while True:
        running = any(process.poll() is None for _, process, _ in workers)
        for k, _, _ in workers:
            with open(log_dir / f"{log_prefix}{k}.log") as reader:
                reader.seek(offsets[k])
                text = reader.read()
                offsets[k] = reader.tell()
            for line in text.splitlines():
                if line.strip() and not any(f in line for f in LOG_FILTERS):
                    print(f"[GPU{k}] {line}", flush=True)
        if not running:
            break
        time.sleep(poll_seconds)
    for _, _, log_file in workers:
        log_file.close()
    return [k for k, process, _ in workers if process.returncode != 0]
