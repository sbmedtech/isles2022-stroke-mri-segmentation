"""Opt-in random seeding and a non-secret environment inventory.

These helpers do not alter notebook training, checkpoints, or thresholds.
"""

import importlib.metadata
import os
import platform
import random

import numpy as np
import torch


def seed_everything(seed=42, *, deterministic=False):
    """Seed Python, NumPy, and PyTorch; return a seeded DataLoader generator.

    Call before creating the model or sampling data. Strict mode rejects
    nondeterministic PyTorch operations. CUDA strict mode requires launching
    with CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) before CUDA initialization.
    This is not RNG-state restoration and does not guarantee resumed-run parity.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not isinstance(deterministic, bool):
        raise ValueError("deterministic must be a bool")
    if deterministic and torch.cuda.is_available():
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG before starting the CUDA process")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    return torch.Generator().manual_seed(seed)


def seed_worker(worker_id):
    """DataLoader worker_init_fn: derive Python/NumPy seeds from PyTorch."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def environment_report():
    """Return JSON-serializable versions and execution settings, never env dumps."""
    packages = {}
    for name in ("numpy", "torch", "monai", "nibabel", "scipy", "matplotlib",
                 "tqdm", "requests", "pandas", "setuptools"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": packages, "cuda_available": cuda,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else [],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(environment_report(), indent=2))
