"""Optional seed initialization and an allowlisted execution-environment report.

Call these helpers explicitly from experiment code. They are not automatically
wired into the notebook and do not change its training, checkpoints or thresholds.
"""

import importlib.metadata
import os
import platform
import random

import numpy as np
import torch


def seed_everything(seed=42, *, deterministic=False):
    """Initialize common RNGs and return a seeded PyTorch loader generator.

    Call before model initialization or data sampling. Pass the returned
    generator to ``DataLoader(generator=...)``; independent NumPy ``default_rng``
    objects still need their own seeds. ``deterministic`` sets PyTorch's strict
    deterministic flag and the cuDNN flags used below.

    CUDA strict mode requires CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) before
    process/CUDA initialization and may reject unsupported operations. Initial
    seeding is not checkpoint RNG-state restoration and does not guarantee
    identical interrupted runs, software versions or hardware results.
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
    """Seed a DataLoader worker's Python/NumPy RNGs from its PyTorch seed.

    Pass this function as ``worker_init_fn``. PyTorch assigns the worker seed;
    ``worker_id`` is accepted for the DataLoader callback interface.
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def environment_report():
    """Return an allowlisted, JSON-serializable environment inventory.

    Includes package/Python versions, CUDA availability and determinism settings.
    An absent package is recorded as None. Only the two named execution variables
    below are read; credentials and the full process environment are not dumped.
    An inventory describes execution context, not experimental reproducibility.
    """
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
