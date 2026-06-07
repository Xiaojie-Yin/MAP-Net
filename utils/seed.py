import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(
    seed: int = 42,
    deterministic: bool = False,
    benchmark: Optional[bool] = None,
) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: random seed.
        deterministic: whether to force deterministic CuDNN behavior.
        benchmark: torch.backends.cudnn.benchmark.
                   If None, set to not deterministic.
    """
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)

    if benchmark is None:
        torch.backends.cudnn.benchmark = not bool(deterministic)
    else:
        torch.backends.cudnn.benchmark = bool(benchmark)


def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker seed function.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_torch_generator(seed: int = 42) -> torch.Generator:
    """
    Build a deterministic torch.Generator for DataLoader.
    """
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator