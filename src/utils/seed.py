"""Seeding utility shared by every run.

All arms call this with the same seed before building the model, so AdamW, Lion
and Lion-MANE start from identical initial weights — essential for attributing
any gap to the optimizer alone.
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
