"""Batching utilities for the Wikitext-103 language-model experiments.

The tokenised corpus is stored as a flat ``uint16`` memmap (see
``scripts/prepare_wikitext.py``). ``get_batch`` samples random contiguous windows
of ``seq_len`` tokens and returns the usual next-token-prediction (x, y) pair.
"""

import numpy as np
import torch


def get_batch(data, batch_size: int, seq_len: int):
    """Sample a next-token-prediction batch from a token memmap.

    Args:
        data:       1-D array/memmap of token ids.
        batch_size: number of sequences in the batch.
        seq_len:    tokens per sequence.

    Returns:
        (x, y) int64 tensors of shape (batch_size, seq_len); ``y`` is ``x``
        shifted right by one token.
    """
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + seq_len].astype(np.int64)) for i in ix])
    return x, y
