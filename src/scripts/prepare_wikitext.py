"""Download WikiText-103 (raw) and tokenise it into uint16 ``.bin`` memmaps.

Pulls ``Salesforce/wikitext`` / ``wikitext-103-raw-v1`` from the Hugging Face hub
and encodes each split with the GPT-2 BPE tokenizer into a flat token stream:

    data/train.bin   (~117M tokens)
    data/val.bin     (~245k tokens)
    data/test.bin    (~280k tokens)

    python src/scripts/prepare_wikitext.py
"""

import os

import numpy as np
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")

# (HF split name, output basename)
SPLITS = [("validation", "val"), ("test", "test"), ("train", "train")]
ENCODE_BATCH = 20_000  # lines per tokenizer call (Rust-parallel internally)


def encode_split(lines, out_path, tokenizer) -> int:
    """Tokenise a list of raw lines into one uint16 stream on disk."""
    total = 0
    with open(out_path, "wb") as f:
        for i in tqdm(range(0, len(lines), ENCODE_BATCH), desc=os.path.basename(out_path)):
            enc = tokenizer(lines[i:i + ENCODE_BATCH], add_special_tokens=False)["input_ids"]
            flat = [t for ids in enc for t in ids]
            arr = np.asarray(flat, dtype=np.uint16)
            arr.tofile(f)
            total += arr.size
    return total


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    os.makedirs(DATA_DIR, exist_ok=True)
    print("Loading Salesforce/wikitext : wikitext-103-raw-v1 ...", flush=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    for hf_split, name in SPLITS:
        out_path = os.path.join(DATA_DIR, f"{name}.bin")
        lines = ds[hf_split]["text"]
        n = encode_split(lines, out_path, tokenizer)
        print(f"{name}: {n:,} tokens -> {out_path}", flush=True)

    print("DATA PREP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
