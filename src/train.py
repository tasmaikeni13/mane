"""Entry point for every run in the MANE study (WikiText-103, ~39M LM).

One command runs a single (optimizer, seed) configuration and writes
``<out_dir>/wikitext103/<optimizer>/seed<seed><tag>.csv`` (+ ``.diag.jsonl``
and ``.final.json``). The three arms differ ONLY in the optimizer update:

    python src/train.py --optimizer adamw --seed 0 --gpu 0
    python src/train.py --optimizer lion  --seed 0 --gpu 1
    python src/train.py --optimizer mane  --seed 0 --gpu 1

Notes for the RTX 6000 (Turing) target: fp32 only — do NOT enable bf16.
"""

import argparse
import os
import sys
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from optim.factory import default_hparams  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="MANE vs AdamW vs Lion on WikiText-103.")
    p.add_argument("--optimizer", required=True, choices=["adamw", "lion", "mane"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=256)
    # optimizer overrides (defaults come from optim.factory.DEFAULTS)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--beta1", type=float, default=None)
    p.add_argument("--beta2", type=float, default=None)
    p.add_argument("--warmup", type=int, default=None)
    # MANE-specific
    p.add_argument("--beta-v", type=float, default=None)
    p.add_argument("--budget-ratio", type=float, default=None,
                   help="ratio of total noise budget relative to vanilla Lion (default: 1.0)")
    p.add_argument("--trust-cap", type=float, default=None,
                   help="maximum trust multiplier Γ for reliable coordinates (default: 3.0)")
    # logging / eval
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--diag-every", type=int, default=1000)
    p.add_argument("--eval-iters", type=int, default=24)
    p.add_argument("--eval-batch", type=int, default=32)
    p.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data"))
    p.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "results"))
    p.add_argument("--tag", default="", help="suffix for ablation run filenames")
    return p.parse_args()


def build_config(args):
    hp = default_hparams(args.optimizer)
    betas = (args.beta1 if args.beta1 is not None else hp["betas"][0],
             args.beta2 if args.beta2 is not None else hp["betas"][1])
    return SimpleNamespace(
        optimizer=args.optimizer,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr if args.lr is not None else hp["lr"],
        weight_decay=args.weight_decay if args.weight_decay is not None else hp["weight_decay"],
        betas=betas,
        warmup=args.warmup if args.warmup is not None else hp.get("warmup", 0),
        beta_v=args.beta_v if args.beta_v is not None else hp.get("beta_v", 0.99),
        budget_ratio=args.budget_ratio if args.budget_ratio is not None else hp.get("budget_ratio", 1.0),
        trust_cap=args.trust_cap if args.trust_cap is not None else hp.get("trust_cap", 3.0),
        log_every=args.log_every,
        diag_every=args.diag_every,
        eval_iters=args.eval_iters,
        eval_batch=args.eval_batch,
        data_dir=args.data_dir,
        tag=f"{args.optimizer}-s{args.seed}{args.tag}",
    )


def main():
    args = parse_args()
    cfg = build_config(args)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available, running on CPU.", flush=True)

    out_dir = os.path.join(args.out_dir, "wikitext103", cfg.optimizer)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"seed{cfg.seed}{args.tag}.csv")

    print(f"=== {cfg.tag} | device={device} | lr={cfg.lr} wd={cfg.weight_decay} "
          f"betas={cfg.betas} steps={cfg.steps} batch={cfg.batch_size}x{cfg.seq_len}"
          + (f" | budget_ratio={cfg.budget_ratio} trust_cap={cfg.trust_cap} beta_v={cfg.beta_v}"
             if cfg.optimizer == "mane" else "") + " ===", flush=True)
    print(f"    -> {out_csv}", flush=True)

    from engine import lm
    lm.run(cfg, device, out_csv)
    print(f"=== done: {cfg.tag} ===", flush=True)


if __name__ == "__main__":
    main()
