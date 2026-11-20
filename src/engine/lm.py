"""WikiText-103 training/eval loop for the Lion-MANE study.

Fairness contract between arms:
  * identical model init per seed (set_seed before model build),
  * identical total tokens per step: baselines draw one batch of ``batch_size``;
    Lion-MANE draws ``batch_size // 2`` from each of two disjoint folds,
  * identical schedule (linear warmup + cosine), single param group, uniform wd,
  * identical FIXED validation windows (seeded independently of the run seed),
    so val curves are comparable point-for-point across arms and seeds.

Outputs per run:
  <out>.csv         step-level metrics (loss, val, lr, throughput, gate stats…)
  <out>.diag.jsonl  periodic per-layer-group diagnostics
  <out>.final.json  full-split final evaluation (entire val + test sets)
"""

import csv
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from models.transformer import LanguageModel
from optim.factory import make_optimizer, make_scheduler, FOLD_BASED
from utils.seed import set_seed
from utils.lm_data import get_batch

EVAL_SEED = 1234  # fixed eval windows, shared by every run


def _group_of(name: str) -> str:
    if name.startswith("embedding"):
        return "emb"
    if "self_attn" in name:
        return "attn"
    if "linear1" in name or "linear2" in name:
        return "ffn"
    if "norm" in name:
        return "ln"
    return "other"


def _batch_from_ix(data, ix, seq_len):
    x = torch.stack([torch.from_numpy(data[i:i + seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + seq_len].astype(np.int64)) for i in ix])
    return x, y


@torch.no_grad()
def evaluate_fixed(model, data, eval_ix, criterion, device, eval_batch, seq_len):
    """Mean CE over the FIXED validation windows (same for every run)."""
    model.eval()
    total, n = 0.0, 0
    for s in range(0, len(eval_ix), eval_batch):
        ix = eval_ix[s:s + eval_batch]
        x, y = _batch_from_ix(data, ix.tolist(), seq_len)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        n += 1
    model.train()
    return total / max(n, 1)


@torch.no_grad()
def full_eval(model, data, criterion, device, seq_len, batch=32):
    """Mean CE over non-overlapping windows covering an entire split."""
    model.eval()
    n_win = (len(data) - 1) // seq_len
    starts = [w * seq_len for w in range(n_win)]
    total, n = 0.0, 0
    for s in range(0, len(starts), batch):
        ix = starts[s:s + batch]
        x, y = _batch_from_ix(data, ix, seq_len)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        n += 1
    model.train()
    return total / max(n, 1)


@torch.no_grad()
def _grad_norm(params):
    sq = None
    for p in params:
        if p.grad is None:
            continue
        s = p.grad.square().sum()
        sq = s if sq is None else sq + s
    return 0.0 if sq is None else sq.sqrt().item()


@torch.no_grad()
def _fold_agreement(params, grads_a):
    """(sign-agreement fraction, ‖ḡ‖) of the two fold gradients."""
    agree, count, sq = None, 0, None
    for p, g_a in zip(params, grads_a):
        if p.grad is None or g_a is None:
            continue
        a = (torch.sign(g_a) == torch.sign(p.grad)).sum()
        agree = a if agree is None else agree + a
        s = (0.5 * (g_a + p.grad)).square().sum()
        sq = s if sq is None else sq + s
        count += p.numel()
    if not count:
        return float("nan"), float("nan")
    return agree.item() / count, sq.sqrt().item()


@torch.no_grad()
def _adamw_update_rms(optimizer):
    """RMS of AdamW's would-be update in lr units: m̂/(sqrt(v̂)+eps)."""
    sq_sum, count = 0.0, 0
    for group in optimizer.param_groups:
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        for p in group["params"]:
            state = optimizer.state.get(p)
            if not state or "exp_avg" not in state:
                continue
            t = state["step"] if isinstance(state["step"], (int, float)) else state["step"].item()
            if t == 0:
                continue
            m_hat = state["exp_avg"] / (1 - beta1 ** t)
            v_hat = state["exp_avg_sq"] / (1 - beta2 ** t)
            upd = m_hat / (v_hat.sqrt() + eps)
            sq_sum += upd.square().sum().item()
            count += upd.numel()
    return math.sqrt(sq_sum / count) if count else float("nan")


def run(cfg, device, out_csv):
    set_seed(cfg.seed)
    model = LanguageModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params/1e6:.2f}M", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(cfg.optimizer, model.parameters(), cfg.lr,
                               cfg.weight_decay, betas=cfg.betas, beta_v=cfg.beta_v,
                               budget_ratio=getattr(cfg, "budget_ratio", 1.0),
                               trust_cap=getattr(cfg, "trust_cap", 3.0))
    scheduler = make_scheduler(optimizer, cfg.steps, warmup_steps=cfg.warmup)

    train_data = np.memmap(os.path.join(cfg.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(cfg.data_dir, "val.bin"), dtype=np.uint16, mode="r")
    test_path = os.path.join(cfg.data_dir, "test.bin")
    test_data = np.memmap(test_path, dtype=np.uint16, mode="r") if os.path.exists(test_path) else None

    eval_gen = torch.Generator().manual_seed(EVAL_SEED)
    eval_ix = torch.randint(len(val_data) - cfg.seq_len - 1,
                            (cfg.eval_iters * cfg.eval_batch,), generator=eval_gen)

    fold_based = cfg.optimizer in FOLD_BASED
    if fold_based:
        mid = len(train_data) // 2
        fold_a, fold_b = train_data[:mid], train_data[mid:]
        fold_batch = cfg.batch_size // 2

    params = list(model.parameters())
    named = [(n, p) for n, p in model.named_parameters()]
    groups = sorted({_group_of(n) for n, _ in named})

    diag_path = os.path.splitext(out_csv)[0] + ".diag.jsonl"
    final_path = os.path.splitext(out_csv)[0] + ".final.json"

    cols = ["step", "train_loss", "train_loss_max", "val_loss", "val_ppl", "lr",
            "tok_per_s", "wall_min", "grad_norm", "update_rms",
            "gate_mean", "gate_rms", "gate_frac_lt_05", "gate_frac_lt_01", "sign_agree"]

    t0 = time.time()
    t_log = t0
    tokens_per_step = cfg.batch_size * cfg.seq_len

    with open(out_csv, "w", newline="") as f, open(diag_path, "w") as fd:
        writer = csv.writer(f)
        writer.writerow(cols)

        model.train()
        loss_acc, loss_max, n_acc = 0.0, 0.0, 0
        for step in range(cfg.steps):
            is_log = (step % cfg.log_every == 0) or (step == cfg.steps - 1)
            is_diag = (step % cfg.diag_every == 0) or (step == cfg.steps - 1)

            if fold_based:
                xa, ya = get_batch(fold_a, fold_batch, cfg.seq_len)
                xb, yb = get_batch(fold_b, fold_batch, cfg.seq_len)
                xa, ya = xa.to(device), ya.to(device)
                xb, yb = xb.to(device), yb.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits_a = model(xa)
                loss_a = criterion(logits_a.reshape(-1, logits_a.size(-1)), ya.reshape(-1))
                loss_a.backward()
                grads_a = [p.grad.detach().clone() if p.grad is not None else None
                           for p in params]

                optimizer.zero_grad(set_to_none=True)
                logits_b = model(xb)
                loss_b = criterion(logits_b.reshape(-1, logits_b.size(-1)), yb.reshape(-1))
                loss_b.backward()

                sign_agree, grad_norm = (_fold_agreement(params, grads_a)
                                         if is_log else (None, None))
                # step() consumes grads_a and p.grad in-place
                optimizer.step(grads_a=grads_a)
                step_loss = 0.5 * (loss_a.item() + loss_b.item())
            else:
                x, y = get_batch(train_data, cfg.batch_size, cfg.seq_len)
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                loss.backward()
                grad_norm = _grad_norm(params) if is_log else None
                sign_agree = None
                optimizer.step()
                step_loss = loss.item()

            scheduler.step()
            loss_acc += step_loss
            loss_max = max(loss_max, step_loss)
            n_acc += 1

            if is_log:
                train_loss = loss_acc / max(n_acc, 1)
                val_loss = evaluate_fixed(model, val_data, eval_ix, criterion,
                                          device, cfg.eval_batch, cfg.seq_len)
                val_ppl = math.exp(min(val_loss, 20.0))
                lr_now = scheduler.get_last_lr()[0]
                now = time.time()
                tok_s = tokens_per_step * n_acc / max(now - t_log, 1e-9)
                t_log = now

                if fold_based and optimizer.last_stats:
                    st = {k: float(v) for k, v in optimizer.last_stats.items()}
                    gate_mean = st["gate_mean"]
                    gate_rms = st["gate_rms"]
                    gate_lt5 = st["gate_frac_lt_05"]
                    gate_lt1 = st["gate_frac_lt_01"]
                    update_rms = lr_now * gate_rms
                elif cfg.optimizer == "adamw":
                    gate_mean = gate_rms = gate_lt5 = gate_lt1 = ""
                    update_rms = lr_now * _adamw_update_rms(optimizer)
                else:  # vanilla lion: |update| = lr exactly on every coordinate
                    gate_mean = gate_rms = gate_lt5 = gate_lt1 = ""
                    update_rms = lr_now

                writer.writerow([step, f"{train_loss:.5f}", f"{loss_max:.5f}",
                                 f"{val_loss:.5f}", f"{val_ppl:.2f}", f"{lr_now:.3e}",
                                 f"{tok_s:.0f}", f"{(now - t0)/60:.2f}",
                                 f"{grad_norm:.4f}" if grad_norm is not None else "",
                                 f"{update_rms:.3e}" if update_rms != "" else "",
                                 f"{gate_mean:.4f}" if gate_mean != "" else "",
                                 f"{gate_rms:.4f}" if gate_rms != "" else "",
                                 f"{gate_lt5:.4f}" if gate_lt5 != "" else "",
                                 f"{gate_lt1:.4f}" if gate_lt1 != "" else "",
                                 f"{sign_agree:.4f}" if sign_agree is not None else ""])
                f.flush()
                msg = (f"[{cfg.tag}] step {step:6d}  train {train_loss:.4f}  "
                       f"val {val_loss:.4f}  ppl {val_ppl:.1f}  {tok_s/1e3:.1f}k tok/s")
                if fold_based and gate_mean != "":
                    msg += f"  gate {gate_mean:.3f}  agree {sign_agree:.3f}"
                print(msg, flush=True)
                loss_acc, loss_max, n_acc = 0.0, 0.0, 0

            if is_diag:
                rec = {"step": step,
                       "param_rms": {g: 0.0 for g in groups},
                       "numel": {g: 0 for g in groups}}
                with torch.no_grad():
                    for name, p in named:
                        g = _group_of(name)
                        rec["param_rms"][g] += p.square().sum().item()
                        rec["numel"][g] += p.numel()
                    for g in groups:
                        if rec["numel"][g]:
                            rec["param_rms"][g] = math.sqrt(rec["param_rms"][g] / rec["numel"][g])
                if fold_based and getattr(optimizer, "last_param_gate_means", None):
                    pg = {g: [0.0, 0] for g in groups}
                    for (name, p), gm in zip(named, optimizer.last_param_gate_means):
                        g = _group_of(name)
                        pg[g][0] += float(gm) * p.numel()
                        pg[g][1] += p.numel()
                    rec["gate_mean"] = {g: (pg[g][0] / pg[g][1] if pg[g][1] else None)
                                        for g in groups}
                fd.write(json.dumps(rec) + "\n")
                fd.flush()

        # ---- final full-split evaluation (the headline numbers) ----
        final = {"steps": cfg.steps, "tag": cfg.tag, "optimizer": cfg.optimizer,
                 "seed": cfg.seed, "lr": cfg.lr, "weight_decay": cfg.weight_decay,
                 "wall_min": (time.time() - t0) / 60}
        v = full_eval(model, val_data, criterion, device, cfg.seq_len)
        final["val_full_loss"] = v
        final["val_full_ppl"] = math.exp(min(v, 20.0))
        if test_data is not None:
            tst = full_eval(model, test_data, criterion, device, cfg.seq_len)
            final["test_full_loss"] = tst
            final["test_full_ppl"] = math.exp(min(tst, 20.0))
        with open(final_path, "w") as ff:
            json.dump(final, ff, indent=2)
        print(f"[{cfg.tag}] FINAL  val_full {final['val_full_loss']:.5f} "
              f"(ppl {final['val_full_ppl']:.2f})"
              + (f"  test_full {final['test_full_loss']:.5f} "
                 f"(ppl {final['test_full_ppl']:.2f})" if test_data is not None else ""),
              flush=True)
