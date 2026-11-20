"""Aggregate Lion-MANE study runs into summary tables and figures.

Reads results/wikitext103/<optimizer>/seed<K><tag>.csv (+ .diag.jsonl, .final.json)
and writes:

  results/summary.md, results/summary.csv
  results/plots/*.png

Runs whose tag contains "_pilot" are reported in the pilot section; untagged runs
are the headline study. Colors follow the entity across every figure
(validated palette): Lion-MANE #2a78d6, AdamW #1baf7a, Lion #eda100.

    python src/scripts/analyze.py
"""

import glob
import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "wikitext103")
OUT_DIR = os.path.join(REPO_ROOT, "results")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")

RUN_RE = re.compile(r"seed(\d+)(.*)\.csv$")

# --- palette (validated; see dataviz notes) ---------------------------------
COLORS = {"lion_mane": "#2a78d6", "adamw": "#1baf7a", "lion": "#eda100"}
SEQ = ["#2a78d6", "#6da7ec", "#9ec5f4", "#cde2fb"]  # single-hue ramp (dark→light)
LABELS = {"lion_mane": "Lion-MANE", "adamw": "AdamW", "lion": "Lion"}
ORDER = ["lion_mane", "adamw", "lion"]
INK, SECOND, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
LINESTYLES = ["-", "--", ":", "-."]


def style_ax(ax, xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel(xlabel, color=SECOND, fontsize=9)
    ax.set_ylabel(ylabel, color=SECOND, fontsize=9)


def end_label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                color=color, fontsize=8.5, fontweight="bold", va="center")


def save(fig, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, REPO_ROOT)}")


def discover():
    """{(optimizer, tag): {seed: {"df":…, "final":…, "diag":…}}}"""
    runs = {}
    for csv_path in glob.glob(os.path.join(RESULTS_DIR, "*", "seed*.csv")):
        opt = os.path.basename(os.path.dirname(csv_path))
        m = RUN_RE.search(os.path.basename(csv_path))
        if not m:
            continue
        seed, tag = int(m.group(1)), m.group(2)
        if "_smoke" in tag:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:  # noqa: BLE001
            print(f"skip {csv_path}: {e}")
            continue
        if df.empty or "val_loss" not in df:
            continue
        entry = {"df": df, "final": None, "diag": None}
        fj = os.path.splitext(csv_path)[0] + ".final.json"
        if os.path.exists(fj):
            with open(fj) as f:
                entry["final"] = json.load(f)
        dj = os.path.splitext(csv_path)[0] + ".diag.jsonl"
        if os.path.exists(dj):
            with open(dj) as f:
                entry["diag"] = [json.loads(line) for line in f if line.strip()]
        runs.setdefault((opt, tag), {})[seed] = entry
    return runs


# ---------------------------------------------------------------- figures ---

def fig_val_loss(main_runs):
    if not main_runs:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, frac, title in ((a1, 0.0, "validation loss"), (a2, 0.55, "final stretch")):
        style_ax(ax, "step", "val loss (fixed windows)")
        ymin, ymax = np.inf, -np.inf
        for opt in ORDER:
            if (opt, "") not in main_runs:
                continue
            for seed, e in sorted(main_runs[(opt, "")].items()):
                df = e["df"]
                cut = df[df["step"] >= frac * df["step"].max()]
                ax.plot(cut["step"], cut["val_loss"], color=COLORS[opt],
                        linewidth=2, zorder=3,
                        label=LABELS[opt] if seed == min(main_runs[(opt, "")]) else None)
                ymin = min(ymin, cut["val_loss"].min())
                ymax = max(ymax, np.percentile(cut["val_loss"], 95))
                if ax is a1:
                    end_label(ax, cut["step"].iloc[-1], cut["val_loss"].iloc[-1],
                              LABELS[opt], COLORS[opt])
        if np.isfinite(ymin):
            ax.set_ylim(ymin - 0.02, ymax + 0.05)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    a1.legend(frameon=False, fontsize=8.5, labelcolor=SECOND)
    save(fig, "val_loss.png")


def fig_final_bars(main_runs):
    rows = []
    for opt in ORDER:
        if (opt, "") not in main_runs:
            continue
        for seed, e in main_runs[(opt, "")].items():
            if e["final"] and "val_full_ppl" in e["final"]:
                rows.append((opt, seed, e["final"]["val_full_ppl"],
                             e["final"].get("test_full_ppl", np.nan)))
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["opt", "seed", "val", "test"])
    agg = df.groupby("opt").agg(val=("val", "mean"), test=("test", "mean"))
    opts = [o for o in ORDER if o in agg.index]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    style_ax(ax, "", "perplexity (full split)")
    x = np.arange(len(opts))
    for j, (col, alpha) in enumerate((("val", 1.0), ("test", 0.55))):
        vals = [agg.loc[o, col] for o in opts]
        bars = ax.bar(x + (j - 0.5) * 0.36, vals, width=0.34,
                      color=[COLORS[o] for o in opts], alpha=alpha,
                      edgecolor=SURFACE, linewidth=2, zorder=3)
        for b, v in zip(bars, vals):
            if not (isinstance(v, float) and math.isnan(v)):
                ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", color=INK, fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[o] for o in opts], color=INK, fontsize=9.5)
    ax.set_title("final perplexity — solid: validation, faded: test",
                 color=INK, fontsize=10, loc="left")
    save(fig, "final_ppl.png")


def fig_train_stability(main_runs):
    if not main_runs:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    style_ax(ax, "step", "rolling std of train loss (8 windows)")
    for opt in ORDER:
        if (opt, "") not in main_runs:
            continue
        seed = min(main_runs[(opt, "")])
        df = main_runs[(opt, "")][seed]["df"]
        r = df["train_loss"].rolling(8).std()
        ax.plot(df["step"], r, color=COLORS[opt], linewidth=2, zorder=3, label=LABELS[opt])
        if np.isfinite(r.iloc[-1]):
            end_label(ax, df["step"].iloc[-1], r.iloc[-1], LABELS[opt], COLORS[opt])
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=SECOND)
    ax.set_title("training smoothness (lower = more stable)", color=INK, fontsize=10, loc="left")
    save(fig, "train_stability.png")


def _mane_entries(main_runs, fallback):
    return main_runs.get(("lion_mane", "")) or (fallback or {})


def fig_gate_dynamics(main_runs, fallback=None):
    entries = _mane_entries(main_runs, fallback)
    if not entries:
        return
    df = entries[min(entries)]["df"]
    if "gate_mean" not in df or df["gate_mean"].dropna().empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    style_ax(ax, "step", "gate value / fraction")
    for col, c, lab in (("gate_mean", SEQ[0], "mean gate γ"),
                        ("gate_rms", SEQ[1], "RMS gate (update size / η)"),
                        ("gate_frac_lt_05", SEQ[2], "frac γ < 0.5"),
                        ("gate_frac_lt_01", SEQ[3], "frac γ < 0.1")):
        if col in df and not df[col].dropna().empty:
            ax.plot(df["step"], df[col], color=c, linewidth=2, zorder=3, label=lab)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=SECOND, ncol=2)
    ax.set_title("MANE gate dynamics", color=INK, fontsize=10, loc="left")
    save(fig, "gate_dynamics.png")


def fig_sign_agreement(main_runs, fallback=None):
    entries = _mane_entries(main_runs, fallback)
    if not entries:
        return
    df = entries[min(entries)]["df"]
    if "sign_agree" not in df or df["sign_agree"].dropna().empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    style_ax(ax, "step", "fraction of coordinates")
    ax.plot(df["step"], df["sign_agree"], color=SEQ[0], linewidth=2, zorder=3)
    ax.axhline(0.5, color=AXIS, linewidth=1, linestyle="--", zorder=1)
    ax.annotate("coin flip", (df["step"].max() * 0.99, 0.5), xytext=(0, -10),
                textcoords="offset points", ha="right", color=MUTED, fontsize=8)
    ax.set_title("fold sign agreement — how much of the model has signal",
                 color=INK, fontsize=10, loc="left")
    save(fig, "sign_agreement.png")


def fig_update_rms(main_runs):
    if not main_runs:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    style_ax(ax, "step", "update RMS (per-coordinate)")
    any_line = False
    for opt in ORDER:
        if (opt, "") not in main_runs:
            continue
        seed = min(main_runs[(opt, "")])
        df = main_runs[(opt, "")][seed]["df"]
        if "update_rms" not in df or df["update_rms"].dropna().empty:
            continue
        ax.plot(df["step"], df["update_rms"], color=COLORS[opt], linewidth=2,
                zorder=3, label=LABELS[opt])
        any_line = True
    if not any_line:
        plt.close(fig)
        return
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=SECOND)
    ax.set_title("actual step sizes taken", color=INK, fontsize=10, loc="left")
    save(fig, "update_rms.png")


def fig_group_gates(main_runs, fallback=None):
    entries = _mane_entries(main_runs, fallback)
    if not entries:
        return
    diag = entries[min(entries)]["diag"]
    if not diag:
        return
    groups = [g for g in ("emb", "attn", "ffn", "ln", "other")
              if any((d.get("gate_mean") or {}).get(g) is not None for d in diag)]
    if not groups:
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(2.6 * len(groups), 3.0),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, groups):
        style_ax(ax, "step", "mean gate γ" if g == groups[0] else "")
        xs = [d["step"] for d in diag if (d.get("gate_mean") or {}).get(g) is not None]
        ys = [d["gate_mean"][g] for d in diag if (d.get("gate_mean") or {}).get(g) is not None]
        ax.plot(xs, ys, color=SEQ[0], linewidth=2, zorder=3)
        ax.set_ylim(0, 1.02)
        ax.set_title(g, color=INK, fontsize=9.5, loc="left")
    fig.suptitle("where the gate acts — per layer group", color=INK, fontsize=10,
                 x=0.01, ha="left")
    save(fig, "group_gates.png")


def fig_pilots(pilot_runs):
    if not pilot_runs:
        return
    present = [o for o in ORDER if any(k[0] == o for k in pilot_runs)]
    fig, axes = plt.subplots(1, len(present), figsize=(4.2 * len(present), 4.0),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, opt in zip(axes, present):
        style_ax(ax, "step", "val loss" if opt == present[0] else "")
        tags = sorted(t for (o, t) in pilot_runs if o == opt)
        for k, tag in enumerate(tags):
            for seed, e in sorted(pilot_runs[(opt, tag)].items()):
                df = e["df"]
                lr = tag.replace("_pilot_lr", "")
                ax.plot(df["step"], df["val_loss"], color=COLORS[opt], linewidth=2,
                        linestyle=LINESTYLES[k % len(LINESTYLES)], zorder=3,
                        label=f"lr {lr}")
        ax.set_title(LABELS[opt], color=INK, fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8, labelcolor=SECOND)
    axes[0].set_ylim(top=8.0)
    fig.suptitle("pilot lr sweep (4k steps)", color=INK, fontsize=10, x=0.01, ha="left")
    save(fig, "pilot_val_loss.png")


# ---------------------------------------------------------------- summary ---

def _ms(x):
    x = [v for v in x if not (isinstance(v, float) and math.isnan(v))]
    if not x:
        return "—"
    fmt = ".4f" if np.mean(x) < 100 else ".2f"
    if len(x) == 1:
        return f"{x[0]:{fmt}}"
    return f"{np.mean(x):{fmt}}±{np.std(x):{fmt}}"


def write_summary(runs):
    main = {k: v for k, v in runs.items() if k[1] == ""}
    pilots = {k: v for k, v in runs.items() if "_pilot" in k[1]}
    ablations = {k: v for k, v in runs.items() if k[1] and "_pilot" not in k[1]}

    lines = ["# Lion-MANE study summary\n"]
    rows = []
    if main:
        lines.append("## Headline (WikiText-103, 39M LM)\n")
        lines.append("| optimizer | seeds | final val loss | best val loss | "
                     "full val PPL | full test PPL | wall (min) |")
        lines.append("|---|---|---|---|---|---|---|")
        for opt in ORDER:
            if (opt, "") not in main:
                continue
            fin, best, vppl, tppl, wall = [], [], [], [], []
            for seed, e in main[(opt, "")].items():
                df = e["df"]
                fin.append(df["val_loss"].iloc[-1])
                best.append(df["val_loss"].min())
                vppl.append((e["final"] or {}).get("val_full_ppl", np.nan))
                tppl.append((e["final"] or {}).get("test_full_ppl", np.nan))
                wall.append((e["final"] or {}).get("wall_min", np.nan))
                rows.append(dict(section="main", optimizer=opt, seed=seed, tag="",
                                 final_val=fin[-1], best_val=best[-1],
                                 val_full_ppl=vppl[-1], test_full_ppl=tppl[-1]))
            lines.append(f"| {LABELS[opt]} | {len(main[(opt, '')])} | {_ms(fin)} | "
                         f"{_ms(best)} | {_ms(vppl)} | {_ms(tppl)} | {_ms(wall)} |")
        if ("lion_mane", "") in main and ("adamw", "") in main:
            mane = np.mean([e["df"]["val_loss"].iloc[-1]
                            for e in main[("lion_mane", "")].values()])
            adam = np.mean([e["df"]["val_loss"].iloc[-1]
                            for e in main[("adamw", "")].values()])
            d = adam - mane
            verdict = ("**Lion-MANE BEATS AdamW**" if d > 0
                       else "**Lion-MANE behind AdamW**")
            lines.append(f"\n{verdict}: final val loss delta (AdamW − MANE) = "
                         f"{d:+.4f} nats ({math.exp(adam) / math.exp(mane):.4f}× PPL ratio).\n")

    if pilots:
        lines.append("\n## Pilot lr sweep (4k steps)\n")
        lines.append("| optimizer | lr | final val loss @4k |")
        lines.append("|---|---|---|")
        for (opt, tag) in sorted(pilots, key=lambda k: (ORDER.index(k[0])
                                                        if k[0] in ORDER else 9, k[1])):
            for seed, e in sorted(pilots[(opt, tag)].items()):
                v = e["df"]["val_loss"].iloc[-1]
                lines.append(f"| {LABELS.get(opt, opt)} | "
                             f"{tag.replace('_pilot_lr', '')} | {v:.4f} |")
                rows.append(dict(section="pilot", optimizer=opt, seed=seed,
                                 tag=tag, final_val=v))

    if ablations:
        lines.append("\n## Ablations\n")
        lines.append("| optimizer | tag | seeds | final val loss |")
        lines.append("|---|---|---|---|")
        for (opt, tag) in sorted(ablations):
            vals = [e["df"]["val_loss"].iloc[-1] for e in ablations[(opt, tag)].values()]
            lines.append(f"| {LABELS.get(opt, opt)} | {tag} | {len(vals)} | "
                         f"{np.mean(vals):.4f} |")
            for seed, e in ablations[(opt, tag)].items():
                rows.append(dict(section="ablation", optimizer=opt, seed=seed,
                                 tag=tag, final_val=e["df"]["val_loss"].iloc[-1]))

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    print("  wrote results/summary.md, results/summary.csv")
    return main, pilots


def main():
    runs = discover()
    if not runs:
        print("no runs found")
        return
    main_runs, pilot_runs = write_summary(runs)
    mane_pilots = {s: e for (o, t), m in sorted(pilot_runs.items())
                   if o == "lion_mane" for s, e in m.items()}
    fig_val_loss(main_runs)
    fig_final_bars(main_runs)
    fig_train_stability(main_runs)
    fig_gate_dynamics(main_runs, mane_pilots)
    fig_sign_agreement(main_runs, mane_pilots)
    fig_update_rms(main_runs)
    fig_group_gates(main_runs, mane_pilots)
    fig_pilots(pilot_runs)
    print("analysis complete")


if __name__ == "__main__":
    main()
