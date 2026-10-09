"""Aggregate results/runs/*.json into the report's tables and figures. Safe to run while the sweep
is still going: anything not finished yet is left out.

Writes results/summary.csv, results/summary.md and figures/{fields,confusion,per_class,ablation,
cross_domain,curves}.png.

Usage: python src/analyze.py [--runs results/runs]
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd

from plotstyle import BLUES, INK_2, INK_MUTED, MODEL_COLOR, MODEL_LABEL, plt, save

FIELDS = ["input", "output", "both"]
FIELD_LABEL = {"input": "Input only", "output": "Output only", "both": "Input + output"}
TRANSFORMS = ["none", "no_markdown", "words_only"]
TRANSFORM_LABEL = {"none": "Raw text", "no_markdown": "Markdown removed", "words_only": "Lowercase words only"}


def load(runs_dir):
    runs = [json.load(open(p)) for p in sorted(glob.glob(f"{runs_dir}/*.json"))]
    rows = [dict(name=r["name"], model=r["model"], field=r["field"], transform=r["transform"],
                 held_out=r["held_out"] or "", seed=r["seed"], epochs=r["epochs_run"],
                 minutes=r["train_seconds"] / 60, n_params=r["n_params"],
                 acc=r["test"]["acc"], f1=r["test"]["macro_f1"],
                 ood_acc=r.get("test_ood", {}).get("acc"), ood_f1=r.get("test_ood", {}).get("macro_f1"))
            for r in runs]
    return runs, pd.DataFrame(rows)


def summarize(df):
    """Mean +- std over seeds for each configuration."""
    keys = ["model", "field", "transform", "held_out"]
    g = df.groupby(keys)
    out = g[["acc", "f1", "ood_acc", "ood_f1", "epochs", "minutes"]].mean()
    out["f1_std"] = g.f1.std().fillna(0)
    out["acc_std"] = g.acc.std().fillna(0)
    out["seeds"] = g.size()
    return out.reset_index()


def grouped_bars(ax, groups, labels, values, errors):
    """values/errors: {model: [one per group]}; CNN and LSTM side by side in each group."""
    models = [m for m in ("cnn", "lstm") if m in values]
    w = 0.36
    for i, m in enumerate(models):
        x = np.arange(len(groups)) + (i - (len(models) - 1) / 2) * (w + 0.02)
        v = np.array(values[m], dtype=float)
        err = errors.get(m)
        err = err if err is not None and np.nansum(err) > 0 else None  # single seed: no error bar
        ax.bar(x, v, w, color=MODEL_COLOR[m], label=MODEL_LABEL[m], yerr=err,
               error_kw=dict(ecolor=INK_2, elinewidth=1, capsize=3))
        for xi, vi in zip(x, v):
            if not np.isnan(vi):
                ax.text(xi, vi + 0.015, f"{vi:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK_2)
    ax.set_xticks(np.arange(len(groups)), [labels[g] for g in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)  # keep every group's slot even before its runs finish
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test macro-F1")
    ax.legend(loc="upper left", ncol=2)


def chance(ax, n_classes):
    ax.axhline(1 / n_classes, color=INK_MUTED, lw=1, ls="--")
    ax.text(ax.get_xlim()[1], 1 / n_classes, f" chance ({1 / n_classes:.3f})", va="center",
            fontsize=7.5, color=INK_MUTED)


def pick(summary, **kw):
    q = summary
    for k, v in kw.items():
        q = q[q[k] == v]
    return q


def fig_fields(summary, n_classes):
    base = pick(summary, transform="none", held_out="")
    vals, errs = {}, {}
    for m in ("cnn", "lstm"):
        s = base[base.model == m].set_index("field").reindex(FIELDS)
        if s.f1.notna().any():
            vals[m], errs[m] = s.f1.tolist(), s.f1_std.fillna(0).tolist()
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(6, 3.4))
    grouped_bars(ax, FIELDS, FIELD_LABEL, vals, errs)
    chance(ax, n_classes)
    ax.set_title("RQ1/RQ2: which text identifies the LLM family?")
    save(fig, "fields")


def fig_ablation(summary, n_classes):
    base = pick(summary, field="output", held_out="")
    vals, errs = {}, {}
    for m in ("cnn", "lstm"):
        s = base[base.model == m].set_index("transform").reindex(TRANSFORMS)
        if s.f1.notna().any():
            vals[m], errs[m] = s.f1.tolist(), s.f1_std.fillna(0).tolist()
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(6, 3.4))
    grouped_bars(ax, TRANSFORMS, TRANSFORM_LABEL, vals, errs)
    chance(ax, n_classes)
    ax.set_title("RQ4: how much of the signal is formatting? (output only)")
    save(fig, "ablation")


def fig_cross_domain(summary):
    ood = summary[(summary.held_out != "") & (summary.field == "output") & (summary.transform == "none")]
    if ood.empty:
        return
    domains = sorted(ood.held_out.unique())
    models = [m for m in ("cnn", "lstm") if m in set(ood.model)]
    fig, axes = plt.subplots(1, len(models), figsize=(3.3 * len(models), 3.2), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        s = ood[ood.model == m].set_index("held_out").reindex(domains)
        x = np.arange(len(domains))
        # Same model in both bars, so the two bars are told apart by lightness + legend, not hue.
        ax.bar(x - 0.19, s.f1, 0.36, color=MODEL_COLOR[m], label="Seen domains (test split)")
        ax.bar(x + 0.19, s.ood_f1, 0.36, color=MODEL_COLOR[m], alpha=0.45, hatch="////",
               edgecolor="white", linewidth=0, label="Held-out domain")
        for xi, a, b in zip(x, s.f1, s.ood_f1):
            ax.text(xi - 0.19, a + 0.015, f"{a:.2f}", ha="center", fontsize=7.5, color=INK_2)
            ax.text(xi + 0.19, b + 0.015, f"{b:.2f}", ha="center", fontsize=7.5, color=INK_2)
        ax.set_xticks(x, [f"hold out\n{d}" for d in domains])
        ax.set_title(MODEL_LABEL[m])
        ax.set_ylim(0, 1)
    axes[0][0].set_ylabel("Macro-F1")
    axes[0][0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle("RQ3: do fingerprints transfer to an unseen domain?", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "cross_domain")


def main_runs(runs, model):
    """The seed-0 output-only raw-text run for one model, or None."""
    for r in runs:
        if (r["model"], r["field"], r["transform"], r["held_out"], r["seed"]) == (model, "output", "none", None, 0):
            return r
    return None


def fig_confusion(runs):
    rs = [r for r in (main_runs(runs, "cnn"), main_runs(runs, "lstm")) if r]
    if not rs:
        return
    fig, axes = plt.subplots(1, len(rs), figsize=(4.4 * len(rs), 4), squeeze=False)
    for ax, r in zip(axes[0], rs):
        cm = np.array(r["confusion"], dtype=float)
        cm = cm / cm.sum(1, keepdims=True)
        ax.imshow(cm, cmap=BLUES, vmin=0, vmax=1)
        cls = r["classes"]
        ax.set_xticks(range(len(cls)), cls, rotation=45, ha="right")
        ax.set_yticks(range(len(cls)), cls)
        for i in range(len(cls)):
            for j in range(len(cls)):
                ax.text(j, i, f"{cm[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.55 else INK_2)
        ax.grid(False)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{MODEL_LABEL[r['model']]} (output only, row-normalized)")
    fig.tight_layout()
    save(fig, "confusion")


def fig_per_class(runs):
    rs = {m: r for m in ("cnn", "lstm") if (r := main_runs(runs, m))}
    if not rs:
        return
    cls = next(iter(rs.values()))["classes"]
    vals = {m: [r["test_per_class_f1"][c] for c in cls] for m, r in rs.items()}
    fig, ax = plt.subplots(figsize=(7, 3.4))
    grouped_bars(ax, cls, {c: c for c in cls}, vals, {})
    ax.set_title("Per-family test F1 (output only)")
    ax.set_ylabel("F1")
    save(fig, "per_class")


def fig_curves(runs):
    rs = [r for r in (main_runs(runs, "cnn"), main_runs(runs, "lstm")) if r]
    if not rs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3))
    for r in rs:
        ep = [c["epoch"] for c in r["curve"]]
        kw = dict(color=MODEL_COLOR[r["model"]], marker="o", markersize=4, label=MODEL_LABEL[r["model"]])
        axes[0].plot(ep, [c["train_loss"] for c in r["curve"]], **kw)
        axes[1].plot(ep, [c["val_macro_f1"] for c in r["curve"]], **kw)
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation macro-F1")
    for ax in axes:
        ax.set_xlabel("Epoch")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    save(fig, "curves")


def write_tables(df, summary):
    df.to_csv("results/all_runs.csv", index=False)
    summary.to_csv("results/summary.csv", index=False)
    s = summary.copy()
    s["test macro-F1"] = s.apply(lambda r: f"{r.f1:.3f} ± {r.f1_std:.3f}" if r.seeds > 1 else f"{r.f1:.3f}", axis=1)
    s["test acc"] = s.acc.map("{:.3f}".format)
    s["held-out F1"] = s.ood_f1.map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    s["epochs"] = s.epochs.map("{:.1f}".format)
    s["min/run"] = s.minutes.map("{:.1f}".format)
    cols = ["model", "field", "transform", "held_out", "seeds", "test acc", "test macro-F1", "held-out F1",
            "epochs", "min/run"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in s[cols].itertuples(index=False)]
    with open("results/summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(s[cols].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    a = ap.parse_args()
    runs, df = load(a.runs)
    if df.empty:
        print("no finished runs yet")
        return
    summary = summarize(df)
    n_classes = len(runs[0]["classes"])
    write_tables(df, summary)
    fig_fields(summary, n_classes)
    fig_ablation(summary, n_classes)
    fig_cross_domain(summary)
    fig_confusion(runs)
    fig_per_class(runs)
    fig_curves(runs)


if __name__ == "__main__":
    main()
