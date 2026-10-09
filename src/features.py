"""RQ4: hand-crafted stylistic features of each family's outputs, independent of the neural models.

Writes results/features.csv (per-family means) and figures/features.png.

Usage: python src/features.py
"""
import re

import numpy as np
import pandas as pd

from plotstyle import INK_2, SERIES, plt, save
from text import tokenize

FEATURES = {
    # name: (label for the figure, function of the output text)
    "tokens": ("Length (tokens)", lambda t: len(tokenize(t))),
    "lines": ("Lines", lambda t: t.count("\n") + 1),
    "headers": ("Markdown headers", lambda t: len(re.findall(r"^\s{0,3}#{1,6}\s", t, re.M))),
    "bullets": ("Bullet / numbered items", lambda t: len(re.findall(r"^\s*([-*+]|\d+[.)])\s", t, re.M))),
    "bold": ("Bold spans", lambda t: t.count("**") // 2),
    "rules": ("Horizontal rules (---)", lambda t: len(re.findall(r"^\s*[-*_]{3,}\s*$", t, re.M))),
    "emoji": ("Emoji", lambda t: len(re.findall(r"[\U0001F300-\U0001FAFF☀-➿]", t))),
    "em_dash": ("Em dashes", lambda t: t.count("—")),
    "type_token": ("Type/token ratio (first 200)",
                   lambda t: len(set(w.lower() for w in tokenize(t)[:200])) / max(1, min(200, len(tokenize(t))))),
}


def main():
    df = pd.read_parquet("data/dataset.parquet")
    for name, (_, fn) in FEATURES.items():
        df[name] = df.output.map(fn)
    table = df.groupby("family")[list(FEATURES)].mean().round(3)
    table["median_tokens"] = df.groupby("family").tokens.median()
    table.to_csv("results/features.csv")
    print(table.to_string())

    # Small multiples: one feature per panel, families in a fixed order, single hue (no identity needed).
    fams = sorted(df.family.unique())
    fig, axes = plt.subplots(3, 3, figsize=(9, 7.2), sharey=True)
    for ax, (name, (label, _)) in zip(axes.flat, FEATURES.items()):
        vals = table[name].reindex(fams)
        ax.barh(fams, vals, color=SERIES[0], height=0.62)
        ax.set_title(label, fontsize=9)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        for y, v in enumerate(vals):
            ax.text(v, y, f" {v:.2f}" if v < 10 else f" {v:.0f}", va="center", fontsize=7, color=INK_2)
        ax.set_xlim(0, vals.max() * 1.25)
    axes[0, 0].invert_yaxis()  # shared y: first family on top in every panel
    fig.suptitle("Mean stylistic features of each family's responses", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "features")


if __name__ == "__main__":
    main()
