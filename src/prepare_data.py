"""Build the (LLM_family, LLM_input, LLM_output) dataset from LMArena arena-human-preference-140k.

Each arena battle shows one prompt to two anonymous models, so every battle yields two rows that
share an input. Splits are made by battle id, so a prompt never appears in more than one split.

Usage: python src/prepare_data.py [--per-family N]
"""
import argparse
import glob
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RAW = "data/raw/train-*.parquet"
OUT = "data/dataset.parquet"
SEED = 448

# Ordered: first matching prefix wins. Models from other families (Gemma, Nova, Command, MiniMax,
# Kimi, Hunyuan) are dropped so that every class is one vendor's family.
FAMILY_RULES = [
    (r"^claude", "Claude"),
    (r"^(gpt-|chatgpt|o3|o4)", "GPT"),
    (r"^gemini", "Gemini"),
    (r"^(qwen|qwq)", "Qwen"),
    (r"^deepseek", "DeepSeek"),
    (r"^llama", "Llama"),
    (r"^(mistral|magistral)", "Mistral"),
    (r"^grok", "Grok"),
]


def family_of(model):
    for pattern, fam in FAMILY_RULES:
        if re.match(pattern, model):
            return fam
    return None


def text_of(message):
    """Concatenate text parts; return None if the message has an image."""
    parts = []
    for c in message["content"]:
        if c["type"] != "text":
            return None
        parts.append(c["text"] or "")
    return "".join(parts).strip()


def domain_of(row):
    tags = row["category_tag"]
    if row["is_code"]:
        return "code"
    if tags["math_v0.1"]["math"]:
        return "math"
    if tags["creative_writing_v0.1"]["creative_writing"]:
        return "writing"
    return "general"


def load_rows():
    cols = ["id", "model_a", "model_b", "conversation_a", "conversation_b", "language",
            "is_code", "category_tag", "conv_metadata", "timestamp"]
    rows = []
    for path in sorted(glob.glob(RAW)):
        df = pq.read_table(path, columns=cols).to_pandas()
        df = df[(df["language"] == "en") & (df["conv_metadata"].map(lambda m: m["turns"]) == 1)]
        for r in df.to_dict("records"):
            domain = domain_of(r)
            for side in "ab":
                model = r[f"model_{side}"]
                fam = family_of(model)
                conv = r[f"conversation_{side}"]
                if fam is None or len(conv) != 2:
                    continue
                prompt, response = text_of(conv[0]), text_of(conv[1])
                if not prompt or not response:
                    continue
                rows.append(dict(battle_id=r["id"], family=fam, model=model, input=prompt,
                                 output=response, domain=domain, timestamp=r["timestamp"]))
        print(f"{path}: {len(rows)} rows so far")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=4000)
    args = ap.parse_args()

    df = load_rows()
    print("\nAvailable rows per family:\n", df["family"].value_counts().to_string())

    # Balance classes by sampling whole rows per family, then split by battle id.
    rng = np.random.default_rng(SEED)
    df = df.groupby("family").sample(n=args.per_family, random_state=SEED)
    battles = rng.permutation(df["battle_id"].unique())
    n = len(battles)
    split_of = {b: "train" for b in battles[: int(0.8 * n)]}
    split_of.update({b: "val" for b in battles[int(0.8 * n): int(0.9 * n)]})
    split_of.update({b: "test" for b in battles[int(0.9 * n):]})
    df["split"] = df["battle_id"].map(split_of)
    df = df.reset_index(drop=True)
    df.to_parquet(OUT, index=False)

    print(f"\nWrote {len(df)} rows to {OUT}")
    print(pd.crosstab(df["family"], df["split"]).to_string())
    print(pd.crosstab(df["family"], df["domain"]).to_string())


if __name__ == "__main__":
    main()
