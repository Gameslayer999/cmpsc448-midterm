"""Tokenization, vocabulary, and text transforms used for ablations.

The tokenizer keeps case, punctuation, markdown symbols, and newlines as tokens, because
formatting is part of what may distinguish one LLM from another.
"""
import re
from collections import Counter

import torch

TOKEN_RE = re.compile(r"\w+|\n|[^\w\s]")
PAD, UNK, SEP = 0, 1, 2
SPECIALS = ["<pad>", "<unk>", "<sep>"]


def tokenize(text):
    return TOKEN_RE.findall(text)


class Vocab:
    def __init__(self, texts, max_size=30000, min_freq=2):
        counts = Counter(t for text in texts for t in tokenize(text))
        words = [w for w, c in counts.most_common(max_size) if c >= min_freq]
        self.itos = SPECIALS + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text, max_len):
        return [self.stoi.get(t, UNK) for t in tokenize(text)[:max_len]]


# ---- Ablation transforms (RQ4): each removes one family of signals from the text. ----

MD_RE = re.compile(r"^\s{0,3}#{1,6}\s*|^\s*[-*+]\s+|^\s*\d+[.)]\s+|\*\*|__|`{1,3}|^\s*[-*_]{3,}\s*$|^>\s?",
                   re.MULTILINE)


def strip_markdown(text):
    """Remove headers, bullets, numbering, bold, code fences, rules; collapse all whitespace."""
    return re.sub(r"\s+", " ", MD_RE.sub("", text)).strip()


def words_only(text):
    """Lowercase alphanumeric words only: no punctuation, formatting, emoji, or line structure."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


TRANSFORMS = {
    "none": lambda t: t,
    "no_markdown": strip_markdown,
    "words_only": words_only,
}


def encode_batch(vocab, examples, field, max_len, transform):
    """examples: list of dicts with 'input' and 'output'. field: input | output | both."""
    seqs = []
    for ex in examples:
        if field == "input":
            ids = vocab.encode(transform(ex["input"]), max_len["input"])
        elif field == "output":
            ids = vocab.encode(transform(ex["output"]), max_len["output"])
        else:
            ids = (vocab.encode(transform(ex["input"]), max_len["input"]) + [SEP]
                   + vocab.encode(transform(ex["output"]), max_len["output"]))
        seqs.append(ids or [UNK])
    lengths = torch.tensor([len(s) for s in seqs])
    batch = torch.full((len(seqs), int(lengths.max())), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        batch[i, : len(s)] = torch.tensor(s)
    return batch, lengths
