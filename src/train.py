"""Train and evaluate one CNN or LSTM configuration.

Usage (single run):
    python src/train.py --model cnn --field output
    python src/train.py --model lstm --field both --transform no_markdown
    python src/train.py --model cnn --field output --held-out code     # RQ3

Each run writes results/runs/<name>.json with test metrics, the confusion matrix, and the
per-epoch training curve.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from models import MODELS
from text import PAD, TRANSFORMS, Vocab, encode_batch

DATA = "data/dataset.parquet"
RUNS = "results/runs"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = {"input": 256, "output": 512}
MAX_LEN_BOTH = {"input": 128, "output": 512}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_splits(held_out=None):
    """Return (train, val, test, test_ood). test_ood holds every row of the held-out domain."""
    df = pd.read_parquet(DATA)
    if held_out is None:
        return (*(df[df.split == s] for s in ("train", "val", "test")), None)
    seen = df[df.domain != held_out]
    return (*(seen[seen.split == s] for s in ("train", "val", "test")), df[df.domain == held_out])


def make_batches(encoded, labels, batch_size, shuffle):
    """Group examples of similar length so the LSTM wastes little time on padding."""
    idx = np.argsort([len(e) for e in encoded], kind="stable")
    batches = [idx[i: i + batch_size] for i in range(0, len(idx), batch_size)]
    if shuffle:
        random.shuffle(batches)
    for b in batches:
        lengths = torch.tensor([len(encoded[i]) for i in b])
        x = torch.full((len(b), int(lengths.max())), PAD, dtype=torch.long)
        for j, i in enumerate(b):
            x[j, : len(encoded[i])] = torch.tensor(encoded[i])
        yield x, lengths, torch.tensor(labels[b]), b


def predict(model, encoded, labels, batch_size=128):
    model.eval()
    preds = np.empty(len(encoded), dtype=int)
    with torch.no_grad():
        for x, lengths, _, b in make_batches(encoded, labels, batch_size, shuffle=False):
            preds[b] = model(x.to(DEVICE), lengths).argmax(1).cpu().numpy()
    return preds


def scores(y, p):
    return {"acc": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro")}


def run(model="cnn", field="output", transform="none", held_out=None, seed=0, epochs=10,
        patience=3, lr=1e-3, batch_size=64, name=None, overwrite=False):
    name = name or f"{model}_{field}_{transform}" + (f"_ood-{held_out}" if held_out else "") + f"_s{seed}"
    path = f"{RUNS}/{name}.json"
    if os.path.exists(path) and not overwrite:
        print(f"[skip] {name} (exists)")
        with open(path) as f:
            return json.load(f)

    set_seed(seed)
    train, val, test, test_ood = load_splits(held_out)
    classes = sorted(train.family.unique())
    cid = {c: i for i, c in enumerate(classes)}
    tf = TRANSFORMS[transform]
    max_len = MAX_LEN_BOTH if field == "both" else MAX_LEN

    texts = []
    if field in ("input", "both"):
        texts += [tf(t) for t in train.input]
    if field in ("output", "both"):
        texts += [tf(t) for t in train.output]
    vocab = Vocab(texts)

    def enc(df):
        ids, lengths = encode_batch(vocab, df.to_dict("records"), field, max_len, tf)
        return [row[:n].tolist() for row, n in zip(ids, lengths)], df.family.map(cid).to_numpy()

    (Xtr, ytr), (Xva, yva), (Xte, yte) = enc(train), enc(val), enc(test)

    net = MODELS[model](len(vocab), len(classes)).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best, best_state, bad, curve = -1, None, 0, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        net.train()
        total, n = 0.0, 0
        for x, lengths, y, _ in make_batches(Xtr, ytr, batch_size, shuffle=True):
            opt.zero_grad()
            loss = loss_fn(net(x.to(DEVICE), lengths), y.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            total, n = total + loss.item() * len(y), n + len(y)
        va = scores(yva, predict(net, Xva, yva))
        curve.append({"epoch": ep, "train_loss": total / n, **{f"val_{k}": v for k, v in va.items()}})
        print(f"[{name}] ep{ep} loss={total / n:.3f} val_acc={va['acc']:.3f} val_f1={va['macro_f1']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if va["macro_f1"] > best:
            best, bad = va["macro_f1"], 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)

    pte = predict(net, Xte, yte)
    result = {
        "name": name, "model": model, "field": field, "transform": transform, "held_out": held_out,
        "seed": seed, "classes": classes, "vocab_size": len(vocab),
        "n_params": sum(p.numel() for p in net.parameters()),
        "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte),
        "epochs_run": len(curve), "best_val_f1": best, "train_seconds": time.time() - t0,
        "curve": curve, "test": scores(yte, pte),
        "test_per_class_f1": dict(zip(classes, f1_score(yte, pte, average=None).tolist())),
        "confusion": confusion_matrix(yte, pte).tolist(),
    }
    if test_ood is not None:
        Xo, yo = enc(test_ood)
        po = predict(net, Xo, yo)
        result.update({"n_test_ood": len(Xo), "test_ood": scores(yo, po),
                       "test_ood_per_class_f1": dict(zip(classes, f1_score(yo, po, average=None).tolist())),
                       "confusion_ood": confusion_matrix(yo, po).tolist()})
    os.makedirs(RUNS, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[{name}] TEST acc={result['test']['acc']:.3f} f1={result['test']['macro_f1']:.3f}"
          + (f" | OOD acc={result['test_ood']['acc']:.3f} f1={result['test_ood']['macro_f1']:.3f}"
             if test_ood is not None else ""), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="cnn")
    ap.add_argument("--field", choices=["input", "output", "both"], default="output")
    ap.add_argument("--transform", choices=list(TRANSFORMS), default="none")
    ap.add_argument("--held-out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    run(a.model, a.field, a.transform, a.held_out, a.seed, a.epochs, overwrite=a.overwrite)
