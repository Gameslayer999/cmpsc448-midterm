"""Non-neural reference point: TF-IDF features + logistic regression, on the same splits as the
CNN/LSTM runs, for input only / output only / input + output.

Writes results/baseline.json.

Usage: python src/baseline.py
"""
import json

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from text import TOKEN_RE


def vectorizer():
    # Same tokenizer as the neural models, so case, punctuation and markdown count as features.
    return TfidfVectorizer(token_pattern=None, tokenizer=TOKEN_RE.findall, lowercase=False,
                           ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True)


def main():
    df = pd.read_parquet("data/dataset.parquet")
    train, test = df[df.split == "train"], df[df.split == "test"]
    results = {}
    for field in ("input", "output", "both"):
        parts = ["input", "output"] if field == "both" else [field]
        vecs = {p: vectorizer().fit(train[p]) for p in parts}
        Xtr = hstack([vecs[p].transform(train[p]) for p in parts]).tocsr()
        Xte = hstack([vecs[p].transform(test[p]) for p in parts]).tocsr()
        clf = LogisticRegression(C=4.0, max_iter=2000).fit(Xtr, train.family)
        pred = clf.predict(Xte)
        results[field] = {"acc": accuracy_score(test.family, pred),
                          "macro_f1": f1_score(test.family, pred, average="macro")}
        print(field, {k: round(v, 3) for k, v in results[field].items()}, flush=True)
        if field == "output":
            # The most family-specific n-grams: highest-weight features per class (RQ4).
            names = np.array(vecs["output"].get_feature_names_out())
            results["top_output_features"] = {
                c: names[np.argsort(clf.coef_[i])[::-1][:15]].tolist() for i, c in enumerate(clf.classes_)}
    with open("results/baseline.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
