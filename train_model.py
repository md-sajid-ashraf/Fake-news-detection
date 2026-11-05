"""Rebuild model.pkl from the five datasets (same pipeline as the notebook).

Usage:  python train_model.py
Label convention everywhere:  1 = real/true news,  0 = fake news.
"""
import os
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import wordpre

BASE = os.path.dirname(os.path.abspath(__file__))


def p(*parts):
    return os.path.join(BASE, *parts)


def join_text(*cols):
    """Concatenate columns WITH a space (the notebook glued words together)."""
    out = cols[0].fillna("").astype(str)
    for c in cols[1:]:
        out = out + " " + c.fillna("").astype(str)
    return out.str.strip()


def load_datasets():
    frames = []

    # 1) news.csv  (REAL / FAKE)
    d = pd.read_csv(p("Data_set", "news", "news.csv"))
    d["Article"] = join_text(d["title"], d["text"])
    d["label"] = d["label"].map({"REAL": 1, "FAKE": 0})
    frames.append(d[["Article", "label"]])

    # 2) True.csv / Fake.csv
    t = pd.read_csv(p("Dataset", "True.csv")); t["label"] = 1
    f = pd.read_csv(p("Dataset", "Fake.csv")); f["label"] = 0
    d = pd.concat([t, f])
    d["Article"] = join_text(d["title"], d["text"])
    frames.append(d[["Article", "label"]])

    # 3) politifact (titles only)
    r = pd.read_csv(p("politifact", "politifact_real.csv")); r["label"] = 1
    f = pd.read_csv(p("politifact", "politifact_fake.csv")); f["label"] = 0
    d = pd.concat([r, f])
    d["Article"] = join_text(d["title"])
    frames.append(d[["Article", "label"]])

    # 4) Kaggle train.csv  -- in this dataset 1 = UNRELIABLE (fake), 0 = reliable.
    #    The original notebook treated 1 as real, which inverted ~20k labels.
    d = pd.read_csv(p("Data-set", "train.csv"))
    d["Article"] = join_text(d["title"], d["text"])
    d["label"] = 1 - d["label"]
    frames.append(d[["Article", "label"]])

    # 5) data.csv  (Label: 1 = real, 0 = fake)
    d = pd.read_csv(p("data.csv"))
    d["Article"] = join_text(d["Headline"], d["Body"])
    d["label"] = d["Label"]
    frames.append(d[["Article", "label"]])

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["label"])
    data["label"] = data["label"].astype(int)
    data = data[data["Article"].str.len() > 0]
    return data.sample(frac=1, random_state=2020).reset_index(drop=True)  # shuffle


def main():
    data = load_datasets()
    print("combined shape:", data.shape, "| label counts:", data["label"].value_counts().to_dict())
    data["Article"] = data["Article"].apply(wordpre)

    x_train, x_test, y_train, y_test = train_test_split(
        data["Article"], data["label"], test_size=0.2, random_state=2020, stratify=data["label"])

    pipe = Pipeline([
        ("vect", CountVectorizer(min_df=2)),
        ("tfidf", TfidfTransformer()),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test)
    print("accuracy: {:.2f}%".format(accuracy_score(y_test, pred) * 100))
    print("confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_test, pred))

    out = p("model.pkl")
    joblib.dump(pipe, out, compress=3)
    print("saved ->", out)


if __name__ == "__main__":
    sys.exit(main())
