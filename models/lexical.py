"""Lexical head: P(turn complete | trailing ASR tokens), a hashed logistic regression.

Input is the transcript visible at time t (finals + latest partial, as they had arrived),
script-normalised by core.text. Features are hashed into 2^14 buckets with crc32 (stable
across processes, unlike Python's salted hash()):

    last 1/2/3 tokens by position, last bigram, lexical class of last 1 and last 2 tokens,
    tag-question flag, token-count bucket.

Inference is one sparse dot product (~10 us). Weights live in an .npz next to the ONNX head.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.text import is_tag_question, normalise, trailing_classes

DIM = 1 << 14


def lexical_keys(tokens: list[str], drop_classes: frozenset[str] = frozenset()) -> list[str]:
    cls = [c if c not in drop_classes else "OTHER" for c in trailing_classes(tokens, 2)]
    tail = tokens[-3:][::-1]
    keys = [f"u{i + 1}:{t}" for i, t in enumerate(tail)]
    if len(tail) >= 2:
        keys.append(f"b:{tail[1]}_{tail[0]}")
    keys += [f"c1:{cls[0]}", f"c12:{cls[1]}_{cls[0]}", f"n:{min(len(tokens), 12) // 3}"]
    if is_tag_question(tokens) and "TAG" not in drop_classes:
        keys.append("tagq")
    keys.append("bias")
    return keys


def hash_keys(keys: list[str]) -> np.ndarray:
    return np.array([zlib.crc32(k.encode("utf-8")) % DIM for k in keys], dtype=np.int64)


@dataclass
class LexicalHead:
    w: np.ndarray
    drop_classes: frozenset[str] = frozenset()

    def logit(self, text: str) -> float:
        toks = normalise(text)
        if not toks:
            return float("nan")
        return float(self.w[hash_keys(lexical_keys(toks, self.drop_classes))].sum())

    def save(self, path: Path) -> None:
        np.savez_compressed(path, w=self.w.astype(np.float32), drop=np.array(sorted(self.drop_classes)))

    @classmethod
    def load(cls, path: Path) -> "LexicalHead":
        d = np.load(path)
        return cls(d["w"].astype(np.float64), frozenset(str(x) for x in d["drop"]))


def design(texts: list[str], drop_classes: frozenset[str] = frozenset()):
    """Sparse design matrix for sklearn."""
    from scipy.sparse import csr_matrix

    rows, cols = [], []
    for i, t in enumerate(texts):
        idx = np.unique(hash_keys(lexical_keys(normalise(t), drop_classes)))
        rows += [i] * len(idx)
        cols += idx.tolist()
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(texts), DIM))


def fit(texts: list[str], y: np.ndarray, drop_classes: frozenset[str] = frozenset(), c: float = 0.5) -> LexicalHead:
    from sklearn.linear_model import LogisticRegression

    X = design(texts, drop_classes)
    m = LogisticRegression(C=c, max_iter=2000, fit_intercept=False)
    m.fit(X, y)
    return LexicalHead(m.coef_.ravel().astype(np.float64), drop_classes)


def oof_logits(texts: list[str], y: np.ndarray, groups: np.ndarray, drop_classes: frozenset[str] = frozenset(), k: int = 5) -> np.ndarray:
    """Out-of-fold logits, so the fusion layer never sees in-sample lexical scores."""
    from sklearn.model_selection import GroupKFold

    out = np.full(len(texts), np.nan)
    for tr, te in GroupKFold(k).split(texts, y, groups):
        head = fit([texts[i] for i in tr], y[tr], drop_classes)
        out[te] = [head.logit(texts[i]) for i in te]
    return out
