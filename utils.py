import os, numpy as np
from typing import List, Dict, Any, Tuple

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A.astype('float32'); B = B.astype('float32')
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T

def add_gaussian_noise(X: np.ndarray, sigma: float) -> np.ndarray:
    rng = np.random.default_rng(42)
    return X + rng.normal(0.0, sigma, size=X.shape).astype('float32')

def quantize(X: np.ndarray, bits: int = 8) -> np.ndarray:
    qmax = 2**(bits-1) - 1
    Y = np.empty_like(X)
    for i,v in enumerate(X):
        s = np.max(np.abs(v)) + 1e-12
        Y[i] = np.round(v / s * qmax) / qmax * s
    return Y

def topk_indices(sim: np.ndarray, k: int = 10):
    idx = np.argpartition(-sim, kth=k-1, axis=1)[:, :k]
    rows = np.arange(sim.shape[0])[:, None]
    order = np.argsort(-sim[rows, idx], axis=1)
    topk = idx[rows, order]
    scores = sim[rows, topk]
    return topk, scores

def mrr_at_k(ranks: np.ndarray, k: int) -> float:
    rr = []
    for r in ranks:
        if 0 < r <= k: rr.append(1.0/r)
        else: rr.append(0.0)
    return float(np.mean(rr)) if rr else 0.0

def hit_at_k(ranks: np.ndarray, k: int) -> float:
    hits = [1.0 if (0 < r <= k) else 0.0 for r in ranks]
    return float(np.mean(hits)) if hits else 0.0

def safe_write_csv(path: str, rows: List[Dict[str, Any]]):
    import csv
    if not rows: return
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow(r)
