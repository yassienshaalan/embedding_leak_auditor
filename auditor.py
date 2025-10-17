import os, numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from utils import ensure_dir, cosine_sim_matrix, add_gaussian_noise, quantize, topk_indices, mrr_at_k, hit_at_k, safe_write_csv

class EmbeddingBackend:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
        self.model = SentenceTransformer(model_name, device=device)
    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=False).astype('float32')

class LeakAuditor:
    def __init__(self, save_dir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device=None):
        ensure_dir(save_dir)
        self.save_dir = save_dir
        self.backend = EmbeddingBackend(model_name, device=device)

    def embed(self, corpus: List[str], queries: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        return self.backend.encode(corpus), self.backend.encode(queries)

    def membership_metrics(self, corpus_emb: np.ndarray, queries_emb: np.ndarray, labels_present: List[bool], ks=(1,5,10)) -> Tuple[Dict[str,float], np.ndarray]:
        sim = cosine_sim_matrix(queries_emb, corpus_emb)
        # ranks for "present" queries via cosine threshold (proxy for exact duplicates)
        ranks = []
        for i in range(sim.shape[0]):
            if labels_present[i]:
                row = sim[i]; order = np.argsort(-row)
                pos = 0
                for j, idx in enumerate(order[:100]):
                    if row[idx] > 0.98:  # tight threshold
                        pos = j+1; break
                ranks.append(pos)
            else:
                ranks.append(0)
        metrics = {f"hit@{k}": hit_at_k(np.array(ranks), k) for k in ks}
        metrics["mrr@10"] = mrr_at_k(np.array(ranks), 10)
        return metrics, sim

    def noise_sweep(self, corpus_emb, queries_emb, labels_present, sigmas=(0.0,0.01,0.03,0.05), qbits=(None,8)) -> List[Dict[str,Any]]:
        rows = []
        for s in sigmas:
            for qb in qbits:
                C, Q = corpus_emb.copy(), queries_emb.copy()
                if s>0: C, Q = add_gaussian_noise(C, s), add_gaussian_noise(Q, s)
                if qb is not None: C, Q = quantize(C, qb), quantize(Q, qb)
                m,_ = self.membership_metrics(C,Q,labels_present)
                rows.append({"sigma":s,"qbits":qb, **m})
        return rows

    def cosine_hist_plot(self, sim: np.ndarray, labels_present: list, out_path: str):
        import numpy as np
        import matplotlib.pyplot as plt

        max_cos = sim.max(axis=1)
        present = np.array([max_cos[i] for i, flag in enumerate(labels_present) if flag], dtype=float)
        absent  = np.array([max_cos[i] for i, flag in enumerate(labels_present) if not flag], dtype=float)

        plt.figure(figsize=(6, 4))

        def plot_group(data: np.ndarray, label: str):
            if data.size == 0:
                return
            lo = float(np.nanmin(data))
            hi = float(np.nanmax(data))
            # Guard against zero or tiny range
            if not np.isfinite(lo) or not np.isfinite(hi):
                return
            if hi - lo < 1e-6:
                pad = 1e-3
                lo -= pad
                hi += pad
            # Choose a sensible bin count for small samples
            bins = max(1, min(30, int(np.ceil(np.sqrt(data.size)))))
            plt.hist(data, bins=bins, range=(lo, hi), alpha=0.6, density=True, label=label)

        plot_group(present, "Present")
        plot_group(absent, "Absent")

        plt.xlabel("Max cosine to corpus")
        plt.ylabel("Density")
        plt.title("Cosine separation")
        plt.legend()
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close()


    def line_plot(self, xs, ys_dict: Dict[str, List[float]], xlabel, ylabel, title, out_path):
        plt.figure(figsize=(6,4))
        for name, ys in ys_dict.items():
            plt.plot(xs, ys, marker="o", label=name)
        plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title); plt.legend()
        plt.savefig(out_path, bbox_inches="tight", dpi=200); plt.close()

    def inversion_light(self, corpus_emb, queries_emb, corpus_texts, queries_texts) -> List[Dict[str,Any]]:
        sim = cosine_sim_matrix(queries_emb, corpus_emb)
        top1, _ = topk_indices(sim, k=1)
        rows = []
        cc = SmoothingFunction()
        for i, idx in enumerate(top1.squeeze(1)):
            recon = corpus_texts[int(idx)]
            ref = queries_texts[i]
            bleu = sentence_bleu([ref.split()], recon.split(), weights=(1.0,0,0,0), smoothing_function=cc.method1)
            rows.append({"i": i, "reconstruction": recon[:200], "bleu1": float(bleu)})
        return rows
