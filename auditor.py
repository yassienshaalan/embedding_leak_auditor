import os, numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from utils import ensure_dir, cosine_sim_matrix, add_gaussian_noise, quantize, topk_indices, mrr_at_k, hit_at_k, safe_write_csv
from inversion import train_prefix_decoder, DEVICE

class EmbeddingBackend:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts, batch_size: int = 64):
        import numpy as np
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            dim = self.model.get_sentence_embedding_dimension()
            return np.zeros((0, dim), dtype="float32")
        emb = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                normalize_embeddings=False, show_progress_bar=False)
        emb = np.asarray(emb, dtype="float32")
        emb = np.atleast_2d(emb)
        return emb

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

    def cosine_hist_plot(self, sim, labels_present, out_path: str):
        import numpy as np
        import matplotlib.pyplot as plt

        # Handle empty similarity matrix safely
        if sim is None or sim.size == 0 or sim.shape[0] == 0:
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, "No queries to plot", ha="center", va="center", fontsize=12)
            plt.axis("off")
            plt.savefig(out_path, bbox_inches="tight", dpi=200)
            plt.close()
            return

        # Row-wise maxima (one per query)
        max_cos = sim.max(axis=1)

        present = np.array([max_cos[i] for i, flag in enumerate(labels_present) if flag], dtype=float)
        absent  = np.array([max_cos[i] for i, flag in enumerate(labels_present) if not flag], dtype=float)

        plt.figure(figsize=(6, 4))

        def plot_group(data: np.ndarray, label: str):
            if data.size == 0:
                return
            lo = float(np.nanmin(data))
            hi = float(np.nanmax(data))
            if not np.isfinite(lo) or not np.isfinite(hi):
                return
            if hi - lo < 1e-6:
                pad = 1e-3
                lo -= pad
                hi += pad
            bins = max(1, min(30, int(np.ceil(np.sqrt(max(1, data.size))))))
            plt.hist(data, bins=bins, range=(lo, hi), alpha=0.6, density=True, label=label)

        plot_group(present, "Present")
        plot_group(absent, "Absent")

        plt.xlabel("Max cosine to corpus")
        plt.ylabel("Density")
        plt.title("Cosine separation")
        if present.size or absent.size:
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
    def gpt2_inversion(self,
                   corpus_texts,
                   corpus_emb,
                   queries_texts,
                   queries_emb,
                   prefix_len: int = 64,
                   epochs: int = 8,
                   bs: int = 16,
                   lm_name: str = "gpt2",
                   max_len: int = 64,
                   train_cap: int = 1200,
                   logger=None):
        from inversion import train_prefix_decoder, DEVICE
        import numpy as np
        import torch
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from rouge_score import rouge_scorer
        import bert_score
        if logger: 
            logger.info(f"[INV] Train subset N={min(train_cap, len(corpus_texts))}, bs={bs}, max_len={max_len}")
        # Train on a capped subset (memory-friendly)
        N = min(train_cap, len(corpus_texts), len(corpus_emb))
        model = train_prefix_decoder(
            corpus_texts=corpus_texts[:N],
            corpus_vecs=corpus_emb[:N],
            lm_name=lm_name,
            embed_dim=corpus_emb.shape[1],
            prefix_len=prefix_len,
            epochs=epochs,
            bs=bs,
            max_len=max_len,
            train_cap=N,   # pass through to loader even if unused
        )

        # Evaluate on present queries (strongest leakage)
        def _norm(s): return " ".join(s.lower().split())
        bucket = {_norm(t): True for t in corpus_texts}
        present_idx = [i for i, q in enumerate(queries_texts) if _norm(q) in bucket]

        if present_idx:
            q_eval_vecs = queries_emb[present_idx]
            q_eval_texts = [queries_texts[i] for i in present_idx]
        else:
            q_eval_vecs = queries_emb[:min(64, len(queries_emb))]
            q_eval_texts = queries_texts[:min(64, len(queries_texts))]

        with torch.no_grad():
            preds = model.generate(
                torch.tensor(q_eval_vecs, dtype=torch.float32, device=DEVICE),
                max_len=max_len, min_len=12, temperature=0.9, top_k=40, top_p=0.9
            )

        # Metrics
        cc = SmoothingFunction()
        bleu = float(np.mean([
            sentence_bleu([r.split()], p.split(), weights=(1,0,0,0), smoothing_function=cc.method1)
            for p, r in zip(preds, q_eval_texts)
        ]))

        rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rougeL = float(np.mean([rs.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, q_eval_texts)]))

        P, R, F1 = bert_score.score(preds, q_eval_texts, lang="en", verbose=False)
        bertF1 = float(F1.mean().item())

        rows = [{"i": i, "ref": r, "gen": p} for i, (p, r) in enumerate(zip(preds, q_eval_texts))]
        return {"bleu": bleu, "rougeL": rougeL, "bertF1": bertF1, "examples": rows}