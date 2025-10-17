import argparse, os, numpy as np
from utils import ensure_dir, safe_write_csv, cosine_sim_matrix, topk_indices
from auditor import LeakAuditor

from ela_datasets import (
    load_ag_news,
    load_pubmed_rct,
    load_financial_phrasebank,
    external_queries_for,
)

def get_loader(name):
    return {
        "ag_news": load_ag_news,
        "pubmed_rct": load_pubmed_rct,
        "financial_phrasebank": load_financial_phrasebank,
    }[name]

def run_one(name: str, args):
    out_dir = os.path.join("results", name); ensure_dir(out_dir)
    loader = get_loader(name)
    corpus, queries = loader(corpus_size=args.corpus_size, query_size=args.query_size, seed=args.seed)
    if len(corpus) == 0 or len(queries) == 0:
        raise RuntimeError(f"{name}: got empty corpus ({len(corpus)}) or queries ({len(queries)}). "
                       f"Try smaller sizes or check dataset availability.")

    # make half the queries exact duplicates for membership testing
    dup_n = min(len(queries)//2, len(corpus)//4)
    queries[:dup_n] = corpus[:dup_n]
    labels_present = [i < dup_n for i in range(len(queries))]

    auditor = LeakAuditor(save_dir=out_dir, model_name=args.model_name)
    corpus_emb, queries_emb = auditor.embed(corpus, queries)

    # Base metrics + histogram
    metrics, sim = auditor.membership_metrics(corpus_emb, queries_emb, labels_present)
    safe_write_csv(os.path.join(out_dir, "metrics_membership_base.csv"), [metrics])
    auditor.cosine_hist_plot(sim, labels_present, os.path.join(out_dir, "cosine_hist_present_absent.png"))

    # Noise/quantization sweep
    sweep = auditor.noise_sweep(corpus_emb, queries_emb, labels_present, sigmas=(0.0,0.01,0.03,0.05), qbits=(None,8))
    safe_write_csv(os.path.join(out_dir, "metrics_membership_sweep.csv"), sweep)
    sigmas = sorted(set([r["sigma"] for r in sweep]))
    hit1 = [np.mean([r["hit@1"] for r in sweep if r["sigma"]==s]) for s in sigmas]
    mrr  = [np.mean([r["mrr@10"] for r in sweep if r["sigma"]==s]) for s in sigmas]
    auditor.line_plot(sigmas, {"Hit@1":hit1, "MRR@10":mrr}, "Noise σ", "Score", f"{name}: Membership vs Noise", os.path.join(out_dir, "hit_mrr_vs_noise.png"))

    # Inversion (lightweight)
    inv_rows = auditor.inversion_light(corpus_emb, queries_emb, corpus, queries)
    safe_write_csv(os.path.join(out_dir, "metrics_inversion.csv"), inv_rows)

    # Save neighbors top-5
    topk, scores = topk_indices(sim, k=5)
    rows = []
    for i in range(topk.shape[0]):
        for j in range(5):
            idx = int(topk[i,j])
            rows.append({"query_id":i, "rank":j+1, "cosine":float(scores[i,j]), "neighbor_idx":idx, "neighbor_text": corpus[idx][:300]})
    safe_write_csv(os.path.join(out_dir, "neighbors_top5.csv"), rows)

    # External queries
    ex = external_queries_for(name)
    if ex:
        ex_emb = auditor.backend.encode(ex)
        ex_sim = cosine_sim_matrix(ex_emb, corpus_emb)
        ex_topk, ex_scores = topk_indices(ex_sim, k=5)
        rows = []
        for i, q in enumerate(ex):
            for j in range(5):
                idx = int(ex_topk[i,j])
                rows.append({"external_query": q, "rank": j+1, "cosine": float(ex_scores[i,j]), "neighbor_text": corpus[idx][:300]})
        safe_write_csv(os.path.join(out_dir, "external_queries_neighbors.csv"), rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ag_news","pubmed_rct","financial_phrasebank"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--corpus-size", type=int, default=1500)
    ap.add_argument("--query-size", type=int, default=300)
    ap.add_argument("--model-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    if args.all:
        for name in ["ag_news","pubmed_rct","financial_phrasebank"]:
            run_one(name, args)
    else:
        if not args.dataset:
            ap.error("Specify --dataset or --all")
        run_one(args.dataset, args)

if __name__ == "__main__":
    main()
