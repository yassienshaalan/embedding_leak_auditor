# Embedding Leak Auditor (ELA)

> **Quantifying privacy risk in embedding-based AI systems.**  
> The **Embedding Leak Auditor (ELA)** measures how much private or sensitive information can be inferred from text embeddings through *retrieval* and *inversion* attacks.

---

## Overview

Modern AI systems depend on embeddings, vector representations that encode the *meaning* of text.  
They power search, chatbots, clustering, and retrieval-augmented generation (RAG).  
Yet, embeddings also carry *latent memory* of their training data.

**ELA** provides a structured, empirical way to measure this memory, transforming abstract privacy concerns into actionable metrics.

**Core Capabilities:**
-  **Membership leakage**: Can someone tell if a text was part of your dataset?
-  **Semantic inversion**: Can embeddings be decoded back to readable text?
-  **Privacy–utility frontier**: How much can we perturb embeddings before breaking their usefulness?
-  **Cross-domain evaluation**: Works across news, biomedical, and financial text.
-  **Visualization**: Generates retrieval, inversion, and defense impact plots.

---

##  Repository Structure
```bash
 embedding_leak_auditor/
├── main.py # Entry point for running audits
├── auditor.py # Core leak auditing logic
├── inversion.py # GPT-2 prefix-tuned inversion module
├── ela_datasets.py # Dataset loaders (AG News, PubMed RCT, Financial Phrasebank)
├── utils.py # Shared utilities for metrics and plots
└── results/ # Experiment outputs: metrics, examples, plots, logs
```
---

##  Environment Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-org>/embedding-leak-auditor.git
cd embedding-leak-auditor
2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt


Tip: For inversion experiments, use a GPU-enabled environment (CUDA ≥ 11.8).
CPU-only runs work fine for retrieval-only experiments.

Running Experiments
Run all datasets (AG News, PubMed RCT, Financial Phrasebank)
python3 main.py --all \
  --enable-gpt2-inversion \
  --lm-name gpt2-medium \
  --inv-epochs 20 \
  --prefix-len 64 \
  --corpus-size 1500 \
  --query-size 300

Run a single dataset
python3 main.py \
  --dataset ag_news \
  --enable-gpt2-inversion \
  --lm-name gpt2 \
  --inv-epochs 8 \
  --prefix-len 48 \
  --corpus-size 800 \
  --query-size 200

Run retrieval-only audit (no GPT-2 inversion)
python3 main.py --dataset financial_phrasebank

Command-Line Flags
Flag	Description	Example
--dataset	Dataset to run (ag_news, pubmed_rct, financial_phrasebank)	--dataset ag_news
--sigma	Gaussian noise applied to embeddings	--sigma 0.05
--qbits	Quantization precision (0 = float32)	--qbits 8
--enable-gpt2-inversion	Enables GPT-2 prefix-tuned decoder	
--lm-name	Language model for inversion (gpt2, gpt2-medium)	--lm-name gpt2-medium
--inv-epochs	Inversion training epochs	--inv-epochs 10
--prefix-len	Prefix token length for inversion	--prefix-len 64
--inv-train-cap	Max number of samples for inversion training	--inv-train-cap 1000
--corpus-size	Corpus size for embeddings	--corpus-size 1500
--query-size	Query size for membership inference	--query-size 300

Output Structure
Each dataset run creates a structured results folder:

results/
├── ag_news/
│   ├── metrics_membership_base.csv
│   ├── metrics_membership_sweep.csv
│   ├── metrics_inversion.csv
│   ├── inversion_gpt2_examples.csv
│   ├── neighbors_top5.csv
│   ├── cosine_hist_present_absent.png
│   ├── hit_mrr_vs_noise.png
│   ├── logs.txt
│   └── ...
├── pubmed_rct/
│   └── ...
└── financial_phrasebank/
    └── ...


Files include:

 metrics_membership_base.csv – Base membership metrics (Hit@k, MRR)

 metrics_membership_sweep.csv – Noise/quantization sweep

 metrics_inversion.csv – BLEU, ROUGE-L, BERT-F1 for inversion

 inversion_gpt2_examples.csv – Real vs. reconstructed text samples

 neighbors_top5.csv – Nearest neighbor retrievals

 logs.txt – Full per-run experiment logs

 Example Interpretation

After a run, you’ll see summaries like:

[RETR] Hit@1(present)=1.00  Hit@5=1.00  MRR=1.00
[INV]  BLEU=0.13  ROUGE-L=0.13  BERT-F1=0.82
[GRADE] Estimated Privacy Risk: HIGH
```

This means:

 Exact duplicates are fully retrievable → strong membership signal.

 Generated text retains topic/intent → semantic leakage.

 Privacy posture: High risk without defenses.

Example Commands Summary
### Run everything (with inversion)
python3 main.py --all --enable-gpt2-inversion --lm-name gpt2-medium --inv-epochs 20

### Run retrieval-only
python3 main.py --dataset ag_news

### Run inversion with smaller corpus
python3 main.py --dataset financial_phrasebank --corpus-size 800 --query-size 200 --inv-epochs 10


Adding New Datasets:

To extend ELA, modify ela_datasets.py:
```bash
def load_new_dataset(corpus_size=1500, query_size=300, seed=1337):
    ds = load_dataset("myorg/new_corpus", split="train")
    texts = [t["text"] for t in ds]
    return random.sample(texts, corpus_size), random.sample(texts, query_size)


Then run:

python3 main.py --dataset new_dataset

```
Evaluating Defenses

ELA supports evaluating various embedding defenses:

Defense	Description	Example Effect
Gaussian Noise (σ)	Adds random perturbation to embeddings	Reduces membership recall
Quantization (qbits)	Reduces numerical precision	Slightly degrades utility
Rotation (planned)	Orthogonal transforms	Scrambles geometry per user
Differential Privacy (future)	Formal privacy guarantees	Strong protection, high cost

You can visualize this through the privacy–utility frontier curves automatically generated in each run.

##  Work in Progress: Toward a Plug-and-Play Library

The Embedding Leak Auditor (ELA) is currently provided as a research-grade toolkit with standalone scripts for running full leakage experiments.
We are actively working to refactor it into a lightweight, installable Python package that integrates easily into your ML workflow.

🔜 Coming Soon:
Install and start auditing in one line.
```bash
pip install eleak-auditor 
```
###  High-level API:
```bash
from leak_auditor import ELA

auditor = ELA(model="all-MiniLM-L6-v2")
report = auditor.run("ag_news")
report.summary()
```

### Modular components for:

*** Retrieval leak detection

*** Embedding inversion analysis

*** Privacy–utility frontier plots

*** Pluggable defenses (noise, quantization, rotation)

*** Seamless integration with Hugging Face, LangChain, and enterprise vector stores.

### Goal

To make privacy auditing as easy and standard as model evaluation — so teams can measure, visualize, and mitigate embedding leakage with minimal setup.

## Citation

Medium article:
https://medium.com/@yassien/embeddings-privacy-and-the-leak-auditor-auditing-the-hidden-memory-of-ai-2e7c78339ad9

 License

MIT License © 2025 Yassien Shaalan
Free for research and non-commercial use with attribution.

Acknowledgments

Developed as part of personal ongoing research into embedding security, governance, and generative model transparency.
Inspired by investigations into membership inference, representation learning, and generative privacy boundaries.

 If this work helps your research or product, please star the repository and share feedback.
