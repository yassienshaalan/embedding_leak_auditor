from typing import List, Tuple
from datasets import load_dataset
import numpy as np
from datasets import load_dataset as hf_load_dataset
def load_ag_news(corpus_size=1500, query_size=300, seed=123) -> Tuple[List[str], List[str]]:
    ds = load_dataset("ag_news")
    texts = [x["text"] for x in ds["train"]]
    rng = np.random.default_rng(seed); idx = rng.permutation(len(texts))
    corpus = [texts[i] for i in idx[:corpus_size]]
    queries = [texts[i] for i in idx[corpus_size:corpus_size+query_size]]
    return corpus, queries

def load_pubmed_rct(corpus_size=1500, query_size=300, seed=123) -> Tuple[List[str], List[str]]:
    """
    Compatibility loader: we keep the function name `load_pubmed_rct` so the rest
    of your pipeline/CLI doesn't change, but we use a biomedical dataset that is
    actually available on the Hugging Face Hub.

    Primary: pubmed_qa (pqa_labeled)  -> use 'context' or 'long_answer'
    Fallback: pubmed_qa (pqa_unlabeled)-> use 'context' or 'question'
    """
    try:
        ds = hf_load_dataset("pubmed_qa", "pqa_labeled")
        texts = []
        for ex in ds["train"]:
            # Prefer the longer field if present
            t = (ex.get("context") or ex.get("long_answer") or ex.get("question"))
            if t:
                texts.append(t)
        if len(texts) < (corpus_size + query_size):
            # Try validation and test splits to bulk up
            for split in ("validation", "test"):
                if split in ds:
                    for ex in ds[split]:
                        t = (ex.get("context") or ex.get("long_answer") or ex.get("question"))
                        if t:
                            texts.append(t)
    except Exception:
        # Fallback: unlabeled split
        ds = hf_load_dataset("pubmed_qa", "pqa_unlabeled")
        texts = []
        for ex in ds["train"]:
            t = (ex.get("context") or ex.get("question"))
            if t:
                texts.append(t)

    # Basic cleaning & sampling
    texts = [t.strip() for t in texts if t and len(t) > 20]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(texts))
    corpus = [texts[i] for i in idx[:corpus_size]]
    queries = [texts[i] for i in idx[corpus_size:corpus_size+query_size]]
    return corpus, queries

def load_financial_phrasebank(corpus_size=1500, query_size=300, seed=123) -> Tuple[List[str], List[str]]:
    ds = load_dataset("financial_phrasebank", "sentences_allagree")
    texts = [x["sentence"] for x in ds["train"]]
    rng = np.random.default_rng(seed); idx = rng.permutation(len(texts))
    corpus = [texts[i] for i in idx[:corpus_size]]
    queries = [texts[i] for i in idx[corpus_size:corpus_size+query_size]]
    return corpus, queries

def external_queries_for(name: str):
    if name == "ag_news":
        return [
            "Tax agency set to roll out a new compliance plan next quarter",
            "Central bank signals rate cuts amid slowing growth",
            "Tech giant unveils AI assistant to summarize legal contracts",
            "Wildfires trigger mass evacuations across coastal towns"
        ]
    if name == "pubmed_rct":
        return [
            "Randomized trial shows treatment reduces migraine frequency",
            "Adverse events include nausea and dizziness in elderly cohort",
            "Patients with diabetes show improved HbA1c after therapy",
            "Double-blind study reports no significant difference in survival"
        ]
    if name == "financial_phrasebank":
        return [
            "Quarterly revenue rose 14% while operating margin narrowed",
            "Credit risk increases as loan delinquencies climb",
            "Analysts maintain buy rating despite soft guidance",
            "Share repurchase program offsets earnings volatility"
        ]
    return []
