from typing import List, Tuple
from datasets import load_dataset
import numpy as np

def load_ag_news(corpus_size=1500, query_size=300, seed=123) -> Tuple[List[str], List[str]]:
    ds = load_dataset("ag_news")
    texts = [x["text"] for x in ds["train"]]
    rng = np.random.default_rng(seed); idx = rng.permutation(len(texts))
    corpus = [texts[i] for i in idx[:corpus_size]]
    queries = [texts[i] for i in idx[corpus_size:corpus_size+query_size]]
    return corpus, queries

def load_pubmed_rct(corpus_size=1500, query_size=300, seed=123) -> Tuple[List[str], List[str]]:
    ds = load_dataset("pubmed_rct", "pubmed_rct")
    texts = [x["abstract"] for x in ds["train"] if x.get("abstract")]
    rng = np.random.default_rng(seed); idx = rng.permutation(len(texts))
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
