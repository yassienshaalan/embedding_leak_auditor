# Embedding Leak Auditor (ELA)

This runs an embedding leakage audit on:
- **AG News** (`ag_news`)
- **PubMed RCT** (`pubmed_rct`)
- **Financial Phrasebank** (`financial_phrasebank`, subset `sentences_allagree`)

It measures **membership (retrieval)** under noise/quantization and a **lightweight inversion**.

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python - <<'PY'
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
PY

# Run all datasets
python main.py --all

# Or run one
python main.py --dataset ag_news
```
Results go to `results/<dataset>/` with CSVs and PNG plots.
