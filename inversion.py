# inversion.py
import math, re, time
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _clean_text(s: str) -> str:
    s = re.sub(r'(\s){2,}', r'\1', s)
    s = re.sub(r'([^\w\s])\1{2,}', r'\1\1', s)
    return s.strip()

class PrefixDecoder(nn.Module):
    """
    Simple prefix-tuning: map an embedding vector e (dim=D) to a sequence
    of P prefix token embeddings (shape [P, H]) that we concatenate to LM input.
    """
    def __init__(self, lm_name: str = "gpt2", embed_dim: int = 384, hidden: int = 768, prefix_len: int = 64):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(lm_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(lm_name)
        self.lm.resize_token_embeddings(len(self.tok))
        self.hidden = self.lm.config.hidden_size
        self.prefix_len = prefix_len

        # Project embedding -> prefix sequence (P*H), then tanh
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, self.hidden * prefix_len),
            nn.Tanh()
        )
        self.ln = nn.LayerNorm(self.hidden)

    def make_prefix(self, evecs: torch.Tensor) -> torch.Tensor:
        """
        evecs: [B, D] -> [B, P, H]
        """
        B, D = evecs.size()
        y = self.proj(evecs).view(B, self.prefix_len, self.hidden)
        y = self.ln(y)
        return y

    def forward(self, evecs: torch.Tensor, labels_ids: torch.Tensor):
        """
        Teacher-forced LM loss conditioned on prefix.
        labels_ids: token ids for target text (padded); returns CE loss.
        """
        B = evecs.size(0)
        prefix = self.make_prefix(evecs)                             # [B,P,H]
        tgt_emb = self.lm.transformer.wte(labels_ids)                # [B,L,H]
        inputs_embeds = torch.cat([prefix, tgt_emb[:, :-1, :]], 1)   # shift
        attn_mask = torch.ones(inputs_embeds.size()[:-1], device=evecs.device)
        out = self.lm(inputs_embeds=inputs_embeds, attention_mask=attn_mask)
        # compute loss on next-token prediction for tgt sequence only
        logits = out.logits[:, self.prefix_len-1:-1, :]              # align
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.tok.pad_token_id)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels_ids.reshape(-1))
        return loss

    @torch.no_grad()
    def generate(self,
                 evecs: torch.Tensor,
                 max_len: int = 64,
                 min_len: int = 12,
                 temperature: float = 0.9,
                 top_k: int = 40,
                 top_p: float = 0.90,
                 repetition_penalty: float = 1.2) -> List[str]:

        B = evecs.size(0)
        prefix = self.make_prefix(evecs)             # [B,P,H]
        # start from BOS (we'll use eos as bos for GPT-2)
        bos = torch.full((B, 1), self.tok.eos_token_id, dtype=torch.long, device=evecs.device)
        cur = self.lm.transformer.wte(bos)           # [B,1,H]
        seq_ids = [[] for _ in range(B)]
        last = None
        past = None

        # prime the cache with prefix
        out = self.lm(inputs_embeds=prefix, use_cache=True)
        past = out.past_key_values

        for t in range(max_len):
            out = self.lm(inputs_embeds=cur, use_cache=True, past_key_values=past)
            past = out.past_key_values
            logits = out.logits[:, -1, :] / max(1e-6, temperature)

            # repetition penalty
            if repetition_penalty != 1.0:
                for b in range(B):
                    for tok_id in seq_ids[b][-8:]:
                        logits[b, tok_id] /= repetition_penalty

            probs = torch.softmax(logits, dim=-1)

            # top-k / nucleus
            if top_k and top_k > 0:
                top_probs, top_idx = torch.topk(probs, k=min(top_k, probs.size(-1)))
                probs = torch.zeros_like(probs).scatter_(1, top_idx, top_probs)
            if 0 < top_p < 1:
                sp, si = torch.sort(probs, dim=-1, descending=True)
                c = torch.cumsum(sp, dim=-1)
                mask = c > top_p
                sp = torch.where(mask, torch.zeros_like(sp), sp)
                probs = torch.zeros_like(probs).scatter_(1, si, sp)
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

            nxt = torch.multinomial(probs, 1).squeeze(1)

            # avoid early EOS
            for b in range(B):
                if t < min_len and nxt[b].item() == self.tok.eos_token_id:
                    # pick next best
                    p = probs[b].clone()
                    p[self.tok.eos_token_id] = 0
                    p = p / (p.sum() + 1e-8)
                    nxt[b] = torch.multinomial(p, 1)

            for b in range(B):
                seq_ids[b].append(int(nxt[b]))

            cur = self.lm.transformer.wte(nxt).unsqueeze(1)

        outs = [self.tok.decode(row, skip_special_tokens=True) for row in seq_ids]
        return [_clean_text(x) for x in outs]

class InvDataset(Dataset):
    def __init__(self, texts: List[str], evecs, tokenizer, max_len=64):
        self.texts = texts
        self.evecs = evecs
        self.tok = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        t = self.texts[i]
        enc = self.tok(t, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
        return {
            "ids": enc.input_ids.squeeze(0),
            "e": torch.tensor(self.evecs[i], dtype=torch.float32)
        }

def train_prefix_decoder(
    corpus_texts, corpus_vecs,
    lm_name="gpt2",
    embed_dim=384,
    prefix_len=64,
    epochs=8,
    bs=16,                 # smaller default
    lr=1e-3,
    max_len=64,
    train_cap=1200         # NEW: limit training set
):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = PrefixDecoder(lm_name=lm_name, embed_dim=embed_dim, prefix_len=prefix_len)
    # reload LM in a memory-friendly way
    model.lm = AutoModelForCausalLM.from_pretrained(
        lm_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True
    )
    model.lm.resize_token_embeddings(len(model.tok))
    if torch.cuda.is_available():
        model.lm.to(DEVICE)
        model.lm.gradient_checkpointing_enable()  # reduce VRAM

    # cap training size
    N = min(train_cap, len(corpus_texts), len(corpus_vecs))
    ds = InvDataset(corpus_texts[:N], corpus_vecs[:N], model.tok, max_len=max_len)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)

    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    steps = epochs * max(1, len(dl))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=max(1, steps//20), num_training_steps=steps)

    model.train()
    for ep in range(epochs):
        tot = 0.0
        for batch in dl:
            ids = batch["ids"].to(DEVICE)
            e = batch["e"].to(DEVICE)
            loss = model(e, ids)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step()
            tot += float(loss.detach().cpu())
        print(f"[INV] epoch {ep+1}/{epochs}  loss={tot/len(dl):.4f}")
    model.eval()
    return model
