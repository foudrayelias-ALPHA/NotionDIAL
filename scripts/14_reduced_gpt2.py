"""Build a vocabulary-reduced GPT-2 for the decomposition-localization test.

Keep only the tokens appearing in a fixed corpus (templates + wikitext slice, all
day/month tokens guaranteed). Transformer blocks are untouched and the kept wte rows
are copied verbatim, so logits over kept tokens are EXACTLY the full model's logits
restricted to the kept set (verified below). This makes full-SPD stochastic masking
satisfiable at C < rank on local hardware.

Outputs: artifacts/reduced_gpt2/{model.pt, meta.json, corpus_ids.pt}
"""

import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module

pipeline = import_module("12_ring_pipeline")
from clocklib.ringlib import DAYS, MONTHS

OUT = ROOT / "artifacts" / "reduced_gpt2"
TARGET_VOCAB = 3000


def main() -> None:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    full = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32).eval()

    tpl = pipeline.template_text()
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    wiki_text = "\n".join(t for t in wiki["text"] if t.strip())

    tpl_ids = tok(tpl, return_tensors="pt").input_ids[0]
    concept_ids = {tok.encode(" " + w)[0] for w in DAYS.words + MONTHS.words}
    lo, hi = 0, len(wiki_text)
    while hi - lo > 10_000:  # bisect a wikitext prefix hitting ~TARGET_VOCAB uniques
        mid = (lo + hi) // 2
        ids = tok(wiki_text[:mid], return_tensors="pt").input_ids[0]
        n = len(set(ids.tolist()) | set(tpl_ids.tolist()) | concept_ids)
        lo, hi = (mid, hi) if n < TARGET_VOCAB else (lo, mid)
    wiki_ids = tok(wiki_text[:lo], return_tensors="pt").input_ids[0]

    prompt_ids: set[int] = set()
    for c in (DAYS, MONTHS):
        for w in c.words:
            for t in c.templates + [c.successor_prompt]:
                prompt_ids |= set(tok.encode(t.format(w)))
    kept = sorted(set(wiki_ids.tolist()) | set(tpl_ids.tolist()) | concept_ids | prompt_ids)
    old2new = {o: n for n, o in enumerate(kept)}
    n_keep = len(kept)
    print(f"kept vocab: {n_keep}")

    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32).eval()
    W = full.get_input_embeddings().weight.detach()
    new_wte = nn.Embedding(n_keep, W.shape[1])
    with torch.no_grad():
        new_wte.weight.copy_(W[kept])
    model.transformer.wte = new_wte
    new_head = nn.Linear(W.shape[1], n_keep, bias=False)
    with torch.no_grad():
        new_head.weight.copy_(W[kept])
    model.lm_head = new_head
    model.config.vocab_size = n_keep
    model.eval().requires_grad_(False)

    day_new = [old2new[tok.encode(" " + w)[0]] for w in DAYS.words]
    month_new = [old2new[tok.encode(" " + w)[0]] for w in MONTHS.words]

    # sanity: restricted logits must match the full model exactly
    day_old = [tok.encode(" " + w)[0] for w in DAYS.words]
    enc = tok("If today is Friday, then tomorrow is", return_tensors="pt")
    lf = full(**enc).logits[0, -1][day_old]
    enc_r = torch.tensor([[old2new[i] for i in enc.input_ids[0].tolist()]])
    lr = model(input_ids=enc_r).logits[0, -1][day_new]
    assert torch.allclose(lf, lr, atol=1e-4), (lf - lr).abs().max()

    corpus = torch.cat([tpl_ids, wiki_ids])
    corpus = torch.tensor([old2new[i] for i in corpus.tolist()])
    OUT.mkdir(exist_ok=True)
    torch.save(model.state_dict(), OUT / "model.pt")
    torch.save(corpus, OUT / "corpus_ids.pt")
    (OUT / "meta.json").write_text(json.dumps(
        {"n_keep": n_keep, "kept": kept, "day_ids": day_new, "month_ids": month_new,
         "tpl_len": int(len(tpl_ids)), "sanity_max_diff": float((lf - lr).abs().max())},
        indent=2))
    print(f"sanity max logit diff: {(lf - lr).abs().max():.2e} -> {OUT}")


if __name__ == "__main__":
    main()
