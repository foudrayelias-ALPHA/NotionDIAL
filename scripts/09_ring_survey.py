"""Step-2 ring survey: do days-of-week / months form circles in a small real LM,
and where (embedding vs layer)?

For each candidate model and each layer: collect concept-token representations
(bare embedding rows + mean residual-stream states over template contexts), then score
ring-ness with the same instruments validated on the clock:
  - top-2 PCA variance share of the (n_concept, d) cloud
  - DFT power fraction at k=1 (the fundamental for a cyclic concept)
  - winding number of the ordered polygon in the top-2 PCA plane
  - shift-eigen test: T = lstsq(X, roll(X)); eigenvalue angles should hit 2*pi*k/n

Usage: python 09_ring_survey.py --model gpt2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clocklib.fourier import freq_power
from clocklib.geometry import winding_number

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
TEMPLATES = [
    "The meeting is scheduled for {}.",
    "She will arrive on {}.",
    "Everything closed last {} evening.",
    "I always go swimming on {}.",
    "The deadline is next {}.",
    "It happened one {} morning.",
    "We usually rest on {}.",
    "The store reopens on {}.",
]


def ring_scores(X: np.ndarray) -> dict:
    """X: (n, d) ordered concept cloud."""
    n = X.shape[0]
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    top2 = float((S[:2] ** 2).sum() / (S**2).sum())
    xy = Xc @ Vt[:2].T
    pw = freq_power(Xc)
    k1 = float(pw[1] / pw.sum()) if pw.sum() > 0 else 0.0
    T, *_ = np.linalg.lstsq(Xc, np.roll(Xc, 1, axis=0), rcond=None)
    eig = np.linalg.eigvals(T)
    eig = eig[np.abs(np.abs(eig) - 1) < 0.35]
    angles = sorted({round(abs(float(np.angle(l))) * n / (2 * np.pi), 1)
                     for l in eig if abs(l.imag) > 1e-6})
    return {"top2_pca_share": top2, "k1_power_frac": k1,
            "winding_top2": winding_number(xy), "shift_eigen_freqs": angles}


@torch.no_grad()
def survey(model_name: str) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    out: dict = {"model": model_name, "concepts": {}}
    for cname, words in (("days", DAYS), ("months", MONTHS)):
        ids = []
        for w in words:
            t = tok.encode(" " + w)
            ids.append(t[0])
        ok_single = all(len(tok.encode(" " + w)) == 1 for w in words)

        emb = model.get_input_embeddings().weight[ids].float().numpy()
        layers: dict = {"embedding": ring_scores(emb)}

        n_layers = model.config.num_hidden_layers
        acc = [np.zeros((len(words), model.config.hidden_size)) for _ in range(n_layers + 1)]
        for tpl in TEMPLATES:
            for wi, w in enumerate(words):
                text = tpl.format(w)
                enc = tok(text, return_tensors="pt")
                pos = enc.input_ids[0].tolist().index(ids[wi])
                hs = model(**enc).hidden_states
                for li in range(n_layers + 1):
                    acc[li][wi] += hs[li][0, pos].float().numpy()
        for li in range(n_layers + 1):
            layers[f"resid_{li}"] = ring_scores(acc[li] / len(TEMPLATES))
        out["concepts"][cname] = {"single_token": ok_single, "token_ids": ids,
                                  "layers": layers}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    args = ap.parse_args()
    res = survey(args.model)
    out = ROOT / "artifacts" / f"ring_survey_{args.model.replace('/', '_')}.json"
    out.write_text(json.dumps(res, indent=2))
    for cname, c in res["concepts"].items():
        print(f"\n== {cname} (single_token={c['single_token']}) ==")
        for lname, s in c["layers"].items():
            print(f"  {lname:>12}: top2={s['top2_pca_share']:.2f} k1={s['k1_power_frac']:.2f} "
                  f"wind={s['winding_top2']:+.1f} shift_freqs={s['shift_eigen_freqs']}")


if __name__ == "__main__":
    main()
