"""SAE-basis correspondence (preregistration_sae.md, c9f1f5b).

Are the planes this program doses the same objects as the SAE features in which
the literature found circular geometry? GPT-2 layer-8 residual, public
GPT-2-small residual SAE (jbloom reformatted release).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from importlib import import_module

from clocklib.fourier import fourier_plane_basis
from clocklib.ringlib import DAYS, LM
from clocklib.unsup import array_pca, principal_cos

w18 = import_module("18_writers_any")

OUT = ROOT / "artifacts" / "sae_gpt2"
SAE_REPO = "jbloom/GPT2-Small-SAEs-Reformatted"
SAE_DIR = "blocks.8.hook_resid_pre"


def load_sae():
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    p = hf_hub_download(SAE_REPO, f"{SAE_DIR}/sae_weights.safetensors")
    w = load_file(p)
    return {k: v.float().numpy() for k, v in w.items()}


def main() -> None:
    lm = LM("gpt2", device="cpu")
    day_tpl = w18.positions(lm, DAYS)
    # day-token states at layer 8, template-averaged (same cloud as the writers run)
    H = np.zeros((7, lm.hidden_size))
    inst = []  # per-instance states for latent statistics
    with torch.no_grad():
        for enc, wi, pos in day_tpl:
            hs = lm.model(input_ids=enc, output_hidden_states=True).hidden_states
            v = hs[8][0, pos].float().numpy()
            H[wi] += v
            inst.append(v)
        # matched wikitext token sample for baseline activation
        from datasets import load_dataset

        wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                            split="validation")
        text = "\n".join(t for t in wiki["text"] if t.strip())[:5000]
        enc = lm.tok(text, return_tensors="pt").input_ids[:, :256]
        hsw = lm.model(input_ids=enc, output_hidden_states=True).hidden_states
        Wtok = hsw[8][0].float().numpy()  # (256, 768)
    H /= len(DAYS.templates)
    inst = np.stack(inst)  # (56, 768)

    sae = load_sae()
    W_enc, b_enc = sae["W_enc"], sae["b_enc"]
    W_dec, b_dec = sae["W_dec"], sae["b_dec"]

    def acts(X):
        return np.maximum((X - b_dec) @ W_enc + b_enc, 0.0)

    a_day = acts(inst).mean(0)          # (d_sae,)
    a_wiki = acts(Wtok).mean(0)
    sel = a_day - a_wiki
    smax = sel.max()
    chosen = np.nonzero(sel >= 0.5 * smax)[0]
    if len(chosen) > 16:
        chosen = chosen[np.argsort(-sel[chosen])][:16]
    D = W_dec[chosen]                   # (k, 768) decoder rows
    # SAE day subspace top-2 directions (weighted by selectivity)
    Dw = D * sel[chosen][:, None]
    _, _, Vt = np.linalg.svd(Dw - 0.0, full_matrices=False)
    S2 = Vt[:2].T                       # (768, 2)

    BH, rep = array_pca(H, 2)
    Q1 = fourier_plane_basis(H - H.mean(0), 1).T
    cos_BH = principal_cos(S2, BH)
    cos_F = principal_cos(S2, Q1)
    # mass overlap: fraction of the SAE-subspace-projected day cloud variance
    Hc = H - H.mean(0)
    frac_in_S2 = float(((Hc @ S2) ** 2).sum() / (Hc**2).sum())

    # exploratory: circular tuning of the chosen latents across the 7 days
    tuning = acts(H)[:, chosen]         # (7, k)
    from clocklib.fourier import freq_power

    k1_frac = []
    for j in range(tuning.shape[1]):
        pw = freq_power(tuning[:, j:j + 1])
        k1_frac.append(float(pw[1] / max(pw.sum(), 1e-12)))

    res = {
        "n_selected_latents": int(len(chosen)),
        "latent_ids": [int(i) for i in chosen],
        "selectivity": [round(float(sel[i]), 3) for i in chosen],
        "P-S1_cos_SAEsubspace_vs_BH": [round(c, 4) for c in cos_BH],
        "P-S1_pass": bool(np.mean(cos_BH) >= 0.70),
        "P-S2_pass": bool(len(chosen) <= 16),
        "cos_SAEsubspace_vs_fourier_k1": [round(c, 4) for c in cos_F],
        "day_cloud_var_frac_in_SAE_subspace": round(frac_in_S2, 4),
        "BH_top2_share": round(rep["share"], 4),
        "latent_tuning_k1_frac": [round(x, 3) for x in k1_frac],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "correspondence.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
