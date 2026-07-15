"""Step-3 proper: emotion-circumplex manifold (preregistration_emotion.md, c541de4).

Stages: survey (P-E1 structure), attrib (P-E2), sweep (P-E3/P-E4).
Usage: python 24_emotion.py --stage survey --device mps
"""

import argparse
import json
import os
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from clocklib.emotion import ADJECTIVES, CATEGORIES, PROBE_SUFFIX, SCENES
from clocklib.ringlib import DAYS, MONTHS, LM
from clocklib.unsup import array_pca, cloud_plane_power

w18 = import_module("18_writers_any")

MODEL = "meta-llama/Llama-3.2-1B"
OUT = ROOT / "artifacts" / "emotion_llama"
LAMBDAS = w18.LAMBDAS
CATS = list(SCENES.keys())


def scene_items(lm):
    out = []
    for ci, cat in enumerate(CATS):
        for s in SCENES[cat]:
            enc = lm.tok(s, return_tensors="pt").input_ids
            out.append((enc, ci, enc.shape[1] - 1))
    return out


def probe_items(lm):
    out = []
    for ci, cat in enumerate(CATS):
        for s in SCENES[cat]:
            enc = lm.tok(s + PROBE_SUFFIX, return_tensors="pt").input_ids
            out.append((enc, ci))
    return out


@torch.no_grad()
def scene_cloud(wr, items, layer):
    H = np.zeros((len(items), wr.lm.hidden_size))
    for k, (enc, ci, pos) in enumerate(items):
        hs = wr.model(input_ids=enc.to(wr.lm.device),
                      output_hidden_states=True).hidden_states
        H[k] = hs[layer][0, pos].float().cpu().numpy()
    return H


@torch.no_grad()
def behavior(wr, lm, pitems, adj_ids):
    val = {"happy": 1, "excited": 1, "content": 1, "calm": 1,
           "afraid": -1, "angry": -1, "sad": -1, "bored": -1}
    vlist = [val[a] for a in CATS]
    correct = v_ok = 0
    margins = []
    for enc, ci in pitems:
        dl = wr.model(input_ids=enc.to(lm.device)).logits[0, -1][adj_ids].float()
        pred = int(dl.argmax())
        correct += pred == ci
        v_ok += vlist[pred] == vlist[ci]
        margins.append(float(dl[ci] - dl[torch.arange(8) != ci].max()))
    n = len(pitems)
    return correct / n, v_ok / n, float(np.mean(margins))


VAL = {"happy": 1, "excited": 1, "content": 1, "calm": 1,
       "afraid": -1, "angry": -1, "sad": -1, "bored": -1}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    from scipy.stats import mannwhitneyu

    u = mannwhitneyu(pos, neg, alternative="two-sided").statistic
    a = u / (len(pos) * len(neg))
    return float(max(a, 1 - a))


def valence_axis(H: np.ndarray, labels: np.ndarray) -> np.ndarray:
    cent = np.stack([H[labels == ci].mean(0) for ci in range(8)])
    sign = np.array([VAL[c] for c in CATS])
    ax = cent[sign > 0].mean(0) - cent[sign < 0].mean(0)
    return ax / np.linalg.norm(ax)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["survey", "attrib", "sweep", "axis-survey",
                             "axis-sweep"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    lm = LM(MODEL, device=args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    L = lm.probe_layer
    adj_ids = [lm.tok.encode(" " + a, add_special_tokens=False)[0]
               for a in ADJECTIVES]

    if args.stage == "axis-survey":
        wr = w18.Writers(lm)
        items = scene_items(lm)
        labels = np.array([it[1] for it in items])
        vsign = np.array([VAL[CATS[c]] for c in labels])
        NL = lm.n_layers
        Hs = np.zeros((NL + 1, 48, lm.hidden_size))
        with torch.no_grad():
            for k, (enc, ci, pos) in enumerate(items):
                hs = wr.model(input_ids=enc.to(lm.device),
                              output_hidden_states=True).hidden_states
                for l in range(NL + 1):
                    Hs[l, k] = hs[l][0, pos].float().cpu().numpy()
        rows, best = [], (None, 0.0, None)
        for l in range(1, NL + 1):
            ax = valence_axis(Hs[l], labels)
            proj = Hs[l] @ ax
            a = auc(proj[vsign > 0], proj[vsign < 0])
            rows.append({"layer": l, "auc": round(a, 4)})
            if a > best[1]:
                best = (l, a, ax)
        res = {"rows": rows, "best_layer": best[0], "best_auc": round(best[1], 4),
               "P-E1pp_pass": bool(best[1] >= 0.85)}
        (OUT / "axis_survey.json").write_text(json.dumps(res, indent=2))
        np.save(OUT / "axis_refs.npy",
                {"axis": best[2], "layer": best[0]}, allow_pickle=True)
        print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=2))
        print("aucs:", [r["auc"] for r in rows])
        return

    if args.stage == "axis-sweep":
        refs = np.load(OUT / "axis_refs.npy", allow_pickle=True).item()
        ax, Lx = refs["axis"], int(refs["layer"])
        wr = w18.Writers(lm, n_below=Lx, capture_layer=Lx)
        items = scene_items(lm)
        labels = np.array([it[1] for it in items])
        vsign = np.array([VAL[CATS[c]] for c in labels])
        pitems = probe_items(lm)
        ax_t = torch.from_numpy(np.outer(ax, ax)).float()   # rank-1 projector
        W1 = wr.W1
        adj_rows = W1[adj_ids].numpy().astype(np.float64)
        Badj, _ = array_pca(adj_rows, 2)
        P_out = torch.from_numpy(Badj @ Badj.T).float()
        day_tpl = w18.positions(lm, DAYS)
        month_tpl = w18.positions(lm, MONTHS)

        def lcloud(tpl, n, layer):
            H = np.zeros((n, lm.hidden_size))
            with torch.no_grad():
                for enc, wi, pos in tpl:
                    hs = wr.model(input_ids=enc.to(lm.device),
                                  output_hidden_states=True).hidden_states
                    H[wi] += hs[layer][0, pos].float().cpu().numpy()
            return H / (len(tpl) // n)

        Hd = lcloud(day_tpl, 7, Lx)
        BH, _ = array_pca(Hd, 2)
        day_ref = cloud_plane_power(Hd, BH)
        Hm = lcloud(month_tpl, 12, Lx)
        BM, _ = array_pca(Hm, 2)
        month_ref = cloud_plane_power(Hm, BM)
        from datasets import load_dataset

        wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                            split="validation")
        text = "\n".join(t for t in wiki["text"] if t.strip())[:20000]
        wiki_ids = lm.tok(text, return_tensors="pt").input_ids[0][: 8 * 64
                                                                  ].reshape(8, 64).to(lm.device)
        with torch.no_grad():
            ref_logp = F.log_softmax(wr.model(input_ids=wiki_ids).logits.float(),
                                     -1).cpu()

        def measure():
            m = {}
            Hs = scene_cloud(wr, items, Lx)
            proj = Hs @ ax
            m["axis_auc"] = auc(proj[vsign > 0], proj[vsign < 0])
            # capture: the 48 held-out scene projections onto the frozen valence
            # axis, in frozen scene order (clocklib.emotion SCENES; labels live
            # there for the demo builder to join). No per-row labels stored.
            m["_scene_proj"] = [round(float(p), 4) for p in proj]
            m["acc8"], m["valence"], m["margin"] = behavior(wr, lm, pitems, adj_ids)
            m["day_plane_power"] = cloud_plane_power(lcloud(day_tpl, 7, Lx), BH) \
                / max(day_ref, 1e-12)
            m["month_plane_power"] = cloud_plane_power(lcloud(month_tpl, 12, Lx),
                                                       BM) / max(month_ref, 1e-12)
            with torch.no_grad():
                lp = F.log_softmax(wr.model(input_ids=wiki_ids).logits.float(),
                                   -1).cpu()
            m["wiki_kl"] = float(F.kl_div(lp.flatten(0, 1),
                                          ref_logp.flatten(0, 1),
                                          reduction="batchmean", log_target=True))
            return m

        def plane_mats(P):
            return {nm: wr.arch.removed(wr.orig[nm], P) for nm in wr.orig}

        def rand_mats(seed):
            g = torch.Generator().manual_seed(seed)
            res = {}
            for nm in wr.orig:
                v = torch.randn(lm.hidden_size, generator=g)
                v /= v.norm()
                M = wr.arch.removed(wr.orig[nm], torch.outer(v, v))
                tgt = float(wr.arch.removed(wr.orig[nm], ax_t).norm())
                res[nm] = M * (tgt / max(float(M.norm()), 1e-12))
            return res

        conds = {"T_val_all": plane_mats(ax_t), "T_emo_out": plane_mats(P_out),
                 **{f"C_val_rand_r{s}": rand_mats(s) for s in range(2)}}
        # optional targeted re-run: EMO_ONLY="T_val_all,C_val_rand_r0" regenerates
        # only those cells (leaves the others' committed JSONs untouched)
        only = os.environ.get("EMO_ONLY")
        if only:
            keep = set(only.split(","))
            conds = {k: v for k, v in conds.items() if k in keep}
        sweep_dir = OUT / "axis_sweep"
        sweep_dir.mkdir(exist_ok=True)
        for name, mats in conds.items():
            rows = []
            for lam in LAMBDAS:
                wr.set_removals(lam, mats)
                rows.append({"lam": lam, **measure()})
            wr.restore()
            (sweep_dir / f"{name}.json").write_text(json.dumps(rows))
            r0 = rows[LAMBDAS.index(0.0)]
            print(name, "done; lam0:",
                  json.dumps({k: round(v, 3) for k, v in r0.items()
                              if isinstance(v, (int, float))}), flush=True)
        return

    if args.stage == "survey":
        wr = w18.Writers(lm)
        items = scene_items(lm)
        H = scene_cloud(wr, items, L)
        B, rep = array_pca(H, 2)
        Hc = H - H.mean(0)
        coords = Hc @ B                                   # (48, 2)
        cent = np.stack([coords[np.array([it[1] for it in items]) == ci].mean(0)
                         for ci in range(8)])
        vth = [CATEGORIES[c][0] for c in CATS]
        ath = [CATEGORIES[c][1] for c in CATS]
        rho_v = [float(spearmanr(cent[:, j], vth)[0]) for j in (0, 1)]
        rho_a = [float(spearmanr(cent[:, j], ath)[0]) for j in (0, 1)]
        res = {"top2_share": rep["share"],
               "rho_valence_pc12": rho_v, "rho_arousal_pc12": rho_a,
               "P-E1_pass": bool(max(abs(r) for r in rho_v) >= 0.7),
               "centroids": [[round(float(x), 3) for x in row] for row in cent],
               "cats": CATS}
        (OUT / "survey.json").write_text(json.dumps(res, indent=2))
        np.save(OUT / "refs.npy", {"B": B}, allow_pickle=True)
        print(json.dumps({k: v for k, v in res.items() if k != "centroids"},
                         indent=2))

    elif args.stage == "attrib":
        wr = w18.Writers(lm)
        items = scene_items(lm)
        B = np.load(OUT / "refs.npy", allow_pickle=True).item()["B"]
        P = B @ B.T
        caps = wr.clouds(items, len(items))  # per-scene contributions
        total = caps.pop("_total")
        Tc = total - total.mean(0)
        den = float(((Tc @ B) ** 2).sum())
        alphas = {n: float(np.sum(((C - C.mean(0)) @ P) * (Tc @ P)) / den)
                  for n, C in caps.items()}
        attn = sum(abs(v) for k, v in alphas.items() if k.startswith("attn"))
        mlp = sum(abs(v) for k, v in alphas.items() if k.startswith("mlp"))
        top = max((k for k in alphas if k != "emb"), key=lambda k: abs(alphas[k]))
        res = {"alphas": dict(sorted(alphas.items(), key=lambda kv: -abs(kv[1]))),
               "attn_share": attn, "mlp_share": mlp, "top_writer": top,
               "P-E2_pass": any(abs(v) >= 0.05 for k, v in alphas.items()
                                if k != "emb")}
        (OUT / "attribution.json").write_text(json.dumps(res, indent=2))
        print(json.dumps({"top8": dict(list(res["alphas"].items())[:8]),
                          "attn": attn, "mlp": mlp,
                          "P-E2_pass": res["P-E2_pass"]}, indent=2))

    else:  # sweep
        wr = w18.Writers(lm)
        items = scene_items(lm)
        pitems = probe_items(lm)
        B = np.load(OUT / "refs.npy", allow_pickle=True).item()["B"]
        P_emo = torch.from_numpy(B @ B.T).float()
        W1 = wr.W1
        adj_rows = W1[adj_ids].numpy().astype(np.float64)
        Badj, _ = array_pca(adj_rows, 2)
        P_out = torch.from_numpy(Badj @ Badj.T).float()
        day_tpl = w18.positions(lm, DAYS)
        month_tpl = w18.positions(lm, MONTHS)

        def lcloud(tpl, n, layer):
            H = np.zeros((n, lm.hidden_size))
            with torch.no_grad():
                for enc, wi, pos in tpl:
                    hs = wr.model(input_ids=enc.to(lm.device),
                                  output_hidden_states=True).hidden_states
                    H[wi] += hs[layer][0, pos].float().cpu().numpy()
            return H / (len(tpl) // n)

        Hd = lcloud(day_tpl, 7, L)
        BH, _ = array_pca(Hd, 2)
        day_ref = cloud_plane_power(Hd, BH)
        Hm = lcloud(month_tpl, 12, L)
        BM, _ = array_pca(Hm, 2)
        month_ref = cloud_plane_power(Hm, BM)
        emo_ref = cloud_plane_power(scene_cloud(wr, items, L), B)
        from datasets import load_dataset

        wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                            split="validation")
        text = "\n".join(t for t in wiki["text"] if t.strip())[:20000]
        wiki_ids = lm.tok(text, return_tensors="pt").input_ids[0][: 8 * 64
                                                                  ].reshape(8, 64).to(lm.device)
        with torch.no_grad():
            ref_logp = F.log_softmax(wr.model(input_ids=wiki_ids).logits.float(),
                                     -1).cpu()

        def measure():
            m = {}
            Hs = scene_cloud(wr, items, L)
            m["emo_plane_power"] = cloud_plane_power(Hs, B) / max(emo_ref, 1e-12)
            m["_scene_coords"] = [[round(float(x), 4) for x in row]
                                  for row in ((Hs - Hs.mean(0)) @ B)]
            m["acc8"], m["valence"], m["margin"] = behavior(wr, lm, pitems, adj_ids)
            m["day_plane_power"] = cloud_plane_power(lcloud(day_tpl, 7, L), BH) \
                / max(day_ref, 1e-12)
            m["month_plane_power"] = cloud_plane_power(lcloud(month_tpl, 12, L), BM) \
                / max(month_ref, 1e-12)
            with torch.no_grad():
                lp = F.log_softmax(wr.model(input_ids=wiki_ids).logits.float(),
                                   -1).cpu()
            m["wiki_kl"] = float(F.kl_div(lp.flatten(0, 1),
                                          ref_logp.flatten(0, 1),
                                          reduction="batchmean", log_target=True))
            return m

        def plane_mats(P):
            return {nm: wr.arch.removed(wr.orig[nm], P) for nm in wr.orig}

        def rand_mats(seed):
            g = torch.Generator().manual_seed(seed)
            res = {}
            for nm in wr.orig:
                Q, _ = torch.linalg.qr(torch.randn(lm.hidden_size, 2, generator=g))
                M = wr.arch.removed(wr.orig[nm], Q @ Q.T)
                tgt = float(wr.arch.removed(wr.orig[nm], P_emo).norm())
                res[nm] = M * (tgt / max(float(M.norm()), 1e-12))
            return res

        conds = {"T_emo_all": plane_mats(P_emo), "T_emo_out": plane_mats(P_out),
                 **{f"C_emo_rand_r{s}": rand_mats(s) for s in range(2)}}
        sweep_dir = OUT / "sweep"
        sweep_dir.mkdir(exist_ok=True)
        for name, mats in conds.items():
            rows = []
            for lam in LAMBDAS:
                wr.set_removals(lam, mats)
                rows.append({"lam": lam, **measure()})
            wr.restore()
            (sweep_dir / f"{name}.json").write_text(json.dumps(rows))
            r0 = rows[LAMBDAS.index(0.0)]
            print(name, "done; lam0:",
                  json.dumps({k: round(v, 3) for k, v in r0.items()
                              if not k.startswith("_")}), flush=True)


if __name__ == "__main__":
    main()
